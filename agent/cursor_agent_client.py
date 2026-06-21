"""OpenAI-compatible facade that forwards Hermes requests to ``cursor-agent``.

This adapter lets Hermes treat the Cursor Agent CLI as a chat-style backend so
every Cursor user (Hobby/Pro/Pro+/Ultra/Teams) can route Hermes calls through
their existing Cursor subscription / API credits.

Per request we spawn ``cursor-agent -p`` with ``--output-format stream-json``,
pass the formatted conversation as the prompt (via stdin to avoid the argv
length limit), then parse the line-delimited JSON events into a single OpenAI
chat-completion response.

Design notes:

- One subprocess per request (no shared long-running session). Warm sessions
  via ``--resume`` are an opt-in path documented below.
- Default ``--mode ask`` keeps Cursor read-only — useful when we just want the
  model as an LLM rather than letting it edit files.
- Default workspace is an ephemeral temp dir so the agent never sees the
  caller's repo. Override via ``HERMES_CURSOR_WORKSPACE`` or the ``workspace``
  ctor arg.
- Tool calls follow the Copilot-ACP convention: tools are described in the
  system prompt and the model emits ``<tool_call>{...}</tool_call>`` blocks
  that we lift back into OpenAI ``tool_calls``.
- The CLI auth (``cursor-agent login`` or ``CURSOR_API_KEY``) is what governs
  identity; we forward ``CURSOR_API_KEY`` to the subprocess and let the CLI
  resolve it (same as the IDE does).
"""

from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import tempfile
import threading
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from agent.redact import redact_sensitive_text
from providers.cursor_utils import DEFAULT_CURSOR_COMMAND, resolve_cursor_command, resolve_cursor_extra_args

CURSOR_MARKER_BASE_URL = "cursor://agent"
# ``agent`` matches cursor-agent's own default permissionMode (the
# behavior you get from ``cursor-agent -p`` with no ``--mode`` flag).
# This is what a user picking ``cursor`` in ``hermes model`` will expect:
# the same write/edit/shell power they'd have from the cursor CLI directly.
# Users who want read-only behavior set ``HERMES_CURSOR_MODE=ask`` (or
# ``plan``); Hermes' own ``approvals.mode`` config additionally gates any
# tool execution (manual / smart / off) on top of this, identical to every
# other provider.
DEFAULT_CURSOR_MODE = "agent"
DEFAULT_CURSOR_MODEL = "auto"

# cursor-agent CLI accepts only ``ask`` and ``plan`` for ``--mode`` today,
# but ``-p/--print`` *without* ``--mode`` runs in the full-capability
# ``default`` permissionMode (write+shell+everything). We expose that as
# the synthetic ``agent`` value here:
#   - ``ask``   : read-only Q&A. Cursor's built-in mutation tools are
#                 disabled. Hermes-side tools still apply for any work
#                 that needs to touch the user's disk.
#   - ``plan`` : read-only planning mode. Produces structured plan output
#                from Cursor's planner.
#   - ``agent``: omits ``--mode`` so Cursor runs in its IDE-equivalent
#                "default" permissionMode — built-in shell, write, edit,
#                read, etc. all active. Use this when you want Cursor to
#                drive multi-step work end-to-end (it will still emit
#                tool_call events that we surface to Hermes UI).
# Anything else falls back to ``ask``. Don't add new values without
# re-checking ``cursor-agent --help``; passing an unknown ``--mode``
# value causes a hard-crash BrokenPipe with a confusing
# "Allowed choices are plan, ask." stderr.
_VALID_CURSOR_MODES = frozenset({"ask", "plan", "agent"})
_CURSOR_CLI_MODES = frozenset({"ask", "plan"})
# Idle threshold (not wall-clock): the deadline resets on every stream-json
# event from cursor-agent. A turn can legitimately run for much longer than
# this in total wall-clock; what matters is that events keep arriving. If
# nothing arrives for this long, the subprocess is presumed hung and is
# force-killed with a clear TimeoutError. Override via
# ``HERMES_CURSOR_TIMEOUT_SECONDS`` env var. Default is 30 minutes; cursor-
# agent's own internal shell ceiling is 10 minutes so a single shell call
# can chew up that much idle time, and chained internal operations (deep
# greps, large reads after a long shell) routinely push past 15 minutes
# without emitting events. 30 min gives comfortable headroom while still
# catching genuine hangs.
_DEFAULT_TIMEOUT_SECONDS = 1800.0

# Sentinels that mean "no real api key — use the cursor-agent CLI's own login
# session". Hermes's external_process auth path injects these as placeholders;
# forwarding them to ``cursor-agent --api-key`` makes the CLI reject the
# request and close stdin, manifesting upstream as ``BrokenPipeError``.
_API_KEY_SENTINELS = frozenset({
    "",
    "cursor-agent-login",
    "cursor-cli-login",
    "external-process",
    "external_process",
})

# Reuse the tool-call extraction grammar from copilot_acp_client.  We do NOT
# reuse its prompt builder — cursor's model is itself an agentic CLI with its
# own built-in shell/edit/read tools, and the softer ACP wording ("ACP agent
# backend, use ACP capabilities") makes it prefer those built-ins and run the
# work internally, leaving Hermes' tool surface unused.  Cursor needs an
# explicit "you are JUST the LLM, do not execute anything yourself" directive
# (see ``_format_messages_as_prompt`` below).
import json as _json  # noqa: E402

from agent.copilot_acp_client import (  # noqa: E402
    _extract_tool_calls_from_text,
    _render_message_content,
)


def _format_messages_as_prompt(
    messages: list[dict[str, Any]],
    model: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: Any = None,
) -> str:
    """Build the prompt sent to ``cursor-agent`` stdin.

    Key differences vs. the copilot-acp formatter:

    * Hard "you are the LLM, NOT an agent" framing — without this, cursor's
      built-in shell/edit/read tools intercept the request and the agentic
      loop runs entirely inside ``cursor-agent``, so Hermes never sees a
      ``tool_calls`` response (the symptom: chat sessions show 0 tool
      calls even though tools are advertised).
    * Explicit "do NOT run ls/cat/edit yourself" line — empirically required
      to push cursor's model past its default "I'll just do it" reflex.
    * Tool-call grammar identical to copilot-acp so the ``<tool_call>{...}
      </tool_call>`` extractor we share keeps working.
    """
    # Auxiliary-style calls (title generation, compression, vision,
    # mcp router, etc.) come in with NO ``tools`` and just a system+user
    # pair. They want a short, direct response — slapping the full
    # "you are an agent backend, emit tool_call blocks" preamble on top
    # of them makes cursor's harness reply with a verbose multi-paragraph
    # answer or even crash on the formatting constraints. So we keep
    # the heavy preamble only for the agentic chat path (tools provided).
    sections: list[str] = []
    has_tools = bool(tools)
    if has_tools:
        sections.extend([
            "You are powering a chat session inside Hermes Agent.",
            "You have TWO sets of tools available:",
            "(A) Your own built-in cursor-agent tools (shell, read_file, "
            "edit_file, write_file, list_directory, grep, glob, web_fetch). "
            "Use these DIRECTLY for filesystem/shell/search work — they run "
            "on the real workspace, are fast, and Hermes will surface their "
            "results to the user automatically.",
            "(B) Hermes-side tools listed in the schema below. They cover "
            "capabilities your built-in tools do NOT have (skills, MCP "
            "servers, browser automation, remote APIs, etc.). To invoke "
            "one of THESE, emit a "
            "<tool_call>{...}</tool_call> block in OpenAI function-call "
            "shape: "
            '{"id":"call_<n>","type":"function",'
            '"function":{"name":"<tool>","arguments":"<json string>"}}. '
            "``arguments`` MUST be a JSON STRING (escaped), not a nested "
            "object.",
            "RULES:",
            "1. Prefer your built-in tools for any shell command, file "
            "read/write/list/edit, grep, or glob operation — they're "
            "faster than round-tripping through Hermes. CRITICAL for "
            "file creation/modification: ALWAYS use the ``write`` or "
            "``edit`` built-in tools, NEVER ``shell`` with ``echo > "
            "file`` / ``cat > file`` / ``sed -i`` / ``>>``. Only the "
            "write/edit tools report ``linesAdded`` / ``linesRemoved`` "
            "/ ``diffString`` to the harness, which is what Hermes "
            "renders as the colored ``+``/``-`` diff in the UI. Shell "
            "redirections create the file but the user sees no diff "
            "and has no idea what changed.",
            "2. Only emit <tool_call> blocks for tools listed in the "
            "schema below; do NOT invent tool names. Multiple tool_calls "
            "per turn are allowed.",
            "3. Work iteratively (ReAct-style): before each tool batch, "
            "emit ONE short line of plain text saying what you're about "
            "to check and why. After tool results come back, briefly "
            "reflect on what you found before deciding the next step. "
            "Hermes surfaces these intermediate lines to the user as "
            "live narration so they can follow your reasoning.",
            "4. Don't dump every tool call upfront — chain them: think, "
            "tool, reflect, tool, reflect, ... then synthesise the final "
            "answer at the end. If the task genuinely is independent "
            "lookups, parallel tool calls in one batch are fine.",
            "5. If no tool is needed (pure conversation, math, "
            "summarising content already in the transcript), answer as "
            "plain text.",
            "6. Never hallucinate file contents or command output — if "
            "you say \"Reading the file…\" you MUST actually run the "
            "read_file (built-in) or emit a <tool_call> if it's a "
            "Hermes-specific tool.",
            "7. The Hermes UI already shows file edits to the user as a "
            "colored +/- diff right next to each ``edit`` / ``write`` "
            "tool call (and tool calls + diffs are streamed live). Do "
            "NOT re-dump the before/after content or paste the diff "
            "again in your final response — just confirm what was "
            "changed at a high level (e.g. \"updated foo.py to fix the "
            "off-by-one\"). Same for shell output: it's already visible.",
        ])
    else:
        # Lite preamble for aux calls — just enough to keep cursor's
        # harness from running its own tools / writing files / asking
        # clarifying questions when all we want is a single short reply.
        sections.append(
            "Hermes auxiliary call. Answer the user message below directly "
            "and concisely; do not run any tools, do not write files, do "
            "not ask follow-up questions. Plain-text reply only."
        )
    if model:
        sections.append(f"Hermes requested model hint: {model}")

    if isinstance(tools, list) and tools:
        tool_specs: list[dict[str, Any]] = []
        for t in tools:
            if not isinstance(t, dict):
                continue
            fn = t.get("function") or {}
            if not isinstance(fn, dict):
                continue
            name = fn.get("name")
            if not isinstance(name, str) or not name.strip():
                continue
            tool_specs.append(
                {
                    "name": name.strip(),
                    "description": fn.get("description", ""),
                    "parameters": fn.get("parameters", {}),
                }
            )
        if tool_specs:
            sections.append(
                "Hermes-side tools (OpenAI function schema). Emit "
                "<tool_call>{...}</tool_call> blocks to invoke these. "
                "For plain shell / file / grep / glob actions prefer your "
                "own built-in tools instead (they're faster).\n"
                + _json.dumps(tool_specs, ensure_ascii=False)
            )

    if tool_choice is not None:
        sections.append(
            f"Tool choice hint: {_json.dumps(tool_choice, ensure_ascii=False)}"
        )

    transcript: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "unknown").strip().lower()
        if role == "tool":
            role = "tool"
        elif role not in {"system", "user", "assistant"}:
            role = "context"

        content = message.get("content")
        rendered = _render_message_content(content)
        if not rendered:
            continue

        label = {
            "system": "System",
            "user": "User",
            "assistant": "Assistant",
            "tool": "Tool",
            "context": "Context",
        }.get(role, role.title())
        transcript.append(f"{label}:\n{rendered}")

    if transcript:
        sections.append("Conversation transcript:\n\n" + "\n\n".join(transcript))

    sections.append("Continue the conversation from the latest user request.")
    return "\n\n".join(section.strip() for section in sections if section and section.strip())


# ---------------------------------------------------------------------------
# Environment & path helpers
# ---------------------------------------------------------------------------


def _resolve_command() -> str:
    return resolve_cursor_command()


def _resolve_extra_args() -> list[str]:
    return resolve_cursor_extra_args()


def _resolve_mode() -> str:
    mode = os.getenv("HERMES_CURSOR_MODE", "").strip().lower() or DEFAULT_CURSOR_MODE
    if mode not in _VALID_CURSOR_MODES:
        mode = DEFAULT_CURSOR_MODE
    return mode


def _resolve_workspace_override() -> str:
    return os.getenv("HERMES_CURSOR_WORKSPACE", "").strip()


def _resolve_home_dir() -> str:
    """Pick a stable HOME for the child process.

    Mirrors ``agent/copilot_acp_client.py:_resolve_home_dir`` so subprocess
    behaviour stays predictable across providers.
    """
    try:
        from hermes_constants import get_subprocess_home

        profile_home = get_subprocess_home()
        if profile_home:
            return profile_home
    except Exception:
        pass

    home = os.environ.get("HOME", "").strip()
    if home:
        return home

    expanded = os.path.expanduser("~")
    if expanded and expanded != "~":
        return expanded

    # POSIX-only last resort: read the home dir from the password
    # database. ``os.getuid`` does not exist on Windows; gate explicitly
    # so the import-time footgun checker stays clean. (Windows already
    # falls through to the ``USERPROFILE`` / ``HOMEDRIVE+HOMEPATH``
    # branches above; if those failed there is no equivalent password
    # database here, so just bail to ``/tmp``.)
    if hasattr(os, "getuid"):
        try:
            import pwd

            resolved = pwd.getpwuid(os.getuid()).pw_dir.strip()  # windows-footgun: ok
            if resolved:
                return resolved
        except Exception:
            pass

    return "/tmp"


def _build_subprocess_env(api_key: str | None) -> dict[str, str]:
    env = os.environ.copy()
    env["HOME"] = _resolve_home_dir()
    if api_key:
        env["CURSOR_API_KEY"] = api_key
    env.setdefault("NO_COLOR", "1")
    env.setdefault("TERM", "dumb")
    return env


def _cursor_cli_auth_error(command: str, detail: str = "") -> RuntimeError:
    clean_detail = redact_sensitive_text(detail.strip(), force=True) if detail else ""
    suffix = f" Details: {clean_detail}" if clean_detail else ""
    return RuntimeError(
        f"Cursor CLI is not authenticated for '{command}'. "
        "Run `cursor-agent login` again, or set `CURSOR_API_KEY` for "
        f"long-running Hermes/gateway use.{suffix}"
    )


# ---------------------------------------------------------------------------
# Stream-json parser
# ---------------------------------------------------------------------------


def _build_cursor_tool_preview(evt: "_CursorToolEvent") -> str:
    """Compact one-line description of a cursor tool call for the UI.

    Mirrors the spirit of ``_build_tool_preview`` in ``tool_executor.py`` —
    a single short string the spinner / activity feed can show next to
    the tool name. Tool-specific extractors fall back to a JSON dump of
    arguments when we don't have a hand-written formatter.
    """
    args = evt.args or {}
    try:
        if evt.envelope_key == "shellToolCall":
            cmd = args.get("command") or args.get("cmd")
            if isinstance(cmd, list):
                cmd = " ".join(str(part) for part in cmd)
            if isinstance(cmd, str) and cmd.strip():
                return cmd.strip()[:200]
        if evt.envelope_key in (
            "readToolCall",
            "editToolCall",
            "writeToolCall",
            "patchToolCall",
            "deleteToolCall",
        ):
            # Cursor's wire format isn't fully consistent across tool
            # kinds; ``editToolCall.args`` has been seen using
            # ``target_file`` / ``targetFile`` / ``file_path`` while
            # other tools use ``path``. Try them all so the activity
            # feed always shows what was touched.
            path = (
                args.get("path")
                or args.get("file")
                or args.get("filePath")
                or args.get("filename")
                or args.get("target_file")
                or args.get("targetFile")
                or args.get("file_path")
                or args.get("relative_workspace_path")
                or ""
            )
            if isinstance(path, str) and path.strip():
                return path.strip()[:200]
        if evt.envelope_key == "globToolCall":
            pat = args.get("globPattern") or args.get("pattern") or ""
            target = args.get("targetDirectory") or args.get("path") or ""
            label = " in ".join(p for p in (pat, target) if isinstance(p, str) and p.strip())
            if label:
                return label[:200]
        if evt.envelope_key in ("grepToolCall", "searchToolCall"):
            pat = args.get("pattern") or args.get("query") or args.get("regex") or ""
            target = args.get("path") or args.get("targetDirectory") or ""
            if isinstance(pat, str) and pat.strip():
                if isinstance(target, str) and target.strip():
                    return f"{pat} in {target}"[:200]
                return pat.strip()[:200]
            if isinstance(target, str) and target.strip():
                return target.strip()[:200]
        if evt.envelope_key == "listToolCall":
            path = args.get("path") or args.get("directory") or args.get("targetDirectory") or ""
            if isinstance(path, str) and path.strip():
                return path.strip()[:200]
        return json.dumps(args, ensure_ascii=False)[:200]
    except Exception:
        return ""


def _normalize_cursor_tool_name(envelope_key: str) -> str:
    """Map cursor's wire-format ``<thing>ToolCall`` keys to Hermes tool names.

    cursor-agent's stream-json wraps every internal tool call as
    ``"<kind>ToolCall"`` (e.g. ``shellToolCall``, ``readToolCall``). We
    translate the kind so the activity surfaces in Hermes' UI with names
    the user already recognises from other providers.
    """
    if not isinstance(envelope_key, str):
        return "cursor_tool"
    suffix = "ToolCall"
    base = envelope_key[: -len(suffix)] if envelope_key.endswith(suffix) else envelope_key
    if not base:
        return "cursor_tool"
    return {
        "shell": "shell",
        "read": "read_file",
        "list": "list_directory",
        "edit": "edit_file",
        "write": "write_file",
        "patch": "patch",
        "grep": "grep",
        "glob": "glob",
        "search": "search",
        "todo": "todo",
        "delete": "delete_file",
        "task": "task",
        "fetch": "web_fetch",
    }.get(base.lower(), base)


def _summarise_cursor_tool_result(envelope_key: str, payload: dict[str, Any]) -> str:
    """Return a compact human-readable result string for the UI / log.

    Falls back to a generic JSON dump when we don't have a hand-written
    extractor for the tool kind. Best-effort — never raises.
    """
    result = payload.get("result")
    if not isinstance(result, dict):
        return ""
    success = result.get("success")
    if not isinstance(success, dict):
        if "error" in result and isinstance(result["error"], (str, dict)):
            return f"error: {result['error']}"[:400]
        return ""
    try:
        if envelope_key == "shellToolCall":
            stdout = success.get("stdout") or ""
            return stdout if isinstance(stdout, str) else json.dumps(stdout)
        if envelope_key == "readToolCall":
            content = success.get("content") or ""
            total = success.get("totalLines")
            if total is not None:
                return f"({total} lines)\n{content}" if content else f"({total} lines)"
            return content if isinstance(content, str) else json.dumps(content)
        if envelope_key in ("listToolCall", "globToolCall"):
            files = success.get("files") or success.get("entries") or []
            if isinstance(files, list):
                return "\n".join(str(f) for f in files[:200])
        return json.dumps(success, ensure_ascii=False)[:1000]
    except Exception:
        return ""


class _CursorToolEvent:
    """A captured cursor-agent tool invocation (started + completed states).

    Used both for live progress callbacks (Hermes' ``tool_progress_callback``
    surface) and for the post-hoc audit list returned alongside the
    response so sessions can persist what cursor did.
    """

    __slots__ = (
        "call_id", "envelope_key", "name", "args", "started_at",
        "completed_at", "result_text", "is_error", "duration_ms",
        "lines_added", "lines_removed", "diff_string",
    )

    def __init__(self, call_id: str, envelope_key: str, args: dict[str, Any]) -> None:
        self.call_id = call_id
        self.envelope_key = envelope_key
        self.name = _normalize_cursor_tool_name(envelope_key)
        self.args = args
        self.started_at = time.monotonic()
        self.completed_at: float | None = None
        self.result_text: str = ""
        self.is_error: bool = False
        self.duration_ms: int = 0
        # Edit/write result metadata. Cursor's stream-json provides
        # ``linesAdded`` / ``linesRemoved`` / ``diffString`` on the
        # completion event for edit and write operations. We surface
        # the count in the activity feed ("+5 -2") and persist the
        # diff for replays / audits.
        self.lines_added: int | None = None
        self.lines_removed: int | None = None
        self.diff_string: str = ""

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "id": self.call_id,
            "name": self.name,
            "envelope": self.envelope_key,
            "arguments": self.args,
            "result": self.result_text,
            "is_error": self.is_error,
            "duration_ms": self.duration_ms,
        }


class _StreamJsonAccumulator:
    """Accumulates state from a ``cursor-agent --output-format stream-json`` stream.

    Caller feeds parsed JSON events with :meth:`feed`. When a terminal
    ``result`` event arrives the accumulator stores the success/failure state
    and surface text. The instance is reusable per-call but not thread-safe.
    """

    def __init__(self, on_tool_event: Any = None, on_text_event: Any = None) -> None:
        self.text_parts: list[str] = []
        self.reasoning_parts: list[str] = []
        self.session_id: str = ""
        self.request_id: str = ""
        self.model_label: str = ""
        self.duration_ms: int = 0
        self.usage: dict[str, int] = {}
        self.terminal: bool = False
        self.is_error: bool = False
        self.error_message: str = ""
        self.final_result_text: str = ""
        # Ordered transcript of (kind, payload) events as cursor emitted
        # them — used to separate "narrative text between tools" from
        # "final synthesis text" when we assemble the response. Without
        # this, ``assembled_text()`` glues every intermediate text event
        # to the end of the final answer and the user sees a wall of
        # planning prose preceding the actual response.
        self.event_log: list[tuple[str, Any]] = []
        # ``on_tool_event(stage, event)`` — invoked synchronously from
        # ``feed()`` when a ``tool_call`` event arrives.
        self._on_tool_event = on_tool_event
        # ``on_text_event(text)`` — invoked when cursor emits an
        # intermediate ``assistant`` text block (cursor often prints a
        # 1-2 sentence "let me check X next" between tool batches).
        # Surfacing these live as narration events gives the Hermes UI
        # the agentic feel of "tool → text → tool → text" that the user
        # asked about; without it everything bundles into one final
        # answer block.
        self._on_text_event = on_text_event
        self._tool_events: dict[str, _CursorToolEvent] = {}
        self.tool_events: list[_CursorToolEvent] = []
        # Optional caller-provided estimate of "current prompt size" in
        # tokens, used to surface a stable number on the Hermes status
        # bar. Set by ``_create_chat_completion`` before each call.
        self.messages_estimate: int = 0
        # We BUFFER intermediate text instead of dispatching it eagerly
        # so the final text-after-last-tool only appears in the
        # synthesis (assistant response) and not duplicated as a narrate
        # event in the activity feed. Flush rule: when the next tool
        # starts we now know the buffered text was "between tools" and
        # safe to surface. If no more tools come, the buffer is dropped
        # and only ``synthesis_text()`` shows it.
        self._pending_text: list[str] = []

    def feed(self, event: dict[str, Any]) -> None:
        evt_type = event.get("type")
        if not isinstance(evt_type, str):
            return

        if evt_type == "system":
            model = event.get("model")
            if isinstance(model, str):
                self.model_label = model
            session = event.get("session_id")
            if isinstance(session, str):
                self.session_id = session
            return

        if evt_type == "thinking":
            text = event.get("text")
            if isinstance(text, str) and text:
                self.reasoning_parts.append(text)
            return

        if evt_type == "assistant":
            message = event.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, list):
                    for block in content:
                        if not isinstance(block, dict):
                            continue
                        if block.get("type") == "text":
                            text = block.get("text")
                            if isinstance(text, str) and text:
                                self.text_parts.append(text)
                                self.event_log.append(("text", text))
                                # Defer the narrate dispatch — see
                                # ``_pending_text`` docstring above.
                                self._pending_text.append(text)
            return

        if evt_type == "tool_call":
            sub = event.get("subtype")
            # Before recording a NEW tool start, flush any text we'd
            # buffered: by definition that text was "between tools",
            # so it's safe (and useful) to show as narration now.
            if sub == "started" and self._pending_text:
                for buffered in self._pending_text:
                    self._dispatch_text_event(buffered)
                self._pending_text.clear()
            self._consume_tool_call_event(event)
            # Note the tool event order so ``synthesis_text`` can pick
            # the right "final" text. We append only on ``started`` to
            # avoid double-counting; the per-tool event timeline is
            # already preserved in ``self.tool_events``.
            if sub == "started":
                self.event_log.append(("tool", None))
            return

        if evt_type == "result":
            self.terminal = True
            self.is_error = bool(event.get("is_error", False))
            subtype = event.get("subtype")
            if subtype == "error":
                self.is_error = True
            duration = event.get("duration_ms")
            if isinstance(duration, int):
                self.duration_ms = duration
            request = event.get("request_id")
            if isinstance(request, str):
                self.request_id = request
            usage = event.get("usage")
            if isinstance(usage, dict):
                # Cursor emits camelCase keys.
                normalized = {}
                for k, v in usage.items():
                    if isinstance(v, (int, float)):
                        normalized[str(k)] = int(v)
                self.usage = normalized
            result_text = event.get("result")
            if isinstance(result_text, str):
                self.final_result_text = result_text
                if not self.text_parts and not self.is_error:
                    self.text_parts.append(result_text)
            if self.is_error and not self.error_message:
                self.error_message = result_text or "cursor-agent returned an error"
            return

        # Unknown / informational events (e.g. ``user`` echo) — ignore.

    def _consume_tool_call_event(self, event: dict[str, Any]) -> None:
        """Translate one cursor stream-json ``tool_call`` event.

        cursor-agent emits one event with ``subtype="started"`` when the LLM
        decides to use one of its built-in tools (shell, read, edit, ...),
        and a follow-up with ``subtype="completed"`` carrying the result.
        We rebuild a ``_CursorToolEvent`` from those, fire the optional
        progress callback so Hermes' UI can show the activity in real time,
        and stash the final list so the caller can surface "what cursor
        actually did" in the response (e.g. for session audit).
        """
        subtype = event.get("subtype")
        call_id = event.get("call_id")
        if not isinstance(call_id, str) or not call_id:
            return
        tool_call = event.get("tool_call")
        if not isinstance(tool_call, dict) or not tool_call:
            return
        envelope_key = next(iter(tool_call.keys()), "")
        payload = tool_call.get(envelope_key) if isinstance(envelope_key, str) else None
        if not isinstance(payload, dict):
            return
        args_obj = payload.get("args")
        if not isinstance(args_obj, dict):
            args_obj = {}

        if subtype == "started":
            evt = _CursorToolEvent(
                call_id=call_id,
                envelope_key=envelope_key,
                args=args_obj,
            )
            self._tool_events[call_id] = evt
            self.tool_events.append(evt)
            self._fire_tool_event("started", evt)
            return

        if subtype == "completed":
            evt = self._tool_events.get(call_id)
            if evt is None:
                # Cursor sent a completed event we never saw started for —
                # synthesise the started state so the audit list still has it.
                evt = _CursorToolEvent(
                    call_id=call_id,
                    envelope_key=envelope_key,
                    args=args_obj,
                )
                self._tool_events[call_id] = evt
                self.tool_events.append(evt)
                self._fire_tool_event("started", evt)
            evt.completed_at = time.monotonic()
            evt.duration_ms = int((evt.completed_at - evt.started_at) * 1000)
            result = payload.get("result")
            if isinstance(result, dict):
                if "error" in result and result.get("error"):
                    evt.is_error = True
                # Pull diff stats off edit/write/patch completion events
                # so the activity feed can show "+5 -2" next to the
                # path. Cursor only emits these for file-modifying tools.
                success = result.get("success") if isinstance(result, dict) else None
                if isinstance(success, dict):
                    la = success.get("linesAdded")
                    lr = success.get("linesRemoved")
                    ds = success.get("diffString")
                    if isinstance(la, int):
                        evt.lines_added = la
                    if isinstance(lr, int):
                        evt.lines_removed = lr
                    if isinstance(ds, str):
                        evt.diff_string = ds
            evt.result_text = _summarise_cursor_tool_result(envelope_key, payload)
            self._fire_tool_event("completed", evt)
            return

    def _fire_tool_event(self, stage: str, evt: _CursorToolEvent) -> None:
        if self._on_tool_event is None:
            return
        try:
            self._on_tool_event(stage, evt)
        except Exception:
            # A broken UI must never bring down the chat call.
            pass

    def _dispatch_text_event(self, text: str) -> None:
        """Forward an intermediate assistant text event to the UI bridge.

        Errors are swallowed — a broken callback must never abort the
        chat call.
        """
        if self._on_text_event is None:
            return
        try:
            self._on_text_event(text)
        except Exception:
            pass

    def assembled_text(self) -> str:
        return "".join(self.text_parts).strip()

    def synthesis_text(self) -> str:
        """Return only the synthesis portion of the response.

        Cursor's stream interleaves planning prose ("Searching the
        agent directory…") with tool calls, then ends with the actual
        synthesised answer. Gluing every text event together leaves
        the user staring at a wall of "I'll do X next" lines before
        the real answer. This helper returns just the text emitted
        AFTER the last tool call — that's the synthesis.

        Falls back to the full ``assembled_text()`` when:
          * no tools ran (every text is part of the answer);
          * cursor emitted no text after the last tool (rare; we then
            use the cursor-supplied ``result.result`` if it differs
            from the bundled text, otherwise the full bundle so the
            user sees *something*).
        """
        tool_seen = False
        synth: list[str] = []
        for kind, payload in self.event_log:
            if kind == "tool":
                tool_seen = True
                synth.clear()  # drop earlier planning text
            elif kind == "text":
                synth.append(payload)
        if synth:
            return "".join(synth).strip()
        # No text after the last tool. If cursor's ``result.result``
        # carries something useful and distinct, use it; otherwise
        # surface the full bundle so the user isn't left empty-handed.
        if not tool_seen:
            return self.assembled_text()
        if self.final_result_text and self.final_result_text.strip():
            return self.final_result_text.strip()
        return self.assembled_text()

    def narration_text(self) -> str:
        """Return the planning / between-tool prose for transcript replay.

        The live bridge already surfaces each piece individually via
        ``on_text_event``. This helper is for tests / debug consumers
        that want to inspect what was intermediate vs. final.
        """
        narration: list[str] = []
        bucket: list[str] = []
        for kind, payload in self.event_log:
            if kind == "tool":
                if bucket:
                    narration.append("".join(bucket).strip())
                bucket = []
            elif kind == "text":
                bucket.append(payload)
        # The final ``bucket`` is the synthesis — drop it.
        return "\n".join(n for n in narration if n)

    def assembled_reasoning(self) -> str:
        return "".join(self.reasoning_parts).strip()

    def openai_usage(self) -> SimpleNamespace:
        """Translate cursor-agent's per-turn usage into OpenAI-shaped fields.

        Quirk worth knowing: cursor-agent's ``result.usage.inputTokens``
        is the SUM of fresh (non-cached) input tokens across **every
        internal LLM round-trip** in the turn. For an agentic turn that
        runs N tool calls there are roughly N+1 internal model calls
        (one per tool round plus the final text), and each call's input
        grows as tool results accumulate. So inputTokens for a deep
        multi-tool turn can easily reach 1M+ while the model's actual
        context window (e.g. 200K on composer-2.5-fast) was never
        exceeded — cursor reused the cache between calls.

        Hermes' status bar and compressor use ``prompt_tokens`` as a
        proxy for "what's currently in the model's context" (used to
        drive compression decisions and the % bar). Reporting the raw
        cumulative SUM blows the bar past 100% on agentic turns, which
        is both visually wrong and triggers spurious compression.

        Fix: divide the cumulative figures by the number of internal
        rounds we observed (tool_events + 1) to produce an honest
        per-round average that matches the model's actual context use.
        The full billing total is still reported separately for cost
        tracking via ``session_input_tokens``.
        """
        input_tokens_raw = int(self.usage.get("inputTokens", 0))
        output_tokens = int(self.usage.get("outputTokens", 0))
        cache_read_raw = int(self.usage.get("cacheReadTokens", 0))

        rounds = max(len(self.tool_events) + 1, 1)
        per_round_input = input_tokens_raw // rounds if rounds > 0 else input_tokens_raw
        per_round_cache = cache_read_raw // rounds if rounds > 0 else cache_read_raw
        approx_context_tokens = per_round_cache + per_round_input

        # Hermes' messages-based estimate is the canonical "what's in
        # the model's context right now" number (it matches what the
        # next-turn prompt will look like). Prefer it for
        # ``prompt_tokens`` so the status bar stays consistent before,
        # during, and after generation. Fall back to the per-round
        # average when no estimate is set (e.g. accumulator used outside
        # the client, in unit tests).
        if self.messages_estimate > 0:
            prompt_tokens = self.messages_estimate
        else:
            prompt_tokens = approx_context_tokens

        return SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=output_tokens,
            total_tokens=prompt_tokens + output_tokens,
            prompt_tokens_details=SimpleNamespace(cached_tokens=per_round_cache),
            # Preserve raw cursor-side totals for billing / cost tracking
            # consumers that need the actual usage figures.
            cursor_raw_input_tokens=input_tokens_raw,
            cursor_raw_cache_read_tokens=cache_read_raw,
            cursor_internal_rounds=rounds,
            cursor_per_round_context=approx_context_tokens,
        )


# ---------------------------------------------------------------------------
# Inline OpenAI-style namespace shims (mirror copilot_acp_client style)
# ---------------------------------------------------------------------------


class _CursorChatCompletions:
    def __init__(self, client: "CursorAgentClient"):
        self._client = client

    def create(self, **kwargs: Any) -> Any:
        # ``cursor-agent`` exposes streaming via stream-json on its own stdout,
        # but the synchronous ``_create_chat_completion`` already accumulates
        # the full response.  If a caller passes ``stream=True`` we synthesise
        # an OpenAI-style chunk iterator from the final response so the
        # streaming hot path stays iterable.  Without this, iterating the
        # ``SimpleNamespace`` we return surfaces as ``TypeError:
        # 'types.SimpleNamespace' object is not iterable`` (Hermes' chat
        # streaming loop did this).
        stream_requested = bool(kwargs.pop("stream", False))
        kwargs.pop("stream_options", None)  # OpenAI SDK extras — irrelevant
        response = self._client._create_chat_completion(**kwargs)
        if not stream_requested:
            return response
        return _synthesise_stream_chunks(response)


class _CursorChatNamespace:
    def __init__(self, client: "CursorAgentClient"):
        self.completions = _CursorChatCompletions(client)


def _synthesise_stream_chunks(response: Any):
    """Yield OpenAI-style streaming chunks from a non-streaming response.

    Hermes' chat streaming loop expects ``for chunk in stream:`` with each
    chunk shaped like an OpenAI ``ChatCompletionChunk``: ``chunk.choices[0]
    .delta.{content,tool_calls,reasoning,reasoning_content}`` and a final
    chunk carrying ``usage``.  We can't truly stream from the underlying
    subprocess at this layer, but we can split the assembled response into a
    small number of chunks that the loop will accept without crashing.
    """
    try:
        choice = response.choices[0]
    except Exception:
        return

    message = getattr(choice, "message", None)
    if message is None:
        return

    role = "assistant"
    content = getattr(message, "content", "") or ""
    tool_calls = getattr(message, "tool_calls", None) or []
    reasoning = getattr(message, "reasoning", None)
    reasoning_content = getattr(message, "reasoning_content", None)
    finish_reason = getattr(choice, "finish_reason", "stop")
    model = getattr(response, "model", "cursor")
    usage = getattr(response, "usage", None)

    if reasoning_content:
        yield SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        role=role,
                        content=None,
                        tool_calls=None,
                        reasoning=None,
                        reasoning_content=reasoning_content,
                    ),
                    finish_reason=None,
                    index=0,
                )
            ],
            model=model,
            usage=None,
        )
    elif reasoning:
        yield SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        role=role,
                        content=None,
                        tool_calls=None,
                        reasoning=reasoning,
                        reasoning_content=None,
                    ),
                    finish_reason=None,
                    index=0,
                )
            ],
            model=model,
            usage=None,
        )

    if content:
        yield SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        role=role,
                        content=content,
                        tool_calls=None,
                        reasoning=None,
                        reasoning_content=None,
                    ),
                    finish_reason=None,
                    index=0,
                )
            ],
            model=model,
            usage=None,
        )

    if tool_calls:
        # Hermes expects streaming tool_calls to include a per-chunk index.
        for i, tc in enumerate(tool_calls):
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            role=role,
                            content=None,
                            tool_calls=[
                                SimpleNamespace(
                                    index=i,
                                    id=getattr(tc, "id", f"call_{i}"),
                                    type="function",
                                    function=SimpleNamespace(
                                        name=getattr(tc.function, "name", ""),
                                        arguments=getattr(tc.function, "arguments", ""),
                                    ),
                                )
                            ],
                            reasoning=None,
                            reasoning_content=None,
                        ),
                        finish_reason=None,
                        index=0,
                    )
                ],
                model=model,
                usage=None,
            )

    yield SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(
                    role=None,
                    content=None,
                    tool_calls=None,
                    reasoning=None,
                    reasoning_content=None,
                ),
                finish_reason=finish_reason,
                index=0,
            )
        ],
        model=model,
        usage=usage,
    )


# ---------------------------------------------------------------------------
# Main client
# ---------------------------------------------------------------------------


class CursorAgentClient:
    """Minimal OpenAI-client-compatible facade for the Cursor Agent CLI."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        default_headers: dict[str, str] | None = None,
        command: str | None = None,
        args: list[str] | None = None,
        workspace: str | None = None,
        mode: str | None = None,
        timeout_seconds: float | None = None,
        tool_progress_callback: Any = None,
        context_estimate_callback: Any = None,
        **_: Any,
    ):
        candidate_key = (api_key or os.getenv("CURSOR_API_KEY", "") or "").strip()
        # Treat sentinels ("", "cursor-agent-login", …) as "no key" so we don't
        # forward them to ``cursor-agent --api-key`` (which rejects them and
        # closes stdin, producing BrokenPipeError on our writes).
        self.api_key = None if candidate_key in _API_KEY_SENTINELS else candidate_key
        self.base_url = base_url or CURSOR_MARKER_BASE_URL
        self._default_headers = dict(default_headers or {})
        self._command = (command or _resolve_command()).strip() or DEFAULT_CURSOR_COMMAND
        self._extra_args = list(args) if args else _resolve_extra_args()
        chosen_mode = (mode or _resolve_mode()).strip().lower() or DEFAULT_CURSOR_MODE
        if chosen_mode not in _VALID_CURSOR_MODES:
            chosen_mode = DEFAULT_CURSOR_MODE
        self._mode = chosen_mode
        override = workspace or _resolve_workspace_override()
        self._workspace: str | None = override or None  # None ⇒ tmpdir per call
        # Idle timeout (resets per event). Env var > explicit arg > default.
        self._timeout_seconds = float(timeout_seconds) if timeout_seconds else _DEFAULT_TIMEOUT_SECONDS
        env_timeout = os.environ.get("HERMES_CURSOR_TIMEOUT_SECONDS", "").strip()
        if env_timeout:
            try:
                env_timeout_val = float(env_timeout)
                if env_timeout_val > 0:
                    self._timeout_seconds = env_timeout_val
            except ValueError:
                pass

        self._tool_progress_callback = tool_progress_callback
        # Optional hook invoked with the rough messages-based token estimate
        # *before* the subprocess spawns. Used by the host agent to bump
        # the status-bar (``compressor.last_prompt_tokens``) so the input
        # context is visible during long in-flight turns instead of the
        # bar sitting at 0 until the result event arrives.
        self._context_estimate_callback = context_estimate_callback

        # High-water mark for the Hermes status bar. Held only WITHIN
        # a single Hermes user turn: Hermes loops on tool_calls (cursor
        # returning ``<tool_call>`` blocks for Hermes to run), making
        # multiple cursor calls per user prompt. Each call's footprint
        # can vary (different tools attached, different message slices),
        # so the bar must not flicker between those internal calls.
        # Reset automatically on every NEW user turn (detected by the
        # user-message count growing in the messages list); previously
        # this was a session-wide monotonic mark, which incorrectly
        # froze the bar at the highest-activity turn's value and
        # prevented it from reflecting the actual current input across
        # subsequent prompts.
        self._context_high_water: int = 0
        # Last seen count of user messages in the prompt list. Used to
        # detect new user turns so we can reset the high-water above.
        self._last_user_msg_count: int = 0

        self.chat = _CursorChatNamespace(self)
        self.is_closed = False

        self._active_process: subprocess.Popen[str] | None = None
        self._active_process_lock = threading.Lock()
        self._ephemeral_dirs: list[str] = []
        self._dir_lock = threading.Lock()
        # Session-scoped scratch workspace. Lazily minted on first call
        # and REUSED across all subsequent calls in the same chat session
        # so cursor-agent doesn't pay its ~4.5s "fresh-workspace bootstrap"
        # tax on every turn. Cleaned up by ``close()`` along with any
        # other ephemeral dirs.  When ``self._workspace`` (user override)
        # is set we skip this entirely and honour the explicit path.
        self._session_workspace: str | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        proc: subprocess.Popen[str] | None
        with self._active_process_lock:
            proc = self._active_process
            self._active_process = None
        self.is_closed = True
        # New session starts fresh: drop the high-water floor so the
        # status bar reflects current prompt size, not the residual
        # of a previously-large conversation.
        self._context_high_water = 0
        if proc is not None:
            try:
                proc.terminate()
                proc.wait(timeout=2)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        with self._dir_lock:
            dirs, self._ephemeral_dirs = self._ephemeral_dirs, []
            # Drop the cached session workspace ref; if this client is
            # ever re-used after close() (shouldn't happen, but defensive)
            # the next call will lazy-init a fresh dir.
            self._session_workspace = None
        for d in dirs:
            try:
                shutil.rmtree(d, ignore_errors=True)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # OpenAI-compat surface
    # ------------------------------------------------------------------

    def _create_chat_completion(
        self,
        *,
        model: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        timeout: float | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        **_: Any,
    ) -> Any:
        # Estimate context size from what Hermes is actually sending —
        # this is the authoritative answer for the status bar and the
        # compressor. Cursor's reported ``inputTokens`` is the SUM
        # across internal tool round-trips and undercounts the snapshot
        # at end of turn (after averaging) while overcounting at end of
        # turn (raw); neither matches what's in the next-turn prompt.
        # Estimating from messages keeps the bar consistent before,
        # during, and after the call (#cursor-bar-stable).
        # Detect a NEW user turn: Hermes adds a user message at the top
        # of every fresh prompt cycle, then loops internally on tool_calls
        # without adding more user messages. So a strictly increased user-
        # message count is the signal that this is a new prompt and the
        # high-water mark from the previous turn no longer applies.
        try:
            user_msg_count = sum(
                1 for m in (messages or []) if (m or {}).get("role") == "user"
            )
        except Exception:
            user_msg_count = self._last_user_msg_count
        is_new_user_turn = user_msg_count > self._last_user_msg_count
        if is_new_user_turn:
            # Drop the floor so the bar can reflect this turn's actual
            # input size (which may legitimately be smaller than a prior
            # heavy-tool-use turn's per-round average).
            self._context_high_water = 0
        self._last_user_msg_count = user_msg_count

        try:
            from agent.model_metadata import estimate_request_tokens_rough
            self._last_messages_estimate = estimate_request_tokens_rough(
                messages or [], tools=tools or None
            )
        except Exception:
            self._last_messages_estimate = 0

        # Bump the high-water mark NOW (before subprocess spawn) so the
        # status bar reflects input context immediately. Without this the
        # bar shows 0/200K throughout a long in-flight FIRST turn because
        # the compressor only learns about prompt_tokens from the final
        # response.
        if self._last_messages_estimate > self._context_high_water:
            self._context_high_water = self._last_messages_estimate
        if callable(self._context_estimate_callback) and self._last_messages_estimate > 0:
            # On a new user turn, signal the host so it can reset its
            # compressor bar to this turn's estimate (allowing the bar
            # to DROP if appropriate). Otherwise the callback should
            # bump monotonically so the in-loop cursor calls don't
            # flicker the bar down between iterations.
            try:
                self._context_estimate_callback(
                    self._last_messages_estimate, reset=is_new_user_turn
                )
            except TypeError:
                # Backward-compat: older callbacks without ``reset`` kwarg.
                try:
                    self._context_estimate_callback(self._last_messages_estimate)
                except Exception:
                    pass
            except Exception:
                # Never let a UI hook break the actual request.
                pass

        prompt_text = _format_messages_as_prompt(
            messages or [],
            model=model,
            tools=tools,
            tool_choice=tool_choice,
        )

        if timeout is None:
            effective_timeout = self._timeout_seconds
        elif isinstance(timeout, (int, float)):
            effective_timeout = float(timeout)
        else:
            candidates = [
                getattr(timeout, attr, None)
                for attr in ("read", "write", "connect", "pool", "timeout")
            ]
            numeric = [float(v) for v in candidates if isinstance(v, (int, float))]
            effective_timeout = max(numeric) if numeric else self._timeout_seconds

        chosen_model = (model or DEFAULT_CURSOR_MODEL).strip() or DEFAULT_CURSOR_MODEL

        accumulator = self._run_prompt(
            prompt_text=prompt_text,
            model=chosen_model,
            timeout_seconds=effective_timeout,
        )

        # Use the synthesis text (post-last-tool) so the user gets the
        # actual answer without the wall of "let me check X next" prose
        # that cursor's model emits between tool batches. The
        # intermediate prose was already surfaced live via the text-
        # event bridge.
        assistant_text = accumulator.synthesis_text()
        reasoning_text = accumulator.assembled_reasoning() or None

        if accumulator.is_error:
            raise RuntimeError(
                f"cursor-agent reported an error: {accumulator.error_message or assistant_text}"
            )

        tool_calls, cleaned_text = _extract_tool_calls_from_text(assistant_text)

        cursor_internal_tools = [evt.to_public_dict() for evt in accumulator.tool_events]
        # Hand cursor's accumulator our messages-based estimate so
        # ``openai_usage`` can use it as the canonical ``prompt_tokens``
        # the status bar reads from. Without this the bar shows
        # different numbers during vs after generation (cursor's
        # per-round average vs Hermes' messages estimate, ~3x apart).
        #
        # We also gate the estimate by the running high-water mark so
        # the bar never visibly DROPS within a chat session — a wobble
        # we saw when Hermes loops over multiple cursor calls per user
        # turn (each call has a different tools/messages footprint).
        cur_estimate = getattr(self, "_last_messages_estimate", 0) or 0
        # ``openai_usage`` may also use cursor's per-round average; mix
        # it in so the high-water never undercounts when our estimate
        # is too low (e.g. tools=[] on a follow-up call).
        cursor_per_round = self._estimate_per_round_context(accumulator)
        new_high = max(self._context_high_water, cur_estimate, cursor_per_round)
        self._context_high_water = new_high
        accumulator.messages_estimate = new_high
        assistant_message = SimpleNamespace(
            content=cleaned_text,
            tool_calls=tool_calls,
            reasoning=reasoning_text,
            reasoning_content=reasoning_text,
            reasoning_details=None,
            # Audit log of cursor-agent's *internal* tool calls (shell/read/
            # edit/etc. that cursor's harness ran by itself). Hermes' UI is
            # already shown them in real time via tool_progress_callback;
            # this field lets sessions persist what happened.
            cursor_internal_tools=cursor_internal_tools,
        )
        finish_reason = "tool_calls" if tool_calls else "stop"
        choice = SimpleNamespace(
            message=assistant_message,
            finish_reason=finish_reason,
            index=0,
        )
        return SimpleNamespace(
            choices=[choice],
            usage=accumulator.openai_usage(),
            model=chosen_model,
            id=accumulator.request_id or f"cursor-{accumulator.session_id}",
            object="chat.completion",
            cursor_internal_tools=cursor_internal_tools,
        )

    # ------------------------------------------------------------------
    # Subprocess plumbing
    # ------------------------------------------------------------------

    def _build_argv(self, *, model: str, workspace: str) -> list[str]:
        argv = [
            self._command,
            "-p",
            "--output-format",
            "stream-json",
        ]
        # Only forward ``--mode`` to the CLI for values it knows about.
        # The synthetic ``agent`` value means "use cursor's default
        # permissionMode" — achieved by omitting the flag entirely.
        if self._mode in _CURSOR_CLI_MODES:
            argv.extend(["--mode", self._mode])
        argv.extend(
            [
                "--model",
                model,
                "--workspace",
                workspace,
                "--force",
                "--trust",
            ]
        )
        if self.api_key:
            argv.extend(["--api-key", self.api_key])
        argv.extend(self._extra_args)
        return argv

    def _preflight_cli_auth(self) -> None:
        """Fail early when CLI-login mode has expired.

        API-key mode is long-lived and bypasses ``cursor-agent status``. CLI
        login mode is convenient for interactive use but can expire, and an
        expired session otherwise surfaces later as a noisy BrokenPipe/stdin
        failure after spawning ``cursor-agent -p``.
        """
        if self.api_key:
            return
        try:
            out = subprocess.check_output(
                [self._command, "status"],
                text=True,
                timeout=10,
                env=_build_subprocess_env(None),
                stderr=subprocess.STDOUT,
            )
        except FileNotFoundError:
            # Preserve the existing missing-CLI error from the actual Popen
            # path so users get the install guidance already tested there.
            return
        except subprocess.CalledProcessError as exc:
            detail = ""
            if isinstance(exc.output, str):
                detail = exc.output
            stderr = getattr(exc, "stderr", None)
            if isinstance(stderr, str) and stderr.strip():
                detail = f"{detail}\n{stderr}" if detail else stderr
            raise _cursor_cli_auth_error(self._command, detail) from exc
        except Exception as exc:
            raise _cursor_cli_auth_error(self._command, str(exc)) from exc
        if "Logged in as" not in out and "logged in" not in out.lower():
            raise _cursor_cli_auth_error(self._command, out)

    def _allocate_workspace(self) -> tuple[str, bool]:
        """Return ``(workspace, ephemeral)``.

        Strategy:
        1. If the caller pinned an explicit ``workspace`` (env var or kwarg),
           always honour it.
        2. Otherwise, lazily mint ONE temp dir for the whole client session
           and reuse it across calls. Per-turn fresh dirs cost cursor-agent
           ~4.5s of bootstrap overhead each invocation (measured), and there's
           no isolation benefit between turns of the SAME chat session
           anyway — they're already operating on behalf of the same user.

        The session workspace is tracked in ``_ephemeral_dirs`` so
        ``close()`` cleans it up just like the legacy per-call dirs.
        """
        if self._workspace:
            try:
                Path(self._workspace).mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
            return self._workspace, False
        with self._dir_lock:
            if self._session_workspace is None:
                tmp = tempfile.mkdtemp(prefix="hermes-cursor-")
                self._session_workspace = tmp
                self._ephemeral_dirs.append(tmp)
            return self._session_workspace, True

    def _run_prompt(
        self,
        *,
        prompt_text: str,
        model: str,
        timeout_seconds: float,
    ) -> _StreamJsonAccumulator:
        self._preflight_cli_auth()
        workspace, _ephemeral = self._allocate_workspace()
        argv = self._build_argv(model=model, workspace=workspace)

        try:
            proc = subprocess.Popen(
                argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                cwd=workspace,
                env=_build_subprocess_env(self.api_key),
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Could not start Cursor Agent CLI '{self._command}'. "
                "Install Cursor CLI (https://cursor.com/dashboard/integrations) "
                "or set HERMES_CURSOR_COMMAND / CURSOR_AGENT_PATH."
            ) from exc

        if proc.stdin is None or proc.stdout is None:
            proc.kill()
            raise RuntimeError("cursor-agent process did not expose stdin/stdout pipes.")

        self.is_closed = False
        with self._active_process_lock:
            self._active_process = proc

        try:
            # Drain stderr concurrently while we feed stdin so a fast-exiting
            # cursor-agent (e.g. on bad auth) can't deadlock or hide its
            # diagnostic message behind our pipe write.
            stderr_tail: deque[str] = deque(maxlen=80)
            inbox: queue.Queue[dict[str, Any]] = queue.Queue()

            def _stderr_reader_early() -> None:
                if proc.stderr is None:
                    return
                for line in proc.stderr:
                    stderr_tail.append(line.rstrip("\n"))

            err_thread = threading.Thread(target=_stderr_reader_early, daemon=True)
            err_thread.start()

            stdin_error: BaseException | None = None
            try:
                proc.stdin.write(prompt_text)
                proc.stdin.flush()
            except BrokenPipeError as exc:
                # cursor-agent closed stdin before consuming the prompt — almost
                # always means it rejected auth (e.g. invalid API key) or
                # bailed on a flag.  Capture the cause; we'll raise after we
                # have stderr context.
                stdin_error = exc
            except Exception as exc:  # pragma: no cover - defensive
                stdin_error = exc
            finally:
                try:
                    proc.stdin.close()
                except Exception:
                    pass

            if stdin_error is not None:
                # Give the child a moment to flush its error message, then bail.
                try:
                    proc.wait(timeout=3)
                except Exception:
                    pass
                err_thread.join(timeout=1)
                exit_code = getattr(proc, "returncode", None)
                if exit_code is None:
                    try:
                        exit_code = proc.poll()
                    except Exception:
                        exit_code = None
                stderr_text = "\n".join(stderr_tail).strip()
                redacted = redact_sensitive_text(stderr_text, force=True) if stderr_text else ""
                detail = f" stderr: {redacted}" if redacted else ""
                raise RuntimeError(
                    "cursor-agent closed stdin before reading the prompt "
                    f"(exit {exit_code}).{detail}"
                ) from stdin_error

            def _stdout_reader() -> None:
                if proc.stdout is None:
                    return
                for line in proc.stdout:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        inbox.put(json.loads(line))
                    except Exception:
                        # Cursor sometimes prints non-JSON warnings before/after
                        # the JSON stream — preserve them in stderr_tail-like
                        # form so timeouts can surface useful diagnostics.
                        stderr_tail.append("[stdout-non-json] " + line)

            out_thread = threading.Thread(target=_stdout_reader, daemon=True)
            out_thread.start()
            # err_thread is already running from the pre-stdin-write block above.

            accumulator = _StreamJsonAccumulator(
                on_tool_event=self._build_tool_event_bridge(),
                on_text_event=self._build_text_event_bridge(),
            )
            # Idle deadline, not wall-clock. Resets on every successful
            # stream-json event. A turn can run arbitrarily long in total
            # wall time provided the subprocess keeps emitting events
            # (text deltas, tool_calls, tool_results). Only true hangs
            # (no events for ``timeout_seconds``) trigger termination.
            idle_seconds = float(timeout_seconds)
            deadline = time.monotonic() + idle_seconds

            while not accumulator.terminal:
                if time.monotonic() >= deadline:
                    self._terminate_active_proc(proc)
                    raise TimeoutError(
                        f"cursor-agent emitted no events for {idle_seconds:.0f}s; "
                        f"presumed hung. Set HERMES_CURSOR_TIMEOUT_SECONDS to "
                        f"increase the idle threshold."
                    )
                if proc.poll() is not None and inbox.empty():
                    break
                try:
                    event = inbox.get(timeout=0.25)
                except queue.Empty:
                    continue
                # Successful event arrival => subprocess is alive and
                # making progress. Reset the idle deadline.
                deadline = time.monotonic() + idle_seconds
                try:
                    accumulator.feed(event)
                except Exception:
                    # Don't let a malformed event abort the entire request.
                    # Keep draining; if the terminal result never comes,
                    # the idle deadline above will surface the failure.
                    continue

            if not accumulator.terminal:
                stderr_text = "\n".join(stderr_tail).strip()
                redacted = redact_sensitive_text(stderr_text, force=True) if stderr_text else ""
                raise RuntimeError(
                    "cursor-agent exited before emitting a terminal result. "
                    + (f"stderr tail:\n{redacted}" if redacted else "(no stderr)")
                )

            return accumulator
        finally:
            self._terminate_active_proc(proc)

    def _estimate_per_round_context(self, accumulator: "_StreamJsonAccumulator") -> int:
        """Compute cursor's per-round context estimate without mutating it.

        Mirrors the math in :meth:`_StreamJsonAccumulator.openai_usage`
        but returns just the per-round figure so we can feed it into
        the high-water mark before swapping in the messages estimate.
        """
        input_tokens_raw = int(accumulator.usage.get("inputTokens", 0))
        cache_read_raw = int(accumulator.usage.get("cacheReadTokens", 0))
        rounds = max(len(accumulator.tool_events) + 1, 1)
        per_round_input = input_tokens_raw // rounds if rounds > 0 else input_tokens_raw
        per_round_cache = cache_read_raw // rounds if rounds > 0 else cache_read_raw
        return per_round_cache + per_round_input

    def reset_context_baseline(self) -> None:
        """Reset the bar's monotonic floor (e.g. on ``/new`` or compress).

        Hermes' chat session calls ``close()`` on the client when
        starting a fresh session; clients spawned with ``shared=True``
        outlive that. This is the explicit hook for any caller that
        wants the bar to drop back to current-prompt size after a
        deliberate context wipe.
        """
        self._context_high_water = 0

    def _build_text_event_bridge(self) -> Any:
        """Adapter for cursor's intermediate ``assistant`` text events.

        Cursor emits "planning text" between tool batches (e.g.
        "Searching the agent directory…" → tools → "Reading each
        matching file…" → tools → final synthesis). We surface each
        intermediate piece as a synthetic ``narrate`` tool-progress
        event so the Hermes activity feed shows the agentic chain
        live, interleaved with the real tool events — instead of
        bundling everything into one wall of text at the end.

        The synthesis text (the final one after the last tool) is
        excluded so it doesn't double-up with the response body.
        """
        cb = self._tool_progress_callback
        if cb is None:
            return None

        def _bridge(text: str) -> None:
            try:
                preview = text.strip().splitlines()[0] if text else ""
                if len(preview) > 240:
                    preview = preview[:237] + "..."
                if not preview:
                    return
                cb("tool.started", "narrate", preview, {"text": text})
                cb(
                    "tool.completed", "narrate", None, None,
                    duration=0.0, is_error=False, result=text,
                )
            except Exception:
                pass

        return _bridge

    def _build_tool_event_bridge(self) -> Any:
        """Adapter from our ``_CursorToolEvent`` stream to Hermes' callback.

        ``tool_progress_callback(event_type, name, preview, args, ...)`` is
        the same shape Hermes' built-in tools use (see
        ``agent/tool_executor.py``). We translate cursor's "tool_call
        started/completed" stream-json events into ``tool.started`` /
        ``tool.completed`` callbacks so the user's UI shows cursor's
        internal shell/read/edit activity the same way it shows native
        tool calls from Grok, GPT, Claude, etc.

        Without this bridge, cursor's tool activity is invisible to Hermes
        — the user only sees the model's final text and the session's
        ``tool_call_count`` stays at zero even when cursor actually ran
        multiple shell/read commands internally.
        """
        cb = self._tool_progress_callback
        if cb is None:
            return None

        def _bridge(stage: str, evt: _CursorToolEvent) -> None:
            try:
                if stage == "started":
                    preview = _build_cursor_tool_preview(evt)
                    cb("tool.started", evt.name, preview, evt.args)
                elif stage == "completed":
                    # cli.py stores ``function_args`` from tool.started in a
                    # FIFO queue and pops them on tool.completed for display.
                    # ``evt.args`` is the SAME dict reference, so mutating it
                    # here surfaces our diff stats to ``get_cute_tool_message``
                    # without changing the upstream callback signature.
                    if (
                        evt.lines_added is not None
                        or evt.lines_removed is not None
                    ) and isinstance(evt.args, dict):
                        evt.args["_diff_stats"] = {
                            "added": evt.lines_added or 0,
                            "removed": evt.lines_removed or 0,
                        }
                        if evt.diff_string:
                            evt.args["_diff_string"] = evt.diff_string
                    cb(
                        "tool.completed",
                        evt.name,
                        None,
                        None,
                        duration=evt.duration_ms / 1000.0,
                        is_error=evt.is_error,
                        result=evt.result_text,
                    )
            except Exception:
                # The Hermes callback may not accept all our kwargs (e.g.
                # older Hermes builds). Fall back to the simplest form.
                try:
                    cb(f"tool.{stage}", evt.name, evt.result_text or "", evt.args)
                except Exception:
                    pass

        return _bridge

    def _terminate_active_proc(self, proc: subprocess.Popen[str]) -> None:
        with self._active_process_lock:
            current = self._active_process
            if current is proc:
                self._active_process = None
        if proc.poll() is not None:
            return
        # cursor-agent exits naturally a few hundred ms after emitting the
        # ``result`` event. Give it that grace period BEFORE force-killing —
        # SIGTERM forces Node.js to run shutdown hooks which can take
        # longer than just letting it exit on its own.
        try:
            proc.wait(timeout=0.7)
            return
        except subprocess.TimeoutExpired:
            pass
        # Still running — force it.
        try:
            proc.terminate()
            proc.wait(timeout=1.5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def whoami(self) -> dict[str, Any]:
        """Return a dict of ``cursor-agent status`` info (best-effort).

        Used by the doctor / setup flow to surface logged-in user + tier.
        Returns an empty dict if the CLI is missing or not authenticated.
        """
        try:
            out = subprocess.check_output(
                [self._command, "status"],
                text=True,
                timeout=10,
                env=_build_subprocess_env(self.api_key),
            )
        except Exception:
            return {}
        info: dict[str, Any] = {"raw": out.strip()}
        for line in out.splitlines():
            line = line.strip()
            if line.startswith("✓ Logged in as "):
                info["email"] = line.removeprefix("✓ Logged in as ").strip()
                info["authenticated"] = True
        return info


__all__ = [
    "CursorAgentClient",
    "CURSOR_MARKER_BASE_URL",
    "DEFAULT_CURSOR_COMMAND",
    "DEFAULT_CURSOR_MODE",
    "DEFAULT_CURSOR_MODEL",
]
