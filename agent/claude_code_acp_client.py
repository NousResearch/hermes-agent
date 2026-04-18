"""OpenAI-compatible shim that forwards Hermes requests to Claude Code over ACP.

Unlike :mod:`agent.copilot_acp_client`, which scrapes ``<tool_call>`` blocks
from flat text output, this client consumes the **native** ACP events emitted
by ``@zed-industries/claude-agent-acp`` — Claude Code runs its own tool loop
autonomously, so tool-call facts arrive as first-class ``session/update``
messages of kind ``tool_call_start`` and ``tool_call_update``.

The client:

* Builds a per-session Claude Code sandbox (``CLAUDE.md``, skills,
  ``.mcp.json``) via :mod:`agent.claude_code_sandbox` so hermes's persona,
  memory, skills, and tools are available inside the subprocess.
* Keeps one ACP subprocess + one ``sessionId`` alive across turns, matching
  the Claude Code editor lifetime. Subsequent calls reuse the session.
* Routes streamed text → ``stream_delta_callback``, thought chunks →
  ``thinking_callback``, and tool events → ``tool_progress_callback``.
* Accumulates a :class:`ToolCallRecord` list and attaches it to the
  response as ``hermes_tool_trace`` so the AIAgent loop can feed it into
  auto-skill-creation and background review without re-running the tools.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterable, List, Optional

from agent._acp_client_base import (
    _AcpClientBase,
    _format_messages_as_prompt,
    resolve_effective_timeout,
)

logger = logging.getLogger(__name__)


DEFAULT_CLAUDE_CODE_ACP_PACKAGE = "@zed-industries/claude-agent-acp"
ACP_MARKER_BASE_URL = "acp://claude-code"
_DEFAULT_TIMEOUT_SECONDS = 900.0

# Canonical system-preamble lines for Claude Code under ACP. Exported so
# `agent.claude_code_sandbox` can share the same source of truth when it
# composes CLAUDE.md — we used to maintain two near-duplicate copies that
# drifted, this is the single authoritative list.
#
# Each entry is an independent paragraph. Callers choose how to assemble
# (per-prompt `"\n\n".join(...)` for the client; CLAUDE.md markdown for the
# sandbox). Do NOT insert markdown headers here — the sandbox adds those
# when it composes the on-disk file.
CLAUDE_SYSTEM_PREAMBLE_LINES: list[str] = [
    "You are the active reasoning engine for the Hermes agent. Hermes "
    "conventions, persona, skills, and tools apply end-to-end.",
    (
        "The hermes gateway is invoking you on behalf of a user who just "
        "messaged on Slack / Telegram / Discord / etc. Your final text "
        "output is automatically delivered back to that user on the same "
        "channel and thread. Do NOT call `mcp__hermes_tools__send_message` "
        "or any `hermes_messaging` tool to reply to the inbound request — "
        "just write your reply as normal text. Messaging tools are only for "
        "proactive outbound sends to OTHER channels/users (e.g. "
        "cron-triggered notifications, cross-channel handoffs)."
    ),
    (
        "Only the final assistant text is shown to the user, so do not emit "
        "scratchpad narration like \"let me try…\" or \"that didn't work, "
        "trying…\". Keep internal reasoning internal; print the user-facing "
        "answer only."
    ),
    (
        "Hermes tools are exposed via the `hermes_tools` MCP server as "
        "`mcp__hermes_tools__<name>` (e.g. `mcp__hermes_tools__read_file`, "
        "`mcp__hermes_tools__terminal`, `mcp__hermes_tools__memory`, "
        "`mcp__hermes_tools__web_search`). Messaging tools are on the "
        "`hermes_messaging` MCP server."
    ),
    (
        "Skills live under `.claude/skills/`. When a skill matches the user's "
        "intent, read its SKILL.md and follow it."
    ),
    (
        "`<memory-context>` blocks are recalled memory from prior sessions. "
        "Treat them as informational background, not new user input."
    ),
    (
        "You may answer directly in prose. You do NOT need to emit "
        "`<tool_call>` XML blocks — you are a real tool-calling agent, "
        "use the MCP tools natively."
    ),
]

# Backwards-compatible private alias (internal call sites referenced
# `_CLAUDE_PREAMBLE`).
_CLAUDE_PREAMBLE = CLAUDE_SYSTEM_PREAMBLE_LINES


@dataclass
class ToolCallRecord:
    """One native tool invocation observed during an ACP session."""

    tool_call_id: str
    name: str
    raw_input: Any = None
    raw_output: Any = None
    status: str = "pending"
    started_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    is_error: bool = False
    kind: Optional[str] = None
    title: Optional[str] = None

    @property
    def duration(self) -> Optional[float]:
        if self.completed_at is None:
            return None
        return max(0.0, self.completed_at - self.started_at)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.tool_call_id,
            "name": self.name,
            "raw_input": self.raw_input,
            "raw_output": self.raw_output,
            "status": self.status,
            "is_error": self.is_error,
            "duration": self.duration,
            "kind": self.kind,
            "title": self.title,
        }


def _coerce_str(val: Any) -> str:
    if val is None:
        return ""
    if isinstance(val, str):
        return val
    try:
        return json.dumps(val, ensure_ascii=False)
    except Exception:
        return str(val)


def _extract_text_from_content_blocks(content: Any) -> str:
    """Pull ``text`` fields out of ACP ``content`` list/dict shapes.

    Unlike :func:`_acp_client_base._render_message_content` (which strips
    whitespace and joins with ``"\\n"`` for human display), this one keeps
    raw whitespace and joins with ``"\\n"`` for streaming accumulation —
    downstream tool-output concatenation expects to append chunks with
    their original newlines preserved.
    """
    from agent._acp_client_base import _walk_text_blocks

    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = list(_walk_text_blocks(content))
    return "\n".join(parts) if parts else ""


def _short_preview(text: str, limit: int = 160) -> str:
    if not isinstance(text, str):
        text = _coerce_str(text)
    text = " ".join(text.split())
    if len(text) > limit:
        return text[: limit - 1] + "…"
    return text


class ClaudeCodeACPClient(_AcpClientBase):
    """Persistent-session ACP client targeting Claude Code.

    The first :meth:`_create_chat_completion` call spawns the ACP subprocess,
    issues ``initialize`` + ``session/new``, and caches the ``sessionId``.
    Subsequent calls reuse the same subprocess and session — mirroring the
    Claude Code editor lifetime.
    """

    _provider_label = "claude-code-acp"
    _default_command = "npx"
    _default_args = ("-y", DEFAULT_CLAUDE_CODE_ACP_PACKAGE)
    _env_command_vars = ("HERMES_CLAUDE_CODE_ACP_COMMAND", "CLAUDE_ACP_PATH")
    _env_args_var = "HERMES_CLAUDE_CODE_ACP_ARGS"
    _marker_base_url = ACP_MARKER_BASE_URL
    _default_timeout_seconds = _DEFAULT_TIMEOUT_SECONDS
    _client_name = "hermes-agent"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        default_headers: dict[str, str] | None = None,
        acp_command: str | None = None,
        acp_args: list[str] | None = None,
        acp_cwd: str | None = None,
        command: str | None = None,
        args: list[str] | None = None,
        agent: Any | None = None,
        hermes_home: Path | None = None,
        hermes_session_id: str | None = None,
        platform: str | None = None,
        stream_delta_callback: Callable[[str], Any] | None = None,
        thinking_callback: Callable[[str], Any] | None = None,
        tool_progress_callback: Callable[..., Any] | None = None,
        available_tools: Optional[set[str]] = None,
        available_toolsets: Optional[set[str]] = None,
        **_: Any,
    ) -> None:
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            default_headers=default_headers,
            acp_command=acp_command,
            acp_args=acp_args,
            acp_cwd=acp_cwd,
            command=command,
            args=args,
        )
        self._agent = agent
        self._hermes_home = Path(hermes_home) if hermes_home else None
        self._hermes_session_id = (
            hermes_session_id
            or os.environ.get("HERMES_SESSION_ID", "").strip()
            or f"hermes-{int(time.time())}"
        )
        self._platform = platform or os.environ.get("HERMES_SESSION_PLATFORM") or None
        self._available_tools = set(available_tools) if available_tools else None
        self._available_toolsets = set(available_toolsets) if available_toolsets else None

        self.stream_delta_callback = stream_delta_callback
        self.thinking_callback = thinking_callback
        self.tool_progress_callback = tool_progress_callback

        self._session_lock = threading.Lock()
        self._session_proc: subprocess.Popen[str] | None = None
        self._session_inbox: "queue.Queue[dict[str, Any]]" | None = None
        self._session_stderr: deque[str] | None = None
        self._session_id: str | None = None
        self._sandbox_path: Path | None = None
        self._pending_model: str | None = None

        # Tool-trace state is mutated by the ACP reader thread (via
        # `_handle_tool_call_start` / `_handle_tool_call_update`) and read by
        # the main thread inside `_create_chat_completion` and the public
        # `tool_trace` accessor. Guard list/dict mutations and reads with
        # this lock. User callbacks (stream/thinking/tool_progress) must be
        # invoked OUTSIDE the lock — they can re-enter hermes code and
        # deadlock if we hold it across arbitrary user code.
        self._trace_lock = threading.Lock()
        self._tool_trace: list[ToolCallRecord] = []
        self._tool_records_by_id: dict[str, ToolCallRecord] = {}

    # ------------------------------------------------------------------
    # Sandbox + session lifecycle
    # ------------------------------------------------------------------

    def _resolve_hermes_home(self) -> Path:
        if self._hermes_home is not None:
            return self._hermes_home
        try:
            from hermes_constants import get_hermes_home

            return get_hermes_home()
        except Exception:
            return Path.home() / ".hermes"

    def _ensure_sandbox(self, *, model: str | None = None) -> Path:
        if self._sandbox_path is not None:
            return self._sandbox_path
        from agent.claude_code_sandbox import build_session_sandbox

        hermes_home = self._resolve_hermes_home()
        self._sandbox_path = build_session_sandbox(
            self._hermes_session_id,
            self._agent,
            hermes_home=hermes_home,
            platform=self._platform,
            available_tools=self._available_tools,
            available_toolsets=self._available_toolsets,
            model=model or self._pending_model,
        )
        # ACP subprocess must cd into the sandbox so .mcp.json and CLAUDE.md
        # are picked up.
        self._acp_cwd = str(self._sandbox_path)
        return self._sandbox_path

    def _build_session_new_params(self) -> dict[str, Any]:
        sandbox = self._ensure_sandbox()
        mcp_servers = self._load_mcp_servers(sandbox)
        params: dict[str, Any] = {
            "cwd": str(sandbox),
            "mcpServers": mcp_servers,
        }
        return params

    def _load_mcp_servers(self, sandbox: Path) -> list[dict[str, Any]]:
        """Translate the sandbox's .mcp.json into ACP's mcpServers array shape."""
        mcp_path = sandbox / ".mcp.json"
        if not mcp_path.exists():
            return []
        try:
            cfg = json.loads(mcp_path.read_text(encoding="utf-8"))
        except Exception:
            return []
        servers = cfg.get("mcpServers") or {}
        out: list[dict[str, Any]] = []
        for name, spec in servers.items():
            if not isinstance(spec, dict):
                continue
            cmd = spec.get("command")
            if not isinstance(cmd, str) or not cmd.strip():
                continue
            entry: dict[str, Any] = {
                "name": name,
                "command": cmd,
                "args": list(spec.get("args") or []),
            }
            env = spec.get("env") or {}
            if isinstance(env, dict) and env:
                entry["env"] = [
                    {"name": str(k), "value": str(v)}
                    for k, v in env.items()
                ]
            out.append(entry)
        return out

    def _ensure_session(self) -> tuple[
        subprocess.Popen[str],
        "queue.Queue[dict[str, Any]]",
        deque[str],
        str,
    ]:
        # Session-per-client, lazily initialized on first prompt. We deliberately
        # do NOT spawn the ACP subprocess in __init__: ClaudeCodeACPClient is
        # constructed eagerly (e.g., from auxiliary_client bootstrapping) even
        # when no turn is about to run, so import-time subprocess spawning would
        # burn a claude-agent-acp process per import. The lock guards against
        # two threads racing to start a session for the same client.
        with self._session_lock:
            if (
                self._session_proc is not None
                and self._session_proc.poll() is None
                and self._session_id is not None
                and self._session_inbox is not None
                and self._session_stderr is not None
            ):
                return (
                    self._session_proc,
                    self._session_inbox,
                    self._session_stderr,
                    self._session_id,
                )

            self._ensure_sandbox()
            proc, inbox, stderr_tail = self._start_subprocess()
            try:
                self._request(
                    proc, inbox, stderr_tail,
                    "initialize", self._build_initialize_params(),
                    timeout_seconds=self._default_timeout_seconds,
                )
                session = self._request(
                    proc, inbox, stderr_tail,
                    "session/new", self._build_session_new_params(),
                    timeout_seconds=self._default_timeout_seconds,
                ) or {}
                session_id = str(session.get("sessionId") or "").strip()
                if not session_id:
                    raise RuntimeError(
                        "claude-agent-acp did not return a sessionId from session/new."
                    )
            except Exception:
                try:
                    proc.terminate()
                except Exception:
                    pass
                raise

            self._session_proc = proc
            self._session_inbox = inbox
            self._session_stderr = stderr_tail
            self._session_id = session_id
            return proc, inbox, stderr_tail, session_id

    def close(self) -> None:
        with self._session_lock:
            self._session_id = None
            self._session_inbox = None
            self._session_stderr = None
        super().close()
        self._session_proc = None

    # ------------------------------------------------------------------
    # session/update routing (native tool events)
    # ------------------------------------------------------------------

    def _handle_session_update(
        self,
        update: dict[str, Any],
        *,
        dispatch_ctx: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> None:
        ctx = dispatch_ctx or {}
        kind = str(update.get("sessionUpdate") or "").strip()

        if kind == "agent_message_chunk":
            text = _extract_text_from_content_blocks(update.get("content"))
            if text:
                text_parts = ctx.get("text_parts")
                if isinstance(text_parts, list):
                    text_parts.append(text)
                cb = self.stream_delta_callback
                if callable(cb):
                    try:
                        cb(text)
                    except Exception as exc:
                        logger.debug("stream_delta_callback raised: %s", exc)
            return

        if kind == "agent_thought_chunk":
            text = _extract_text_from_content_blocks(update.get("content"))
            if text:
                reasoning_parts = ctx.get("reasoning_parts")
                if isinstance(reasoning_parts, list):
                    reasoning_parts.append(text)
                cb = self.thinking_callback
                if callable(cb):
                    try:
                        cb(text)
                    except Exception as exc:
                        logger.debug("thinking_callback raised: %s", exc)
            return

        if kind == "tool_call_start":
            self._handle_tool_call_start(update)
            return

        if kind == "tool_call_update":
            self._handle_tool_call_update(update)
            return

        if kind == "plan":
            self._handle_plan_update(update)
            return

        if kind == "available_commands_update":
            # Informational — Claude Code announces slash commands available.
            return

        # Unknown update kind — log at debug level so we spot protocol drift.
        logger.debug("Unhandled ACP session/update kind: %s", kind)

    def _handle_tool_call_start(self, update: dict[str, Any]) -> None:
        raw_input = update.get("rawInput")
        if raw_input is None:
            raw_input = update.get("input")
        kind = update.get("kind")
        tool_id_raw = str(update.get("toolCallId") or update.get("id") or "").strip()
        name = str(update.get("title") or update.get("name") or "").strip() or "unknown_tool"

        with self._trace_lock:
            tool_id = tool_id_raw or f"tool-{len(self._tool_trace) + 1}"
            record = ToolCallRecord(
                tool_call_id=tool_id,
                name=name,
                raw_input=raw_input,
                status="in_progress",
                kind=str(kind) if kind else None,
                title=update.get("title"),
            )
            self._tool_trace.append(record)
            self._tool_records_by_id[tool_id] = record

        # Callback is deliberately outside the lock: hermes callbacks can
        # re-enter this client (e.g. via streamed UI updates that look up
        # tool state), so holding the lock across them risks deadlock.
        cb = self.tool_progress_callback
        if callable(cb):
            preview = _short_preview(_coerce_str(raw_input))
            try:
                cb("tool.started", name, preview, raw_input)
            except Exception as exc:
                logger.debug("tool_progress_callback(started) raised: %s", exc)

    def _handle_tool_call_update(self, update: dict[str, Any]) -> None:
        tool_id = str(update.get("toolCallId") or update.get("id") or "").strip()
        raw_output = update.get("rawOutput")
        if raw_output is None:
            raw_output = update.get("output")
        if raw_output is None:
            raw_output = _extract_text_from_content_blocks(update.get("content"))
        status = update.get("status")
        is_error_flag = bool(update.get("isError"))

        # All record mutations and the terminal-status read happen under the
        # lock so a concurrent reader either sees the fully-updated record
        # or the prior snapshot — never a half-written one.
        with self._trace_lock:
            record = self._tool_records_by_id.get(tool_id)
            if record is None:
                # Orphaned update (start event missing or out-of-order under
                # load) — create a placeholder so we still capture output.
                record = ToolCallRecord(
                    tool_call_id=tool_id or f"tool-{len(self._tool_trace) + 1}",
                    name=str(update.get("title") or "unknown_tool"),
                    status="in_progress",
                )
                self._tool_trace.append(record)
                if tool_id:
                    self._tool_records_by_id[tool_id] = record

            if raw_output:
                if record.raw_output is None:
                    record.raw_output = raw_output
                elif isinstance(record.raw_output, str) and isinstance(raw_output, str):
                    record.raw_output = record.raw_output + raw_output
                else:
                    record.raw_output = raw_output

            if isinstance(status, str):
                record.status = status

            if is_error_flag:
                record.is_error = True

            # Terminal-status detection: Claude Code / claude-agent-acp has
            # drifted across versions — some emit `status: completed`, some
            # `isComplete: true`, some `complete: true`. The redundancy is
            # defensive against protocol drift; keep all four checks.
            is_terminal = (
                record.status in {"completed", "failed", "cancelled", "success"}
                or update.get("isComplete") is True
                or update.get("complete") is True
            )
            if is_terminal:
                record.completed_at = time.time()
                # Snapshot fields the callback needs so the call itself can
                # run outside the lock.
                cb_snapshot = (
                    record.name,
                    _short_preview(_coerce_str(record.raw_output)),
                    record.raw_input,
                    record.duration or 0.0,
                    record.is_error,
                )
            else:
                cb_snapshot = None

        # Callback invocation is outside the lock — see _handle_tool_call_start.
        if cb_snapshot is not None:
            cb = self.tool_progress_callback
            if callable(cb):
                name, preview, raw_input_val, duration, is_error_val = cb_snapshot
                try:
                    cb(
                        "tool.completed",
                        name,
                        preview,
                        raw_input_val,
                        duration=duration,
                        is_error=is_error_val,
                    )
                except Exception as exc:
                    logger.debug(
                        "tool_progress_callback(completed) raised: %s", exc
                    )

    def _handle_plan_update(self, update: dict[str, Any]) -> None:
        cb = self.thinking_callback
        if not callable(cb):
            return
        entries = update.get("entries") or update.get("plan") or []
        if not isinstance(entries, list):
            return
        summary_parts: list[str] = []
        for item in entries:
            if isinstance(item, dict):
                content = item.get("content") or item.get("text") or ""
                if isinstance(content, str) and content.strip():
                    summary_parts.append(content.strip())
            elif isinstance(item, str) and item.strip():
                summary_parts.append(item.strip())
        if not summary_parts:
            return
        try:
            cb("📝 plan: " + "; ".join(summary_parts[:3]))
        except Exception as exc:
            logger.debug("thinking_callback(plan) raised: %s", exc)

    # ------------------------------------------------------------------
    # create() entrypoint
    # ------------------------------------------------------------------

    def _create_chat_completion(
        self,
        *,
        model: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        timeout: Any = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        **_: Any,
    ) -> Any:
        prompt_text = _format_messages_as_prompt(
            messages or [],
            preamble=_CLAUDE_PREAMBLE,
            model=model,
            tools=None,  # Tools are exposed via MCP; suppress OpenAI schema dump.
            tool_choice=None,
            tool_call_instructions=None,
            closing=(
                "Continue the conversation from the latest user request. "
                "Use the hermes MCP tools natively."
            ),
        )
        effective_timeout = resolve_effective_timeout(
            timeout, default=self._default_timeout_seconds
        )

        # Remember the caller's model so the sandbox-build (lazy, inside
        # _ensure_session → _ensure_sandbox) can pin it in
        # .claude/settings.local.json for claude-agent-acp to pick up.
        if model:
            self._pending_model = model

        proc, inbox, stderr_tail, session_id = self._ensure_session()

        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        # Mark the trace boundary for this turn so the response's
        # hermes_tool_trace only carries tool calls observed during *this*
        # prompt. Read under the trace lock since the reader thread may be
        # appending from prior in-flight events.
        with self._trace_lock:
            pre_trace_len = len(self._tool_trace)

        # Cancellation: if we were constructed with an AIAgent reference
        # that exposes `_interrupt_requested`, poll it mid-turn so a user
        # Ctrl-C propagates inside seconds instead of waiting for Claude
        # Code's (possibly very long) tool loop to finish.
        def _cancel_check() -> bool:
            return bool(getattr(self._agent, "_interrupt_requested", False))

        try:
            result = self._request(
                proc, inbox, stderr_tail,
                "session/prompt",
                {
                    "sessionId": session_id,
                    "prompt": [{"type": "text", "text": prompt_text}],
                },
                timeout_seconds=effective_timeout,
                dispatch_ctx={
                    "text_parts": text_parts,
                    "reasoning_parts": reasoning_parts,
                },
                cancel_check=_cancel_check if self._agent is not None else None,
            )
        except Exception as exc:
            # Kept broad: session/prompt failures can surface as BrokenPipeError,
            # RuntimeError (protocol error), TimeoutError, or arbitrary
            # adapter-raised exceptions. Log the concrete class + repr at
            # warning level so operators can tell whether they need to look
            # at the subprocess, the network, or Claude Code adapter itself —
            # then drop the cached session so the next call re-initializes.
            logger.warning(
                "Claude Code ACP prompt failed (%s: %r); closing session",
                type(exc).__name__,
                exc,
            )
            self.close()
            raise

        response_text = "".join(text_parts).strip()
        reasoning_text = "".join(reasoning_parts).strip()

        # If the prompt response itself carried a stopReason / message payload,
        # merge any final text (some adapters emit it only at response-time).
        if isinstance(result, dict):
            tail = _extract_text_from_content_blocks(
                result.get("content") or result.get("message")
            )
            if tail and tail not in response_text:
                response_text = (response_text + "\n" + tail).strip() if response_text else tail

        # Snapshot under the lock: the reader thread may still deliver late
        # session/update events between the prompt result and this read.
        # We serialize to dicts inside the critical section so downstream
        # consumers observe a consistent snapshot even if the reader thread
        # continues mutating the underlying records after we release.
        with self._trace_lock:
            turn_trace_dicts = [r.to_dict() for r in self._tool_trace[pre_trace_len:]]

        usage = self._extract_usage(result if isinstance(result, dict) else None)

        assistant_message = SimpleNamespace(
            content=response_text,
            tool_calls=[],  # Claude Code already ran them natively.
            reasoning=reasoning_text or None,
            reasoning_content=reasoning_text or None,
            reasoning_details=None,
        )
        choice = SimpleNamespace(message=assistant_message, finish_reason="stop")
        response = SimpleNamespace(
            choices=[choice],
            usage=usage,
            model=model or "claude-code-acp",
            # NOTE: tool_calls on assistant_message is intentionally empty —
            # Claude Code ran every tool natively inside the ACP subprocess,
            # so there is nothing for hermes to re-execute. Callers that
            # need to know what tools fired (auto-skill-creation,
            # background review) must consume `hermes_tool_trace` instead.
            hermes_tool_trace=turn_trace_dicts,
        )
        return response

    def _extract_usage(self, result: Optional[dict[str, Any]]) -> Any:
        """Best-effort extraction of token usage from a session/prompt result."""
        prompt_tokens = 0
        completion_tokens = 0
        if isinstance(result, dict):
            usage = result.get("usage") or {}
            if isinstance(usage, dict):
                prompt_tokens = int(
                    usage.get("inputTokens")
                    or usage.get("prompt_tokens")
                    or 0
                )
                completion_tokens = int(
                    usage.get("outputTokens")
                    or usage.get("completion_tokens")
                    or 0
                )
        return SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            prompt_tokens_details=SimpleNamespace(cached_tokens=0),
        )

    # ------------------------------------------------------------------
    # Helpers exposed for tests / run_agent integration
    # ------------------------------------------------------------------

    @property
    def tool_trace(self) -> list[ToolCallRecord]:
        """Full list of tool records observed in this client's lifetime."""
        with self._trace_lock:
            return list(self._tool_trace)

    def reset_tool_trace(self) -> None:
        with self._trace_lock:
            self._tool_trace.clear()
            self._tool_records_by_id.clear()


def trace_to_messages_snapshot(
    trace: Iterable[Any],
    *,
    fallback_name: str = "claude_code_tool",
) -> list[dict[str, Any]]:
    """Reconstruct a hermes-shape ``messages_snapshot`` from a tool trace.

    ``trace`` may be an iterable of :class:`ToolCallRecord` instances OR of
    dict records (as produced by :attr:`ToolCallRecord.to_dict`) — both are
    accepted so auto-skill-creation can consume the serialized form.
    Each record becomes an ``assistant`` message containing a ``tool_use``
    block, followed by a ``tool`` message with the ``tool_result`` block.
    """
    messages: list[dict[str, Any]] = []
    for record in trace:
        if hasattr(record, "to_dict"):
            data = record.to_dict()
        elif isinstance(record, dict):
            data = record
        else:
            continue
        call_id = str(data.get("id") or data.get("tool_call_id") or "").strip()
        if not call_id:
            call_id = f"tool-{len(messages) + 1}"
        name = str(data.get("name") or fallback_name).strip() or fallback_name
        raw_input = data.get("raw_input")
        raw_output = data.get("raw_output")

        messages.append(
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": call_id,
                        "name": name,
                        "input": raw_input if raw_input is not None else {},
                    }
                ],
            }
        )
        messages.append(
            {
                "role": "tool",
                "tool_use_id": call_id,
                "content": _coerce_str(raw_output) if raw_output is not None else "",
            }
        )
    return messages
