"""OpenAI-compatible shim that forwards Hermes requests to `claude -p`.

This adapter lets Hermes treat the local Claude Code CLI as a chat-style model
endpoint. Each request spawns a short-lived `claude -p` subprocess with ALL of
Claude Code's built-in tools disabled (`--tools ""`), feeds the formatted
conversation on stdin, reads the single JSON result, and converts it back into
the minimal shape Hermes expects from an OpenAI client.

Why this shape:
  * The genuine, unmodified `claude` binary makes the network call using the
    user's existing Claude Code OAuth/subscription session (plain `-p`, never
    `--bare` which would force ANTHROPIC_API_KEY). Traffic is normal Claude
    Code usage, not pay-per-token API billing — there is no identity forgery,
    it IS Claude Code.
  * Hermes keeps its own agent loop and executes ALL tools. `claude -p` runs
    with no tools of its own; instead the tool schemas are described in the
    prompt and the model emits <tool_call>{...}</tool_call> text blocks, which
    this client parses back into OpenAI tool-call objects for Hermes to run.

Mirrors agent/copilot_acp_client.py (the existing external-process provider),
minus the JSON-RPC/ACP protocol — `claude -p --output-format stream-json` is a
simple one-shot request/response over stdio.

Tool handling: the model emits NATIVE `tool_use` blocks for the tools described
in the prompt (captured straight from the stream and converted to OpenAI
tool-call objects). Because those tools are not registered with Claude Code,
it cannot execute them — it just reports `error_max_turns` after the single
allowed turn, which we intentionally ignore since the assistant message was
already emitted. A `<tool_call>{...}</tool_call>` text fallback is also parsed
in case a model answers in text instead of a native tool_use block.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)

from tools.environments.local import hermes_subprocess_env

CLAUDE_CLI_MARKER_BASE_URL = "acp://claude-cli"
_DEFAULT_TIMEOUT_SECONDS = 900.0
_DEFAULT_MODEL = "sonnet"
_DEFAULT_EFFORT = "medium"
_VALID_EFFORTS = {"low", "medium", "high", "xhigh", "max"}

_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_TOOL_CALL_JSON_RE = re.compile(
    r"\{\s*\"id\"\s*:\s*\"[^\"]+\"\s*,\s*\"type\"\s*:\s*\"function\"\s*,\s*\"function\"\s*:\s*\{.*?\}\s*\}",
    re.DOTALL,
)

# System prompt that pins `claude -p` into a pure model-endpoint role: no
# Claude-Code agent behaviour, emit tool calls as text for Hermes to execute.
_ORCHESTRATOR_SYSTEM_PROMPT = (
    "You are being used as a stateless model endpoint by the Hermes agent "
    "harness. Hermes owns the agent loop and executes every tool itself; you "
    "must NOT attempt to run tools, edit files, or take actions yourself. "
    "Respond to the conversation as the assistant. "
    "IMPORTANT: If a tool should be called, do NOT act on it — instead output "
    "one or more <tool_call>{...}</tool_call> blocks, each containing a single "
    "JSON object in OpenAI function-call shape: "
    '{"id": "<unique>", "type": "function", "function": {"name": "<tool>", '
    '"arguments": "<json-string>"}}. The arguments field MUST be a JSON string. '
    "If no tool is needed, just answer normally."
)


def _resolve_command() -> str:
    return (
        os.getenv("HERMES_CLAUDE_CLI_COMMAND", "").strip()
        or os.getenv("CLAUDE_CLI_PATH", "").strip()
        or "claude"
    )


def _resolve_extra_args() -> list[str]:
    raw = os.getenv("HERMES_CLAUDE_CLI_ARGS", "").strip()
    if not raw:
        return []
    return shlex.split(raw)


def _resolve_home_dir() -> str:
    """Return a stable HOME for the child `claude` process."""
    home = os.environ.get("HOME", "").strip()
    if home:
        return home
    expanded = os.path.expanduser("~")
    if expanded and expanded != "~":
        return expanded
    try:
        import pwd

        resolved = pwd.getpwuid(os.getuid()).pw_dir.strip()  # windows-footgun: ok — POSIX fallback inside try/except
        if resolved:
            return resolved
    except Exception:
        pass
    return "/tmp"


def _build_subprocess_env() -> dict[str, str]:
    # `claude -p` legitimately needs the user's Claude Code OAuth credentials
    # (Keychain / ~/.claude/.credentials.json), reached via HOME. Route through
    # the central helper so Tier-1 Hermes secrets are still stripped, then pin
    # HOME so the credential store resolves.
    env = hermes_subprocess_env(inherit_credentials=True)
    env["HOME"] = _resolve_home_dir()
    from hermes_constants import apply_subprocess_home_env

    apply_subprocess_home_env(env)
    return env


def _normalize_effort(value: Any) -> str:
    if isinstance(value, str) and value.strip().lower() in _VALID_EFFORTS:
        return value.strip().lower()
    return _DEFAULT_EFFORT


def _normalize_model(value: Any) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return _DEFAULT_MODEL


def _format_messages_as_prompt(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    tool_choice: Any = None,
) -> str:
    """Serialize the OpenAI message list + tool schemas into a single prompt."""
    sections: list[str] = []

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
                "Available tools (OpenAI function schema). When using a tool, "
                "emit ONLY <tool_call>{...}</tool_call> with one JSON object "
                "containing id/type/function{name,arguments}. arguments must be "
                "a JSON string.\n" + json.dumps(tool_specs, ensure_ascii=False)
            )

    if tool_choice is not None:
        sections.append(f"Tool choice hint: {json.dumps(tool_choice, ensure_ascii=False)}")

    transcript: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "unknown").strip().lower()
        if role not in {"system", "user", "assistant", "tool"}:
            role = "context"
        rendered = _render_message_content(message.get("content"))
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


def _render_message_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, dict):
        if "text" in content:
            return str(content.get("text") or "").strip()
        if "content" in content and isinstance(content.get("content"), str):
            return str(content.get("content") or "").strip()
        return json.dumps(content, ensure_ascii=True)
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
        return "\n".join(parts).strip()
    return str(content).strip()


def _build_openai_tool_call(
    *, call_id: str, name: str, arguments: str
) -> ChatCompletionMessageToolCall:
    """Build an OpenAI-compatible tool-call object for downstream handling."""
    return ChatCompletionMessageToolCall(
        id=call_id,
        call_id=call_id,
        response_item_id=None,
        type="function",
        function=Function(name=name, arguments=arguments),
    )


def _extract_tool_calls_from_text(
    text: str,
) -> tuple[list[ChatCompletionMessageToolCall], str]:
    """Parse <tool_call> blocks out of the response text into OpenAI tool calls.

    Returns (tool_calls, cleaned_text_with_blocks_removed).
    """
    if not isinstance(text, str) or not text.strip():
        return [], ""

    extracted: list[ChatCompletionMessageToolCall] = []
    consumed_spans: list[tuple[int, int]] = []

    def _try_add_tool_call(raw_json: str) -> None:
        try:
            obj = json.loads(raw_json)
        except Exception:
            return
        if not isinstance(obj, dict):
            return
        fn = obj.get("function")
        if not isinstance(fn, dict):
            return
        fn_name = fn.get("name")
        if not isinstance(fn_name, str) or not fn_name.strip():
            return
        fn_args = fn.get("arguments", "{}")
        if not isinstance(fn_args, str):
            fn_args = json.dumps(fn_args, ensure_ascii=False)
        call_id = obj.get("id")
        if not isinstance(call_id, str) or not call_id.strip():
            call_id = f"claude_cli_call_{len(extracted) + 1}"
        extracted.append(
            _build_openai_tool_call(
                call_id=call_id, name=fn_name.strip(), arguments=fn_args
            )
        )

    for m in _TOOL_CALL_BLOCK_RE.finditer(text):
        _try_add_tool_call(m.group(1))
        consumed_spans.append((m.start(), m.end()))

    # Only try the bare-JSON fallback when no XML blocks were found.
    if not extracted:
        for m in _TOOL_CALL_JSON_RE.finditer(text):
            _try_add_tool_call(m.group(0))
            consumed_spans.append((m.start(), m.end()))

    if not consumed_spans:
        return extracted, text.strip()

    consumed_spans.sort()
    merged: list[tuple[int, int]] = []
    for start, end in consumed_spans:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))

    parts: list[str] = []
    cursor = 0
    for start, end in merged:
        if cursor < start:
            parts.append(text[cursor:start])
        cursor = max(cursor, end)
    if cursor < len(text):
        parts.append(text[cursor:])

    cleaned = "\n".join(p.strip() for p in parts if p and p.strip()).strip()
    return extracted, cleaned


def _completion_to_stream_chunks(completion: SimpleNamespace) -> list[SimpleNamespace]:
    """Convert a one-shot response into OpenAI-style stream chunks.

    Only used if a caller requests stream=True despite the provider being
    excluded from streaming in the conversation loop — kept for parity with
    the copilot-acp client so nothing crashes on an unexpected stream request.
    """
    choice = completion.choices[0]
    message = choice.message
    tool_call_deltas = None
    if message.tool_calls:
        tool_call_deltas = []
        for index, tool_call in enumerate(message.tool_calls):
            tool_call_deltas.append(
                SimpleNamespace(
                    index=index,
                    id=getattr(tool_call, "id", None),
                    type=getattr(tool_call, "type", "function"),
                    function=SimpleNamespace(
                        name=getattr(tool_call.function, "name", None),
                        arguments=getattr(tool_call.function, "arguments", None),
                    ),
                )
            )
    delta = SimpleNamespace(
        role="assistant",
        content=message.content or None,
        tool_calls=tool_call_deltas,
        reasoning_content=message.reasoning_content,
        reasoning=message.reasoning,
    )
    data_chunk = SimpleNamespace(
        choices=[SimpleNamespace(index=0, delta=delta, finish_reason=choice.finish_reason)],
        model=completion.model,
        usage=None,
    )
    usage_chunk = SimpleNamespace(choices=[], model=completion.model, usage=completion.usage)
    return [data_chunk, usage_chunk]


def _normalize_timeout(timeout: Any) -> float:
    if timeout is None:
        return _DEFAULT_TIMEOUT_SECONDS
    if isinstance(timeout, (int, float)):
        return float(timeout)
    candidates = [
        getattr(timeout, attr, None)
        for attr in ("read", "write", "connect", "pool", "timeout")
    ]
    numeric = [float(v) for v in candidates if isinstance(v, (int, float))]
    return max(numeric) if numeric else _DEFAULT_TIMEOUT_SECONDS


class _ClaudeChatCompletions:
    def __init__(self, client: "ClaudeCLIClient"):
        self._client = client

    def create(self, **kwargs: Any) -> Any:
        return self._client._create_chat_completion(**kwargs)


class _ClaudeChatNamespace:
    def __init__(self, client: "ClaudeCLIClient"):
        self.completions = _ClaudeChatCompletions(client)


class ClaudeCLIClient:
    """Minimal OpenAI-client-compatible facade for `claude -p`."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        default_headers: dict[str, str] | None = None,
        command: str | None = None,
        args: list[str] | None = None,
        acp_command: str | None = None,
        acp_args: list[str] | None = None,
        acp_cwd: str | None = None,
        cwd: str | None = None,
        **_: Any,
    ):
        self.api_key = api_key or "claude-cli"
        self.base_url = base_url or CLAUDE_CLI_MARKER_BASE_URL
        self._default_headers = dict(default_headers or {})
        self._command = command or acp_command or _resolve_command()
        self._extra_args = list(args or acp_args or _resolve_extra_args())
        self._cwd = str(Path(cwd or acp_cwd or os.getcwd()).resolve())
        self.chat = _ClaudeChatNamespace(self)
        self.is_closed = False

    def close(self) -> None:
        # One-shot subprocess per request (spawned and reaped inside
        # _run_prompt), so there is no long-lived process to tear down.
        self.is_closed = True

    def _build_argv(self, *, model: str, effort: str) -> list[str]:
        argv = [
            self._command,
            "-p",
            # stream-json surfaces the assistant's content blocks (text +
            # native tool_use) even when the turn ends in error_max_turns.
            "--output-format",
            "stream-json",
            "--verbose",
            "--model",
            model,
            "--effort",
            effort,
            # Disable every built-in tool: claude acts as a pure model endpoint,
            # never executing anything. Hermes runs the tools it parses out.
            "--tools",
            "",
            # Neutralize Claude Code's own agent context so it behaves as a bare
            # model. NOT --bare (that would force ANTHROPIC_API_KEY and skip the
            # OAuth session we specifically want to use).
            "--system-prompt",
            _ORCHESTRATOR_SYSTEM_PROMPT,
            "--exclude-dynamic-system-prompt-sections",
            "--disable-slash-commands",
            # Exactly one assistant turn — this is a stateless model endpoint,
            # not an agent. If the model calls a tool, the turn cap trips
            # error_max_turns, which _run_prompt ignores (the assistant message
            # with the tool_use block was already streamed).
            "--max-turns",
            "1",
        ]
        argv.extend(self._extra_args)
        return argv

    def _create_chat_completion(
        self,
        *,
        model: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        timeout: float | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        stream: bool = False,
        extra_body: dict[str, Any] | None = None,
        **_: Any,
    ) -> Any:
        # Live model + effort: both are read per call so a mid-session `/model`
        # or `/effort` change takes effect on the very next turn. Effort arrives
        # via extra_body from ClaudeCLIProfile.build_extra_body (a CLI flag, not
        # an API field); model is the standard OpenAI `model` kwarg.
        resolved_model = _normalize_model(model)
        effort_hint = None
        if isinstance(extra_body, dict):
            effort_hint = extra_body.get("_hermes_claude_effort")
        resolved_effort = _normalize_effort(effort_hint)

        prompt_text = _format_messages_as_prompt(
            messages or [], tools=tools, tool_choice=tool_choice
        )
        text, reasoning_text, native_tool_calls = self._run_prompt(
            prompt_text,
            model=resolved_model,
            effort=resolved_effort,
            timeout_seconds=_normalize_timeout(timeout),
        )

        # Prefer native tool_use blocks captured from the stream; fall back to
        # <tool_call> text blocks only if the model answered in text instead.
        if native_tool_calls:
            tool_calls = native_tool_calls
            cleaned_text = text.strip()
        else:
            tool_calls, cleaned_text = _extract_tool_calls_from_text(text)

        usage = SimpleNamespace(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            prompt_tokens_details=SimpleNamespace(cached_tokens=0),
        )
        assistant_message = SimpleNamespace(
            content=cleaned_text,
            tool_calls=tool_calls,
            reasoning=reasoning_text or None,
            reasoning_content=reasoning_text or None,
            reasoning_details=None,
        )
        finish_reason = "tool_calls" if tool_calls else "stop"
        choice = SimpleNamespace(message=assistant_message, finish_reason=finish_reason)
        completion = SimpleNamespace(
            choices=[choice], usage=usage, model=resolved_model
        )
        if stream:
            return _completion_to_stream_chunks(completion)
        return completion

    def _run_prompt(
        self, prompt_text: str, *, model: str, effort: str, timeout_seconds: float
    ) -> tuple[str, str, list[ChatCompletionMessageToolCall]]:
        """Spawn `claude -p`, feed the prompt on stdin, parse the stream.

        Returns (text, reasoning, native_tool_calls). A non-zero exit is only
        fatal if nothing usable was captured from the stream — a tool-calling
        turn legitimately exits with error_max_turns after emitting its
        assistant message.
        """
        argv = self._build_argv(model=model, effort=effort)
        try:
            proc = subprocess.run(
                argv,
                input=prompt_text,
                capture_output=True,
                text=True,
                cwd=self._cwd,
                env=_build_subprocess_env(),
                timeout=timeout_seconds,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Could not start Claude Code CLI command '{self._command}'. "
                "Install Claude Code or set HERMES_CLAUDE_CLI_COMMAND/CLAUDE_CLI_PATH."
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"`claude -p` timed out after {timeout_seconds:.0f}s."
            ) from exc

        text, reasoning, tool_calls, fatal_error = _parse_stream(proc.stdout)

        # Only surface an error when the stream yielded no assistant content at
        # all. error_max_turns after a tool_use block is expected and benign.
        if not text and not tool_calls:
            if fatal_error:
                raise RuntimeError(f"`claude -p` reported an error: {fatal_error[:500]}")
            stderr = (proc.stderr or "").strip()
            raise RuntimeError(
                f"`claude -p` exited with code {proc.returncode} and no output: "
                f"{stderr[:500] or '(no stderr)'}"
            )
        return text, reasoning, tool_calls


def _parse_stream(
    stdout: str,
) -> tuple[str, str, list[ChatCompletionMessageToolCall], str]:
    """Parse `claude -p --output-format stream-json` output.

    Collects assistant text blocks, thinking blocks (as reasoning), and native
    tool_use blocks (converted to OpenAI tool calls). Returns
    (text, reasoning, tool_calls, fatal_error_message).
    """
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: list[ChatCompletionMessageToolCall] = []
    fatal_error = ""

    for line in (stdout or "").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except Exception:
            continue
        if not isinstance(event, dict):
            continue
        etype = event.get("type")
        if etype == "assistant":
            message = event.get("message") or {}
            for block in message.get("content", []) or []:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype == "text":
                    t = block.get("text")
                    if isinstance(t, str) and t.strip():
                        text_parts.append(t)
                elif btype == "thinking":
                    t = block.get("thinking") or block.get("text")
                    if isinstance(t, str) and t.strip():
                        reasoning_parts.append(t)
                elif btype == "tool_use":
                    name = block.get("name")
                    if not isinstance(name, str) or not name.strip():
                        continue
                    call_id = block.get("id")
                    if not isinstance(call_id, str) or not call_id.strip():
                        call_id = f"claude_cli_call_{len(tool_calls) + 1}"
                    args = block.get("input")
                    args_str = json.dumps(args, ensure_ascii=False) if args is not None else "{}"
                    tool_calls.append(
                        _build_openai_tool_call(
                            call_id=call_id, name=name.strip(), arguments=args_str
                        )
                    )
        elif etype == "result":
            # error_max_turns is expected on a tool-calling turn; only record a
            # genuinely fatal error (e.g. auth failure) with no assistant output.
            if event.get("is_error") and event.get("subtype") != "error_max_turns":
                msg = event.get("result")
                errs = event.get("errors")
                if isinstance(msg, str) and msg.strip():
                    fatal_error = msg.strip()
                elif isinstance(errs, list) and errs:
                    fatal_error = "; ".join(str(e) for e in errs)
                else:
                    fatal_error = str(event.get("subtype") or "unknown error")

    return (
        "\n".join(p.strip() for p in text_parts if p.strip()).strip(),
        "\n".join(p.strip() for p in reasoning_parts if p.strip()).strip(),
        tool_calls,
        fatal_error,
    )
