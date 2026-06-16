"""Drop-in Anthropic-client shim that runs inference through the **official
Claude Code client** (`claude -p`), so Hermes can use a personal Claude
Max/Pro subscription without sending subscription tokens to api.anthropic.com
or spoofing a `claude-cli` user-agent.

The shim mimics the subset of ``anthropic.Anthropic`` that Hermes calls:

    client.messages.create(**anthropic_kwargs) -> Message-like
    client.close()

so the agent loop, ``AnthropicTransport.normalize_response``, validation, and
cache-stat extraction all run unchanged (they only read ``.content``,
``.stop_reason``, ``.usage``).

See ``docs/claude-code-provider.md`` for the design, the harness-mismatch
limits, and the compliance scope (personal-subscription use via the official
client). The tool half lives in ``agent/claude_code_bridge.py``.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Tools advertised by the bridge appear to the model under this prefix
# (``mcp__<server-key>__<tool>``). The server key is set in the --mcp-config
# file we write below; keep the two in sync.
_MCP_SERVER_KEY = "hermes"
_MCP_PREFIX = f"mcp__{_MCP_SERVER_KEY}__"

_DEFAULT_TIMEOUT_S = 600


class ClaudeCodeError(RuntimeError):
    """Raised when the `claude` CLI cannot produce a usable completion."""


# --- Message / block shims (quack like anthropic.types) ---------------------

class _Block:
    """Minimal stand-in for an Anthropic content block."""

    def __init__(self, data: Dict[str, Any]):
        self.type = data.get("type")
        self.text = data.get("text", "")
        self.thinking = data.get("thinking", "")
        self.signature = data.get("signature")
        self.data = data.get("data")
        self.id = data.get("id")
        self.name = data.get("name")
        self.input = data.get("input", {})
        self._raw = data

    # normalize_response runs _to_plain_data(block); give it the original dict.
    def model_dump(self, *args, **kwargs) -> Dict[str, Any]:  # noqa: D401
        return dict(self._raw)

    def dict(self, *args, **kwargs) -> Dict[str, Any]:
        return dict(self._raw)


class _Usage:
    def __init__(self, data: Optional[Dict[str, Any]]):
        data = data or {}
        self.input_tokens = data.get("input_tokens", 0) or 0
        self.output_tokens = data.get("output_tokens", 0) or 0
        self.cache_read_input_tokens = data.get("cache_read_input_tokens", 0) or 0
        self.cache_creation_input_tokens = (
            data.get("cache_creation_input_tokens", 0) or 0
        )


class _Message:
    def __init__(
        self,
        content: List[_Block],
        stop_reason: str,
        usage: Optional[Dict[str, Any]],
        model: str,
    ):
        self.id = "claude-code"
        self.role = "assistant"
        self.type = "message"
        self.content = content
        self.stop_reason = stop_reason
        self.stop_sequence = None
        self.model = model
        self.usage = _Usage(usage)


# --- Prompt serialization ---------------------------------------------------

def _system_text(system: Any) -> str:
    """Flatten an Anthropic `system` (str | list[text-block]) to plain text."""
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        parts = []
        for block in system:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "\n\n".join(p for p in parts if p)
    return ""


def _render_content(content: Any) -> str:
    """Render one message's content (str | list of blocks) to transcript text."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: List[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            parts.append(block.get("text", ""))
        elif btype == "thinking":
            # Prior reasoning is context, not output — include briefly.
            continue
        elif btype == "tool_use":
            name = block.get("name", "tool")
            args = json.dumps(block.get("input", {}), ensure_ascii=False)
            parts.append(f"[called tool `{name}` with input {args}]")
        elif btype == "tool_result":
            inner = block.get("content")
            rendered = _render_content(inner) if isinstance(inner, list) else (
                inner if isinstance(inner, str) else json.dumps(inner, ensure_ascii=False)
            )
            tool_id = block.get("tool_use_id", "")
            parts.append(f"[tool result for {tool_id}]: {rendered}")
        elif btype == "image":
            parts.append("[image omitted — not supported via Claude Code provider]")
    return "\n".join(p for p in parts if p)


def _serialize_messages(messages: List[Dict[str, Any]]) -> str:
    """Serialize the Anthropic message history into a single prompt.

    `claude -p` is stateless per invocation and has no first-class way to
    inject prior assistant tool_use + externally-produced tool_result blocks,
    so the full conversation is replayed as a text transcript (documented
    limitation — see docs/claude-code-provider.md). The latest user turn is
    surfaced last so it reads as the active request.
    """
    if not messages:
        return ""

    # A single leading user message with no prior context: pass it straight
    # through so simple one-shot turns aren't wrapped in transcript scaffolding.
    if len(messages) == 1 and messages[0].get("role") == "user":
        return _render_content(messages[0].get("content"))

    lines: List[str] = [
        "Continue this conversation. Respond only as the assistant's next turn.",
        "",
        "<conversation>",
    ]
    for msg in messages:
        role = msg.get("role", "user")
        rendered = _render_content(msg.get("content"))
        if not rendered:
            continue
        label = "Assistant" if role == "assistant" else "User"
        lines.append(f"{label}: {rendered}")
    lines.append("</conversation>")
    return "\n".join(lines)


# --- CLI invocation ---------------------------------------------------------

def resolve_claude_command() -> Optional[str]:
    """Resolve the `claude` binary path, honoring env overrides."""
    candidate = (
        os.getenv("HERMES_CLAUDE_CODE_COMMAND", "").strip()
        or os.getenv("CLAUDE_CODE_CLI_PATH", "").strip()
        or "claude"
    )
    return shutil.which(candidate) or (candidate if os.path.isabs(candidate) else None)


def _build_command(
    *,
    claude_bin: str,
    model: str,
    tools_file: Optional[str],
    mcp_config_file: Optional[str],
    system_file: Optional[str],
    allowed_tools: List[str],
) -> List[str]:
    cmd = [
        claude_bin,
        "-p",
        "--output-format", "stream-json",
        "--verbose",
        # One model turn only: the official client stops at the first tool_use
        # (stop_reason=tool_use) WITHOUT executing the tool, so Hermes keeps
        # ownership of tool execution.
        "--max-turns", "1",
        "--permission-mode", "default",
        # Disable Claude Code's built-in tools; the model may only see Hermes'
        # tools (exposed via the MCP bridge below).
        "--tools", "",
    ]
    if model:
        cmd += ["--model", _map_model(model)]
    if system_file:
        cmd += ["--system-prompt-file", system_file]
    if tools_file and mcp_config_file:
        cmd += [
            "--strict-mcp-config",
            "--mcp-config", mcp_config_file,
        ]
        if allowed_tools:
            cmd += ["--allowedTools", ",".join(allowed_tools)]
    extra = os.getenv("HERMES_CLAUDE_CODE_ARGS", "").strip()
    if extra:
        import shlex
        cmd += shlex.split(extra)
    return cmd


def _map_model(model: str) -> str:
    """Map a Hermes model id to something `claude --model` accepts.

    Passes through CLI aliases (sonnet/opus/haiku) and full Anthropic ids
    unchanged; only normalizes an empty/auto value.
    """
    m = (model or "").strip()
    if not m or m == "auto":
        return "sonnet"
    return m


def _parse_stream(stdout: str) -> _Message:
    """Parse `claude -p --output-format stream-json` output into a _Message."""
    blocks: List[_Block] = []
    stop_reason: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None
    model = ""
    result_error: Optional[str] = None

    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            evt = json.loads(line)
        except Exception:
            continue
        if not isinstance(evt, dict):
            continue
        etype = evt.get("type")
        if etype == "assistant":
            message = evt.get("message", {})
            model = message.get("model") or model
            for raw_block in message.get("content", []) or []:
                if not isinstance(raw_block, dict):
                    continue
                if raw_block.get("type") == "tool_use":
                    name = raw_block.get("name", "")
                    if name.startswith(_MCP_PREFIX):
                        raw_block = dict(raw_block)
                        raw_block["name"] = name[len(_MCP_PREFIX):]
                blocks.append(_Block(raw_block))
        elif etype == "result":
            stop_reason = evt.get("stop_reason") or stop_reason
            usage = evt.get("usage") or usage
            if evt.get("is_error") and evt.get("subtype") not in (
                "success", "error_max_turns"
            ):
                result_error = evt.get("subtype") or "error"

    if not blocks and result_error:
        raise ClaudeCodeError(f"claude -p failed: {result_error}")

    if stop_reason is None:
        stop_reason = (
            "tool_use"
            if any(b.type == "tool_use" for b in blocks)
            else "end_turn"
        )
    # If the turn was cut at a tool boundary, Anthropic semantics expect
    # tool_use as the stop reason regardless of the harness's max-turns marker.
    if any(b.type == "tool_use" for b in blocks):
        stop_reason = "tool_use"

    return _Message(content=blocks, stop_reason=stop_reason, usage=usage, model=model)


class _Messages:
    """Mirrors `client.messages` — only `.create()` is used (non-streaming)."""

    def create(self, **kwargs: Any) -> _Message:
        claude_bin = resolve_claude_command()
        if not claude_bin:
            raise ClaudeCodeError(
                "The `claude` CLI was not found on PATH. Install Claude Code and "
                "run `claude /login` (or `claude setup-token`), or set "
                "HERMES_CLAUDE_CODE_COMMAND / CLAUDE_CODE_CLI_PATH."
            )

        model = kwargs.get("model", "")
        messages = kwargs.get("messages", []) or []
        system = kwargs.get("system")
        tools = kwargs.get("tools") or []

        prompt = _serialize_messages(messages)
        system_text = _system_text(system)

        tmpdir = tempfile.mkdtemp(prefix="hermes-claude-code-")
        tools_file = None
        mcp_config_file = None
        system_file = None
        allowed_tools: List[str] = []
        try:
            if system_text:
                system_file = os.path.join(tmpdir, "system.txt")
                with open(system_file, "w", encoding="utf-8") as fh:
                    fh.write(system_text)
            if tools:
                tools_file = os.path.join(tmpdir, "tools.json")
                with open(tools_file, "w", encoding="utf-8") as fh:
                    json.dump(tools, fh)
                bridge_mod = "agent.claude_code_bridge"
                mcp_config = {
                    "mcpServers": {
                        _MCP_SERVER_KEY: {
                            "command": _python_executable(),
                            "args": ["-m", bridge_mod, tools_file],
                        }
                    }
                }
                mcp_config_file = os.path.join(tmpdir, "mcp.json")
                with open(mcp_config_file, "w", encoding="utf-8") as fh:
                    json.dump(mcp_config, fh)
                allowed_tools = [
                    _MCP_PREFIX + t["name"]
                    for t in tools
                    if isinstance(t, dict) and t.get("name")
                ]

            cmd = _build_command(
                claude_bin=claude_bin,
                model=model,
                tools_file=tools_file,
                mcp_config_file=mcp_config_file,
                system_file=system_file,
                allowed_tools=allowed_tools,
            )

            timeout_s = _timeout_seconds()
            try:
                proc = subprocess.run(
                    cmd,
                    input=prompt,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    timeout=timeout_s,
                    # Ensure the bridge subprocess imports `agent.*` from this repo.
                    env=_bridge_env(),
                )
            except subprocess.TimeoutExpired as exc:
                raise ClaudeCodeError(
                    f"`claude -p` timed out after {timeout_s}s"
                ) from exc
            except FileNotFoundError as exc:
                raise ClaudeCodeError(f"Could not launch `claude`: {exc}") from exc

            message = _parse_stream(proc.stdout or "")
            if not message.content and (proc.returncode or 0) != 0:
                raise ClaudeCodeError(
                    "`claude -p` produced no output "
                    f"(exit {proc.returncode}): {(proc.stderr or '').strip()[:500]}"
                )
            return message
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


def _python_executable() -> str:
    import sys
    return sys.executable or "python3"


def _bridge_env() -> Dict[str, str]:
    """Env for the claude subprocess: ensure the bridge can import `agent.*`."""
    env = dict(os.environ)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    existing = env.get("PYTHONPATH", "")
    parts = [repo_root] + ([existing] if existing else [])
    env["PYTHONPATH"] = os.pathsep.join(parts)
    return env


def _timeout_seconds() -> int:
    raw = os.getenv("HERMES_CLAUDE_CODE_TIMEOUT", "").strip()
    if raw.isdigit():
        return int(raw)
    return _DEFAULT_TIMEOUT_S


class ClaudeCodeClient:
    """Anthropic-SDK-shaped client backed by the official `claude` CLI."""

    def __init__(self, model: str = "", **_ignored: Any):
        self.model = model
        self.messages = _Messages()

    def close(self) -> None:  # parity with anthropic.Anthropic
        return None


def build_claude_code_client(model: str = "", **kwargs: Any) -> ClaudeCodeClient:
    return ClaudeCodeClient(model=model, **kwargs)
