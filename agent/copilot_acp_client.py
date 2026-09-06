"""OpenAI-compatible shim that forwards Hermes requests to `copilot --acp`.

Each request starts a short-lived ACP session, sends the formatted conversation
as one prompt, collects text chunks, and returns the minimal OpenAI-client shape.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import queue
import re
import shlex
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from agent.acp_openai_bridge import (
    completion_to_stream_chunks as _completion_to_stream_chunks,
    extract_tool_calls_from_text as _extract_tool_calls_from_text,
    render_tool_bridge_sections as _render_tool_bridge_sections,
)
from agent.file_safety import get_read_block_error, get_write_denied_error, is_write_approval_required
from agent.redact import redact_sensitive_text
from tools.environments.local import hermes_subprocess_env

ACP_MARKER_BASE_URL = "acp://copilot"
logger = logging.getLogger(__name__)
_DEFAULT_TIMEOUT_SECONDS = 900.0
# Stderr fingerprint of the deprecated `gh copilot` extension. Require BOTH the product name
# AND a deprecation marker: the NEW `@github/copilot` CLI legitimately mentions "copilot-cli".
_DEPRECATION_REQUIRED = ("gh-copilot",)
_DEPRECATION_MARKERS = ("has been deprecated", "no commands will be executed")
_ROLE_LABELS = {"system": "System", "user": "User", "assistant": "Assistant", "tool": "Tool", "context": "Context"}
# Probe verdicts per binary path (~50ms --help paid once per process). Only definitive
# True/False is cached, so a CLI installed mid-session is picked up.
_ACP_PROBE_CACHE: dict[str, bool] = {}
_PROMPT_PREAMBLE = (
    "You are being used as the active ACP agent backend for Hermes.",
    "Use ACP capabilities to complete tasks.",
    "IMPORTANT: If you take an action with a tool, you MUST output tool calls using <tool_call>{...}</tool_call> blocks with JSON exactly in OpenAI function-call shape.",
    "If no tool is needed, answer normally.",
)
_INITIALIZE_PARAMS = {
    "protocolVersion": 1,
    "clientCapabilities": {"fs": {"readTextFile": True, "writeTextFile": True}},
    "clientInfo": {"name": "hermes-agent", "title": "Hermes Agent", "version": "0.0.0"},
}
_DEPRECATED_CLI_ERROR = (
    "Hermes ACP mode requires the NEW GitHub Copilot CLI (github.com/github/copilot-cli), but the binary it just "
    "spawned is the deprecated `gh copilot` extension.\n\n"
    "Install the new CLI:\n  npm install -g @github/copilot\n  # then verify with: copilot --help\n\n"
    "If `copilot` already resolves to the new CLI but you still see this,\npoint Hermes at it explicitly:\n"
    "  export HERMES_COPILOT_ACP_COMMAND=/path/to/new/copilot\n\n"
    "Alternative: use the `copilot` provider (no ACP, hits the Copilot API\ndirectly with a Copilot subscription "
    "token) via `hermes setup`.\n\nOriginal error:\n"
)


def _is_gh_copilot_deprecation_message(stderr_text: str) -> bool:
    """True iff stderr looks like the deprecated gh-copilot extension's banner."""
    lower = stderr_text.lower()
    return any(req in lower for req in _DEPRECATION_REQUIRED) and any(m in lower for m in _DEPRECATION_MARKERS)


def _resolve_command() -> str:
    return os.getenv("HERMES_COPILOT_ACP_COMMAND", "").strip() or os.getenv("COPILOT_CLI_PATH", "").strip() or "copilot"


def _resolve_args() -> list[str]:
    return shlex.split(os.getenv("HERMES_COPILOT_ACP_ARGS", "").strip()) or ["--acp", "--stdio"]


def _acp_supported(command: str, args: list[str]) -> bool | None:
    """Tri-state ``--acp`` probe (a CLI without the flag exits 1 and the parent would wait the
    full child timeout for stdout that never arrives). True = help advertises --acp; False =
    help ran cleanly without it (caller fast-fails); None = inconclusive (binary missing /
    --help failed → normal spawn error). Skipped when ``--acp`` is not in ``args`` (custom transport)."""
    if "--acp" not in args:
        return True
    if (cached := _ACP_PROBE_CACHE.get(command)) is not None:
        return cached
    try:
        probe = subprocess.run(
            [command, "--help"], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=5,
            stdin=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if probe.returncode != 0:
        return None
    # ``--acp`` as a flag token; tolerate spacing and ``[--acp]`` variants.
    verdict = _ACP_PROBE_CACHE[command] = bool(re.search(r"(?:^|[\s\[])--acp(?:[\s=\],]|$)", probe.stdout, re.MULTILINE))
    return verdict


def _resolve_home_dir() -> str:
    """Stable HOME for child ACP processes; /tmp as a last resort so the child never starts HOME-less."""
    if home := os.environ.get("HOME", "").strip():
        return home
    if (expanded := os.path.expanduser("~")) and expanded != "~":
        return expanded
    try:
        import pwd

        return pwd.getpwuid(os.getuid()).pw_dir.strip() or "/tmp"  # windows-footgun: ok — POSIX fallback inside try/except (pwd import fails on Windows)
    except Exception:
        return "/tmp"


def _build_subprocess_env() -> dict[str, str]:
    from hermes_constants import apply_subprocess_home_env

    # Copilot ACP drives a model and needs LLM provider credentials; the central helper still
    # strips Tier-1 secrets (bot tokens, GitHub auth, infra).
    # See #29157.
    env = hermes_subprocess_env(inherit_credentials=True)
    env["HOME"] = _resolve_home_dir()
    apply_subprocess_home_env(env)
    return env


def _jsonrpc_result(message_id: Any, result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": message_id, "result": result}


def _jsonrpc_error(message_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": message_id, "error": {"code": code, "message": message}}


def _enabled_ids(entries: Any, key: str) -> set[str]:
    """Ids of ``entries`` (dicts) whose ``_meta.copilotEnablement`` is not ``disabled``."""
    return {str(e.get(key) or "").strip() for e in (entries or []) if isinstance(e, dict)
            and str((e.get("_meta") or {}).get("copilotEnablement") or "").strip().lower() != "disabled"}


def _model_selection_request(session: dict[str, Any], requested_model: str) -> tuple[str, dict[str, str]] | None:
    """ACP request selecting ``requested_model`` for ``session``: stable v1
    ``session/set_config_option``, else Copilot's pre-stabilization ``session/set_model``
    when no model config option is advertised. A reported model list is authoritative:
    unknown and policy-disabled ids return None instead of being sent."""
    session_id = str(session.get("sessionId") or "").strip()
    requested_model = str(requested_model or "").strip()
    if not session_id or not requested_model or requested_model == "copilot-acp":
        return None
    options = [o for o in (session.get("configOptions") or []) if isinstance(o, dict) and "model" in (o.get("category"), o.get("id"))]
    if options:
        if requested_model not in _enabled_ids(options[0].get("options"), "value"):
            return None
        return "session/set_config_option", {"sessionId": session_id, "configId": str(options[0].get("id") or "model"), "value": requested_model}
    available = _enabled_ids((session.get("models") or {}).get("availableModels"), "modelId")
    return None if available and requested_model not in available else ("session/set_model", {"sessionId": session_id, "modelId": requested_model})


def _format_messages_as_prompt(
    messages: list[dict[str, Any]], model: str | None = None, tools: list[dict[str, Any]] | None = None, tool_choice: Any = None,
) -> str:
    # Deliberately no "requested model" line: the model is applied for real via ACP session/set_model;
    # a prompt-text mention makes a substituted backend model FALSELY self-identify as the requested
    # one. Copilot has no tools of its own that collide with Hermes', so forward the whole toolset.
    sections: list[str] = [*_PROMPT_PREAMBLE, *_render_tool_bridge_sections(tools, tool_choice)]
    transcript: list[str] = []
    for message in (m for m in messages if isinstance(m, dict)):
        role = str(message.get("role") or "unknown").strip().lower()
        if rendered := _render_message_content(message.get("content")):
            transcript.append(f"{_ROLE_LABELS.get(role, 'Context')}:\n{rendered}")
    if transcript:
        sections.append("Conversation transcript:\n\n" + "\n\n".join(transcript))
    sections.append("Continue the conversation from the latest user request.")
    return "\n\n".join(section.strip() for section in sections if section and section.strip())


def _render_message_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, dict):
        if "text" in content:
            return str(content.get("text") or "").strip()
        return content["content"].strip() if isinstance(content.get("content"), str) else json.dumps(content, ensure_ascii=True)
    if isinstance(content, list):
        parts = [item if isinstance(item, str) else item["text"].strip() for item in content if isinstance(item, str)
                 or (isinstance(item, dict) and isinstance(item.get("text"), str) and item["text"].strip())]
        return "\n".join(parts).strip()
    return str(content).strip()


def _ensure_path_within_cwd(path_text: str, cwd: str) -> Path:
    if not Path(path_text).is_absolute():
        raise PermissionError("ACP file-system paths must be absolute.")
    resolved, root = Path(path_text).resolve(), Path(cwd).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise PermissionError(f"Path '{resolved}' is outside the session cwd '{root}'.") from exc
    return resolved


def _effective_timeout(timeout: Any) -> float:
    """Normalise a float or httpx.Timeout-like object to wall-clock seconds (largest component wins)."""
    if isinstance(timeout, (int, float)):
        return float(timeout)
    candidates = [getattr(timeout, attr, None) for attr in ("read", "write", "connect", "pool", "timeout")]
    return max((float(v) for v in candidates if isinstance(v, (int, float))), default=_DEFAULT_TIMEOUT_SECONDS)


def _fs_read_text_file(params: dict[str, Any], cwd: str) -> Any:
    path = _ensure_path_within_cwd(str(params.get("path") or ""), cwd)
    if block_error := get_read_block_error(str(path)):
        raise PermissionError(block_error)
    try:
        content = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        content = ""
    line, limit = params.get("line"), params.get("limit")
    if isinstance(line, int) and line > 1:
        end = line - 1 + limit if isinstance(limit, int) and limit > 0 else None
        content = "".join(content.splitlines(keepends=True)[line - 1:end])
    return {"content": redact_sensitive_text(content, force=True) if content else content}


def _fs_write_text_file(params: dict[str, Any], cwd: str) -> Any:
    path = _ensure_path_within_cwd(str(params.get("path") or ""), cwd)
    if denied := get_write_denied_error(str(path)):
        raise PermissionError(denied)
    if is_write_approval_required(str(path)):  # soft-gated for interactive tools; the ACP shim has no human channel → fail closed
        raise PermissionError(f"Write denied: '{path}' requires interactive approval and cannot be written through the ACP file bridge.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(params.get("content") or ""), encoding="utf-8")
    return None


_FS_HANDLERS = {"fs/read_text_file": _fs_read_text_file, "fs/write_text_file": _fs_write_text_file}


class CopilotACPClient:
    """Minimal OpenAI-client-compatible facade for Copilot ACP."""

    # Declared for agent/auxiliary_client.py: this shim drives an ACP subprocess over stdio, so it is
    # already a complete client (never re-dispatch through a wire adapter) and async-safe as-is.
    HERMES_SKIP_TRANSPORT_WRAP = True
    HERMES_SKIP_ASYNC_WRAP = True

    def __init__(
        self, *, api_key: str | None = None, base_url: str | None = None, default_headers: dict[str, str] | None = None,
        acp_command: str | None = None, acp_args: list[str] | None = None, acp_cwd: str | None = None, command: str | None = None,
        args: list[str] | None = None, **_: Any,
    ):
        self.api_key, self.base_url = api_key or "copilot-acp", base_url or ACP_MARKER_BASE_URL
        self._default_headers = dict(default_headers or {})
        self._acp_command = acp_command or command or _resolve_command()
        self._acp_args = list(acp_args or args or _resolve_args())
        self._acp_cwd = str(Path(acp_cwd or os.getcwd()).resolve())
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create_chat_completion))
        self.is_closed, self._active_process = False, None
        self._active_process_lock = threading.Lock()

    def close(self) -> None:
        with self._active_process_lock:
            proc, self._active_process = self._active_process, None
        self.is_closed = True
        try:
            if proc is not None:
                proc.terminate()
                proc.wait(timeout=2)
        except Exception:
            with contextlib.suppress(Exception):
                proc.kill()

    def _create_chat_completion(
        self, *, model: str | None = None, messages: list[dict[str, Any]] | None = None, timeout: float | None = None,
        tools: list[dict[str, Any]] | None = None, tool_choice: Any = None, stream: bool = False, **_: Any,
    ) -> Any:
        prompt_text = _format_messages_as_prompt(messages or [], model=model, tools=tools, tool_choice=tool_choice)
        response_text, reasoning = self._run_prompt(prompt_text, timeout_seconds=_effective_timeout(timeout), model=model)
        tool_calls, cleaned_text = _extract_tool_calls_from_text(response_text)
        message = SimpleNamespace(
            content=cleaned_text, tool_calls=tool_calls, reasoning=reasoning or None, reasoning_content=reasoning or None,
            reasoning_details=None,
        )
        completion = SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="tool_calls" if tool_calls else "stop")],
            usage=SimpleNamespace(prompt_tokens=0, completion_tokens=0, total_tokens=0, prompt_tokens_details=SimpleNamespace(cached_tokens=0)),
            model=model or "copilot-acp",
        )
        return _completion_to_stream_chunks(completion) if stream else completion

    def _spawn(self) -> subprocess.Popen[str]:
        # Fast-fail when the CLI rejects --acp (else the parent waits the full child timeout for stdout that
        # never arrives). ``None`` falls through to the spawn's established start error.
        if _acp_supported(self._acp_command, self._acp_args) is False:
            preview = " ".join(self._acp_args[:3]) if self._acp_args else "(none)"
            raise RuntimeError(
                f"ACP transport not supported by '{self._acp_command}': `{preview}` is rejected as an unknown option. This "
                "usually means the CLI is an older release (e.g. Claude Code v2.x) or a different tool than expected. Either "
                "install a CLI that ships with --acp support (e.g. `@github/copilot` late 2025+), or set "
                "HERMES_COPILOT_ACP_COMMAND / HERMES_COPILOT_ACP_ARGS to a working pair."
            )
        try:
            from hermes_cli._subprocess_compat import windows_hide_flags  # hide the Windows console flash (#56747); pipes intact for the ACP wire

            # Hide the console the CLI child would otherwise flash on Windows (#56747). Hide-only — stdio
            # pipes stay intact for the ACP wire.
            proc = subprocess.Popen(
                [self._acp_command] + self._acp_args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, encoding='utf-8', errors='replace', bufsize=1, cwd=self._acp_cwd, env=_build_subprocess_env(),
                creationflags=windows_hide_flags(),
            )
        except FileNotFoundError as exc:
            raise RuntimeError(f"Could not start Copilot ACP command '{self._acp_command}'. Install GitHub Copilot CLI or set "
                               "HERMES_COPILOT_ACP_COMMAND/COPILOT_CLI_PATH.") from exc
        if proc.stdin is None or proc.stdout is None:
            proc.kill()
            raise RuntimeError("Copilot ACP process did not expose stdin/stdout pipes.")
        self.is_closed = False
        with self._active_process_lock:
            self._active_process = proc
        return proc

    def _run_prompt(self, prompt_text: str, *, timeout_seconds: float, model: str | None = None) -> tuple[str, str]:
        # The CLI's `--model` spawn flag is deliberately NOT used: `copilot --acp` validates it (unknown id
        # aborts the spawn) but ignores it for the session; the model is applied after session/new instead.
        requested_model = str(model or "").strip()
        proc = self._spawn()
        inbox: queue.Queue[dict[str, Any]] = queue.Queue()
        stderr_tail: deque[str] = deque(maxlen=40)

        def _decode(line: str) -> dict[str, Any]:
            try:
                return json.loads(line)
            except Exception:
                return {"raw": line.rstrip("\n")}

        def _pump(stream, sink) -> None:
            for line in stream or ():
                sink(line)

        threading.Thread(target=_pump, args=(proc.stdout, lambda line: inbox.put(_decode(line))), daemon=True).start()
        threading.Thread(target=_pump, args=(proc.stderr, lambda line: stderr_tail.append(line.rstrip("\n"))), daemon=True).start()
        request_ids = iter(range(1, 1 << 62))

        def _request(method: str, params: dict[str, Any], *, text_parts: list[str] | None = None, reasoning_parts: list[str] | None = None) -> Any:
            request_id = next(request_ids)
            proc.stdin.write(json.dumps({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}) + "\n")
            proc.stdin.flush()
            deadline = time.monotonic() + timeout_seconds
            while time.monotonic() < deadline and proc.poll() is None:
                try:
                    msg = inbox.get(timeout=0.1)
                except queue.Empty:
                    continue
                if self._handle_server_message(
                    msg, process=proc, cwd=self._acp_cwd, text_parts=text_parts, reasoning_parts=reasoning_parts
                ) or msg.get("id") != request_id:
                    continue
                if "error" in msg:
                    err = msg.get("error") or {}
                    raise RuntimeError(f"Copilot ACP {method} failed: {err.get('message') or err}")
                return msg.get("result")
            stderr_text = "\n".join(stderr_tail).strip()
            if proc.poll() is not None and stderr_text:
                if _is_gh_copilot_deprecation_message(stderr_text):
                    raise RuntimeError(_DEPRECATED_CLI_ERROR + stderr_text)
                raise RuntimeError(f"Copilot ACP process exited early: {stderr_text}")
            raise TimeoutError(f"Timed out waiting for Copilot ACP response to {method}.")

        try:
            _request("initialize", _INITIALIZE_PARAMS)
            session = _request("session/new", {"cwd": self._acp_cwd, "mcpServers": []}) or {}
            session_id = str(session.get("sessionId") or "").strip()
            if not session_id:
                raise RuntimeError("Copilot ACP did not return a sessionId.")
            if requested_model and requested_model != "copilot-acp":
                try:
                    if (selection := _model_selection_request(session, requested_model)) is not None:
                        _request(*selection)
                    else:
                        logger.warning("Copilot ACP does not offer model %r; using the session default.", requested_model)
                except Exception as exc:
                    logger.warning("Copilot ACP model selection for %r failed; continuing with the session default: %s", requested_model, exc)
            text_parts: list[str] = []
            reasoning_parts: list[str] = []
            prompt = {"sessionId": session_id, "prompt": [{"type": "text", "text": prompt_text}]}
            _request("session/prompt", prompt, text_parts=text_parts, reasoning_parts=reasoning_parts)
            return "".join(text_parts), "".join(reasoning_parts)
        finally:
            self.close()

    def _handle_server_message(
        self, msg: dict[str, Any], *, process: subprocess.Popen[str], cwd: str, text_parts: list[str] | None, reasoning_parts: list[str] | None,
    ) -> bool:
        """Consume a server->client message; True when handled (notification or request answered)."""
        method = msg.get("method")
        if not isinstance(method, str):
            return False
        if method == "session/update":
            update = (msg.get("params") or {}).get("update") or {}
            content = update.get("content") or {}
            chunk_text = str(content.get("text") or "") if isinstance(content, dict) else ""
            sinks = {"agent_message_chunk": text_parts, "agent_thought_chunk": reasoning_parts}
            if chunk_text and (sink := sinks.get(str(update.get("sessionUpdate") or "").strip())) is not None:
                sink.append(chunk_text)
            return True
        if process.stdin is None:
            return True
        message_id = msg.get("id")
        if method == "session/request_permission":
            response = _jsonrpc_result(message_id, {"outcome": {"outcome": "cancelled"}})
        elif method in _FS_HANDLERS:
            try:
                response = _jsonrpc_result(message_id, _FS_HANDLERS[method](msg.get("params") or {}, cwd))
            except Exception as exc:
                response = _jsonrpc_error(message_id, -32602, str(exc))
        else:
            response = _jsonrpc_error(message_id, -32601, f"ACP client method '{method}' is not supported by Hermes yet.")
        process.stdin.write(json.dumps(response) + "\n")
        process.stdin.flush()
        return True
