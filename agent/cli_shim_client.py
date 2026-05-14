"""OpenAI-compatible shim that forwards Hermes requests to local CLIs.

Supports four sub-modes selected by the `model` field on each request:

  claude-sonnet-cli   -> `claude --print --model sonnet`
  claude-opus-cli     -> `claude --print --model opus`
  codex-gpt5-cli      -> `codex exec --model gpt-5`
  gemini-cli          -> delegates to CopilotACPClient-style ACP loop
                        against `gemini --acp` (full tool-use ACP path)

All paths share the OpenAI-shaped facade. claude/codex use a single-shot
subprocess.run(... input=prompt ...) and regex-extract tool calls from
the response text. gemini uses the full ACP protocol via the copilot
ACP loop logic so streaming tool-use works.

The class deliberately reuses _format_messages_as_prompt and
_extract_tool_calls_from_text from agent.copilot_acp_client.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import threading
from types import SimpleNamespace
from typing import Any

from agent.copilot_acp_client import (
    CopilotACPClient,
    _format_messages_as_prompt,
    _extract_tool_calls_from_text,
    _build_subprocess_env,
)

CLI_SHIM_BASE_URL = "cli://shim"
_DEFAULT_TIMEOUT_SECONDS = 900.0


# Per-model dispatch table.
# Each entry: (command_resolver, args_template, supports_acp)
def _dispatch_for_model(model: str) -> dict[str, Any]:
    m = (model or "").strip().lower()
    if m in ("claude-sonnet-cli", "claude-sonnet", "sonnet-cli"):
        return {
            "mode": "print",
            "command": "claude",
            "args": ["--print", "--model", "sonnet"],
            "label": "claude-sonnet-cli",
        }
    if m in ("claude-opus-cli", "claude-opus", "opus-cli"):
        return {
            "mode": "print",
            "command": "claude",
            "args": ["--print", "--model", "opus"],
            "label": "claude-opus-cli",
        }
    if m in ("codex-gpt5-cli", "codex-cli", "codex"):
        return {
            "mode": "print",
            "command": "codex",
            "args": ["exec", "--model", "gpt-5"],
            "label": "codex-gpt5-cli",
        }
    if m in ("gemini-cli", "gemini-acp", "gemini"):
        return {
            "mode": "acp",
            "command": "gemini",
            "args": ["--acp"],
            "label": "gemini-cli",
        }
    # Default: treat as a hint for `claude --model <model>`.
    return {
        "mode": "print",
        "command": "claude",
        "args": ["--print", "--model", model],
        "label": f"claude-{model}-cli",
    }


def _resolve_timeout(timeout: Any) -> float:
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


class _CliShimChatCompletions:
    def __init__(self, client: "CliShimClient"):
        self._client = client

    def create(self, **kwargs: Any) -> Any:
        return self._client._create_chat_completion(**kwargs)


class _CliShimChatNamespace:
    def __init__(self, client: "CliShimClient"):
        self.completions = _CliShimChatCompletions(client)


class CliShimClient:
    """Minimal OpenAI-client-compatible facade for local CLIs.

    The `model` field on each request selects the underlying CLI; the
    constructor's `model` is a default fallback.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        default_headers: dict[str, str] | None = None,
        model: str | None = None,
        command: str | None = None,
        args: list[str] | None = None,
        cwd: str | None = None,
        **_: Any,
    ):
        self.api_key = api_key or "cli-shim"
        self.base_url = base_url or CLI_SHIM_BASE_URL
        self._default_headers = dict(default_headers or {})
        self._default_model = model or "claude-sonnet-cli"
        self._command_override = command
        self._args_override = list(args) if args else None
        self._cwd = cwd or os.getcwd()
        self.chat = _CliShimChatNamespace(self)
        self.is_closed = False
        self._active_process: subprocess.Popen[str] | None = None
        self._active_process_lock = threading.Lock()

    def close(self) -> None:
        self.is_closed = True
        proc: subprocess.Popen[str] | None
        with self._active_process_lock:
            proc = self._active_process
            self._active_process = None
        if proc is None:
            return
        try:
            proc.terminate()
            proc.wait(timeout=2)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

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
        effective_model = model or self._default_model
        dispatch = _dispatch_for_model(effective_model)

        prompt_text = _format_messages_as_prompt(
            messages or [],
            model=effective_model,
            tools=tools,
            tool_choice=tool_choice,
        )
        timeout_seconds = _resolve_timeout(timeout)

        if dispatch["mode"] == "acp":
            # Delegate to the ACP subprocess loop — reuse the copilot ACP
            # client's protocol, just pointed at the gemini binary.
            acp_client = CopilotACPClient(
                api_key="cli-shim-gemini",
                base_url="acp://gemini",
                command=self._command_override or dispatch["command"],
                args=self._args_override or dispatch["args"],
                acp_cwd=self._cwd,
            )
            try:
                response_text, reasoning_text = acp_client._run_prompt(
                    prompt_text, timeout_seconds=timeout_seconds
                )
            finally:
                acp_client.close()
        else:
            response_text, reasoning_text = self._run_print(
                prompt_text,
                command=self._command_override or dispatch["command"],
                args=self._args_override or dispatch["args"],
                timeout_seconds=timeout_seconds,
            )

        tool_calls, cleaned_text = _extract_tool_calls_from_text(response_text)

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
        return SimpleNamespace(
            choices=[choice],
            usage=usage,
            model=dispatch["label"],
        )

    def _run_print(
        self,
        prompt_text: str,
        *,
        command: str,
        args: list[str],
        timeout_seconds: float,
    ) -> tuple[str, str]:
        """Single-shot subprocess.run for --print/--exec style CLIs."""
        resolved = shutil.which(command) or command
        try:
            proc = subprocess.Popen(
                [resolved] + list(args),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=self._cwd,
                env=_build_subprocess_env(),
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Could not start CLI '{command}'. "
                f"Install it or check PATH."
            ) from exc

        with self._active_process_lock:
            self._active_process = proc
        self.is_closed = False

        try:
            stdout, stderr = proc.communicate(
                input=prompt_text, timeout=timeout_seconds
            )
        except subprocess.TimeoutExpired as exc:
            try:
                proc.kill()
            except Exception:
                pass
            raise TimeoutError(
                f"CLI '{command}' timed out after {timeout_seconds}s"
            ) from exc
        finally:
            with self._active_process_lock:
                self._active_process = None

        if proc.returncode != 0:
            tail = (stderr or "").strip()[-800:]
            raise RuntimeError(
                f"CLI '{command}' exited {proc.returncode}: {tail}"
            )

        return (stdout or "").strip(), ""
