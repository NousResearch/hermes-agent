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

# Concurrency caps — prevent the 27-agent fleet from OOM-killing the host
# by spawning 27 simultaneous claude/codex subprocesses (~500MB each).
# Override via env vars for tuning without code change.
_GLOBAL_MAX = int(os.environ.get("HERMES_CLI_SHIM_GLOBAL_MAX", "6"))
_PER_CLI_MAX = {
    "claude": int(os.environ.get("HERMES_CLI_SHIM_CLAUDE_MAX", "3")),
    "codex":  int(os.environ.get("HERMES_CLI_SHIM_CODEX_MAX",  "4")),
    "gemini": int(os.environ.get("HERMES_CLI_SHIM_GEMINI_MAX", "3")),
}
_GLOBAL_SEMAPHORE = threading.BoundedSemaphore(_GLOBAL_MAX)
_PER_CLI_SEMAPHORES: dict[str, threading.BoundedSemaphore] = {
    cli: threading.BoundedSemaphore(cap) for cli, cap in _PER_CLI_MAX.items()
}
_SEMAPHORE_ACQUIRE_TIMEOUT = float(
    os.environ.get("HERMES_CLI_SHIM_QUEUE_TIMEOUT", "120")
)


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
    if m in ("codex-gpt5-cli", "codex-cli", "codex", "codex-gpt55-cli", "codex-gpt5.5-cli"):
        # ChatGPT $200 Pro subscription — unrestricted gpt-5.5 access.
        # Pinning the model explicitly ensures we get the latest codex model
        # regardless of the OAuth account's interactive default.
        # We use --output-last-message to get clean output instead of the
        # interactive scaffold; the temp-file path is filled in at call time.
        return {
            "mode": "codex_exec",
            "command": "codex",
            "args": ["exec", "--skip-git-repo-check", "--model", "gpt-5.5"],
            "label": "codex-gpt5.5-cli",
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


class _ConcurrencyGate:
    """Acquire both global + per-CLI semaphores or raise quickly.

    Prevents the 27-agent fleet from spawning 27 simultaneous CLI subprocesses
    and OOM-killing the host. If we can't acquire within the queue timeout,
    we raise so the agent fails fast and tries the next provider in the chain.
    """

    def __init__(self, cli_name: str) -> None:
        self._cli_name = cli_name
        self._per_cli = _PER_CLI_SEMAPHORES.get(cli_name)
        self._holding_global = False
        self._holding_per_cli = False

    def __enter__(self) -> "_ConcurrencyGate":
        if not _GLOBAL_SEMAPHORE.acquire(timeout=_SEMAPHORE_ACQUIRE_TIMEOUT):
            raise RuntimeError(
                f"cli-shim global concurrency cap reached "
                f"({_GLOBAL_MAX}); queue timeout {_SEMAPHORE_ACQUIRE_TIMEOUT}s "
                f"exceeded waiting for slot."
            )
        self._holding_global = True
        if self._per_cli is not None:
            if not self._per_cli.acquire(timeout=_SEMAPHORE_ACQUIRE_TIMEOUT):
                _GLOBAL_SEMAPHORE.release()
                self._holding_global = False
                raise RuntimeError(
                    f"cli-shim per-CLI cap reached for '{self._cli_name}' "
                    f"({_PER_CLI_MAX.get(self._cli_name)}); queue timeout "
                    f"{_SEMAPHORE_ACQUIRE_TIMEOUT}s exceeded."
                )
            self._holding_per_cli = True
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._holding_per_cli and self._per_cli is not None:
            try:
                self._per_cli.release()
            except ValueError:
                pass
            self._holding_per_cli = False
        if self._holding_global:
            try:
                _GLOBAL_SEMAPHORE.release()
            except ValueError:
                pass
            self._holding_global = False


class _CliShimChatCompletions:
    def __init__(self, client: "CliShimClient"):
        self._client = client

    def create(self, **kwargs: Any) -> Any:
        # When hermes asks for streaming, return a synthetic single-chunk
        # iterator that wraps the whole response. The CLIs we shell out to
        # don't support per-token streaming, so we collect the full response
        # and emit it as one delta chunk + a final chunk with finish_reason.
        stream_requested = bool(kwargs.pop("stream", False))
        response = self._client._create_chat_completion(**kwargs)
        if not stream_requested:
            return response
        return _wrap_response_as_stream(response)


def _wrap_response_as_stream(response: Any):
    """Yield OpenAI-shaped streaming chunks from a fully-formed response.

    Hermes' stream consumer iterates chunks, each with .choices[].delta.
    We emit:
      1) one chunk with the full content as a delta
      2) one chunk per tool_call (if any) with the full tool args
      3) one final chunk with finish_reason set and usage attached
    """
    choice = response.choices[0]
    msg = choice.message
    finish_reason = choice.finish_reason or "stop"
    model = response.model

    # Chunk 1: content delta
    content = getattr(msg, "content", "") or ""
    delta1 = SimpleNamespace(
        role="assistant",
        content=content,
        tool_calls=None,
        reasoning=getattr(msg, "reasoning", None),
        reasoning_content=getattr(msg, "reasoning_content", None),
    )
    yield SimpleNamespace(
        id="cli-shim-stream-1",
        model=model,
        choices=[SimpleNamespace(index=0, delta=delta1, finish_reason=None)],
        usage=None,
    )

    # Chunk 2+: tool_calls (each as its own delta following OpenAI shape)
    tool_calls = getattr(msg, "tool_calls", None) or []
    for i, tc in enumerate(tool_calls):
        delta_tc = SimpleNamespace(
            role=None,
            content=None,
            tool_calls=[
                SimpleNamespace(
                    index=i,
                    id=getattr(tc, "id", f"call_{i}"),
                    type="function",
                    function=SimpleNamespace(
                        name=getattr(tc.function, "name", ""),
                        arguments=getattr(tc.function, "arguments", "{}"),
                    ),
                )
            ],
        )
        yield SimpleNamespace(
            id=f"cli-shim-stream-tc-{i}",
            model=model,
            choices=[SimpleNamespace(index=0, delta=delta_tc, finish_reason=None)],
            usage=None,
        )

    # Final chunk: finish_reason + usage
    delta_final = SimpleNamespace(role=None, content=None, tool_calls=None)
    yield SimpleNamespace(
        id="cli-shim-stream-final",
        model=model,
        choices=[SimpleNamespace(index=0, delta=delta_final, finish_reason=finish_reason)],
        usage=response.usage,
    )


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
            with _ConcurrencyGate(dispatch["command"]):
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
        elif dispatch["mode"] == "codex_exec":
            with _ConcurrencyGate(dispatch["command"]):
                response_text, reasoning_text = self._run_codex_exec(
                    prompt_text,
                    command=self._command_override or dispatch["command"],
                    args=self._args_override or dispatch["args"],
                    timeout_seconds=timeout_seconds,
                )
        else:
            with _ConcurrencyGate(dispatch["command"]):
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

    def _run_codex_exec(
        self,
        prompt_text: str,
        *,
        command: str,
        args: list[str],
        timeout_seconds: float,
    ) -> tuple[str, str]:
        """codex exec with --output-last-message <tmpfile> for clean output.

        Codex CLI's stdout in `exec` mode is a noisy scaffold ('user\\n...\\ncodex\\n...').
        Using --output-last-message writes ONLY the final assistant message to
        a tempfile, which we then read back.
        """
        import tempfile

        resolved = shutil.which(command) or command
        with tempfile.NamedTemporaryFile(
            mode="r", suffix=".txt", delete=False, prefix="codex-cli-shim-"
        ) as tmp:
            tmp_path = tmp.name

        full_args = list(args) + ["--output-last-message", tmp_path]

        try:
            proc = subprocess.Popen(
                [resolved] + full_args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=self._cwd,
                env=_build_subprocess_env(),
            )
        except FileNotFoundError as exc:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
            raise RuntimeError(
                f"Could not start CLI '{command}'. Install it or check PATH."
            ) from exc

        with self._active_process_lock:
            self._active_process = proc
        self.is_closed = False

        try:
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

            # Read the clean last-message output
            try:
                with open(tmp_path, "r", encoding="utf-8") as f:
                    response_text = f.read().strip()
            except FileNotFoundError:
                # Codex didn't write the file (no assistant turn happened);
                # fall back to scrubbing stdout.
                response_text = self._scrub_codex_stdout(stdout or "")

            return response_text, ""
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    @staticmethod
    def _scrub_codex_stdout(raw: str) -> str:
        """Last-resort fallback: extract the assistant turn from noisy codex stdout."""
        lines = raw.splitlines()
        # codex output format: "...\ncodex\n<message>\ntokens used\n<n>"
        try:
            idx = lines.index("codex")
            tail = lines[idx + 1 :]
            if "tokens used" in tail:
                tail = tail[: tail.index("tokens used")]
            return "\n".join(tail).strip()
        except ValueError:
            return raw.strip()
