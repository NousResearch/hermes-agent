"""Python adapter for Cursor's TypeScript SDK.

Cursor currently publishes the official SDK as the Node package ``@cursor/sdk``.
Hermes is Python-first, so this module shells out to a small Node bridge and
normalizes the result to the OpenAI-like response shape expected by the agent
loop.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterable


class CursorSdkError(RuntimeError):
    """Raised when the Cursor SDK bridge reports an error."""


def _bridge_path() -> Path:
    return Path(__file__).with_name("cursor_sdk_bridge.mjs")


def _text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif item.get("type") == "image_url":
                    parts.append("[image omitted]")
        return "\n".join(part for part in parts if part)
    if isinstance(content, dict):
        for key in ("text", "content"):
            value = content.get(key)
            if isinstance(value, str):
                return value
    return "" if content is None else str(content)


def _messages_to_cursor_prompt(messages: Iterable[dict[str, Any]]) -> str:
    """Flatten Hermes/OpenAI-style history into one Cursor SDK prompt."""
    rendered: list[str] = [
        "You are running as the Cursor SDK provider inside Hermes Agent.",
        "Use the conversation transcript below as context and answer the latest user request.",
        "Do not mention this transcript wrapper. Do not fabricate tool calls.",
        "",
        "Conversation:",
    ]

    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or "user").strip() or "user"
        content = _text_from_content(msg.get("content")).strip()
        if not content:
            continue
        if role == "tool":
            tool_name = str(msg.get("tool_name") or msg.get("name") or "tool")
            rendered.append(f"Tool result ({tool_name}):\n{content}")
        elif role == "assistant":
            rendered.append(f"Assistant:\n{content}")
        elif role in {"system", "developer"}:
            rendered.append(f"System:\n{content}")
        else:
            rendered.append(f"User:\n{content}")

    rendered.extend(
        [
            "",
            "Respond now with the assistant's next message only.",
        ]
    )
    return "\n\n".join(rendered)


def _response_namespace(
    *,
    content: str,
    model: str,
    finish_reason: str = "stop",
) -> SimpleNamespace:
    message = SimpleNamespace(role="assistant", content=content, tool_calls=None)
    choice = SimpleNamespace(index=0, message=message, finish_reason=finish_reason)
    usage = SimpleNamespace(prompt_tokens=0, completion_tokens=0, total_tokens=0)
    return SimpleNamespace(
        id="cursor-sdk-response",
        object="chat.completion",
        model=model,
        choices=[choice],
        usage=usage,
    )


def _run_bridge(
    payload: dict[str, Any],
    *,
    timeout: float | None = None,
    on_delta: Callable[[str], None] | None = None,
    on_reasoning_delta: Callable[[str], None] | None = None,
    on_status: Callable[[str], None] | None = None,
    interrupt_check: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    bridge = _bridge_path()
    if not bridge.exists():
        raise CursorSdkError(f"Cursor SDK bridge not found: {bridge}")

    cmd = [os.getenv("CURSOR_SDK_NODE", "node"), str(bridge)]
    env = os.environ.copy()
    hermes_home = env.get("HERMES_HOME")
    if not hermes_home:
        try:
            from hermes_constants import get_hermes_home

            hermes_home = str(get_hermes_home())
        except Exception:
            hermes_home = str(Path.home() / ".hermes")
        env["HERMES_HOME"] = hermes_home

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        env=env,
    )
    assert proc.stdin is not None
    assert proc.stdout is not None

    proc.stdin.write(json.dumps(payload))
    proc.stdin.close()

    lines: "queue.Queue[str | None]" = queue.Queue()

    def _read_stdout() -> None:
        try:
            for stdout_line in proc.stdout:
                lines.put(stdout_line)
        finally:
            lines.put(None)

    reader = threading.Thread(target=_read_stdout, daemon=True)
    reader.start()

    started = time.monotonic()
    final: dict[str, Any] | None = None
    errors: list[str] = []

    while True:
        if interrupt_check and interrupt_check():
            proc.terminate()
            raise InterruptedError("Agent interrupted during Cursor SDK call")
        if timeout is not None and timeout > 0 and time.monotonic() - started > timeout:
            proc.terminate()
            raise TimeoutError(f"Cursor SDK call timed out after {int(timeout)}s")

        try:
            line = lines.get(timeout=0.1)
        except queue.Empty:
            if proc.poll() is not None:
                try:
                    line = lines.get_nowait()
                except queue.Empty:
                    break
            else:
                continue
        if line is None:
            continue

        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        etype = event.get("type")
        if etype == "delta":
            text = event.get("text")
            if isinstance(text, str) and text and on_delta:
                on_delta(text)
        elif etype == "reasoning_delta":
            text = event.get("text")
            if isinstance(text, str) and text and on_reasoning_delta:
                on_reasoning_delta(text)
        elif etype == "status":
            message = event.get("message")
            if isinstance(message, str) and message and on_status:
                on_status(message)
        elif etype == "final":
            final = event
        elif etype == "error":
            message = str(event.get("message") or "Cursor SDK bridge failed")
            code = event.get("code")
            errors.append(f"{code}: {message}" if code else message)

    stderr = ""
    if proc.stderr is not None:
        stderr = proc.stderr.read().strip()
    returncode = proc.wait(timeout=2)
    if returncode != 0:
        detail = errors[-1] if errors else stderr or f"exit code {returncode}"
        raise CursorSdkError(detail)
    if errors and final is None:
        raise CursorSdkError(errors[-1])
    if final is None:
        raise CursorSdkError(stderr or "Cursor SDK bridge exited without a final response")
    return final


def create_cursor_sdk_response(
    *,
    api_key: str,
    model: str,
    messages: list[dict[str, Any]],
    cwd: str | None = None,
    runtime: str | None = None,
    timeout: float | None = None,
    stream: bool = False,
    on_delta: Callable[[str], None] | None = None,
    on_reasoning_delta: Callable[[str], None] | None = None,
    on_status: Callable[[str], None] | None = None,
    interrupt_check: Callable[[], bool] | None = None,
) -> SimpleNamespace:
    prompt = _messages_to_cursor_prompt(messages)
    payload = {
        "operation": "prompt",
        "apiKey": api_key,
        "model": model,
        "prompt": prompt,
        "cwd": cwd or os.getcwd(),
        "runtime": runtime or os.getenv("CURSOR_SDK_RUNTIME", "local"),
        "stream": bool(stream),
    }
    final = _run_bridge(
        payload,
        timeout=timeout,
        on_delta=on_delta,
        on_reasoning_delta=on_reasoning_delta,
        on_status=on_status,
        interrupt_check=interrupt_check,
    )
    content = str(final.get("result") or final.get("text") or "")
    status = str(final.get("status") or "finished").lower()
    finish_reason = "stop" if status == "finished" else status
    return _response_namespace(content=content, model=model, finish_reason=finish_reason)


def list_cursor_models(*, api_key: str, timeout: float = 8.0) -> list[str]:
    payload = {"operation": "models", "apiKey": api_key}
    final = _run_bridge(payload, timeout=timeout)
    models = final.get("models")
    if not isinstance(models, list):
        return []
    ids: list[str] = []
    for item in models:
        if isinstance(item, str):
            ids.append(item)
        elif isinstance(item, dict) and isinstance(item.get("id"), str):
            ids.append(item["id"])
            for alias in item.get("aliases") or []:
                if isinstance(alias, str):
                    ids.append(alias)
    return sorted(dict.fromkeys(ids))


def main(argv: list[str] | None = None) -> int:
    argv = argv or sys.argv[1:]
    if argv and argv[0] == "--prompt-from-stdin":
        data = json.load(sys.stdin)
        response = create_cursor_sdk_response(**data)
        print(response.choices[0].message.content)
        return 0
    raise SystemExit("usage: python -m agent.cursor_sdk_adapter --prompt-from-stdin")


if __name__ == "__main__":
    raise SystemExit(main())
