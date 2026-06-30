#!/usr/bin/env python3
"""Local OpenAI-compatible bridge for Google Antigravity CLI (``agy``).

The bridge intentionally exposes only the explicitly approved AGY model and
binds to loopback by default.  Each request starts a fresh, sandboxed AGY
print-mode process and converts its final response into OpenAI chat-completion
format so Hermes can use it as a normal model provider.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse


MODEL_MAP = {
    "gemini-3.5-flash-high": "Gemini 3.5 Flash (High)",
}
DEFAULT_MODEL = "gemini-3.5-flash-high"
DEFAULT_TIMEOUT_SECONDS = 600
MAX_INLINE_PROMPT_BYTES = 100_000
_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
    re.DOTALL,
)
_FINAL_RESPONSE_RE = re.compile(
    r"<hermes_final>\s*(.*?)\s*</hermes_final>",
    re.DOTALL,
)

app = FastAPI(title="Hermes AGY Bridge", version="1.0.0")


def _render_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts)
    if content is None:
        return ""
    return json.dumps(content, ensure_ascii=False)


def build_agy_prompt(payload: dict[str, Any]) -> str:
    sections = [
        "You are the inference backend for Hermes Agent.",
        (
            "Do not invoke Antigravity tools or modify files. Hermes owns tool "
            "execution. Answer the conversation directly unless a Hermes tool "
            "is required."
        ),
        (
            "For a normal final response, output the answer inside exactly one "
            "<hermes_final>...</hermes_final> block. Do not put progress notes "
            "inside that block."
        ),
    ]

    tools = payload.get("tools")
    if isinstance(tools, list) and tools:
        sections.append(
            "When a Hermes tool is required, output only one or more blocks in "
            "this exact format, with OpenAI function-call JSON and arguments "
            "encoded as a JSON string:\n"
            '<tool_call>{"id":"call_1","type":"function","function":'
            '{"name":"tool_name","arguments":"{\\"key\\":\\"value\\"}"}}</tool_call>\n'
            "Available Hermes tools:\n"
            + json.dumps(tools, ensure_ascii=False)
        )

    transcript: list[str] = []
    for message in payload.get("messages") or []:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "context").strip().title()
        content = _render_content(message.get("content")).strip()
        if content:
            transcript.append(f"{role}:\n{content}")
    if transcript:
        sections.append("Conversation:\n\n" + "\n\n".join(transcript))

    sections.append("Continue from the latest user request.")
    return "\n\n".join(sections)


def extract_tool_calls(text: str) -> tuple[list[dict[str, Any]], str]:
    tool_calls: list[dict[str, Any]] = []
    spans: list[tuple[int, int]] = []
    for match in _TOOL_CALL_RE.finditer(text):
        try:
            item = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        function = item.get("function") if isinstance(item, dict) else None
        if not isinstance(function, dict) or not function.get("name"):
            continue
        arguments = function.get("arguments", "{}")
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, ensure_ascii=False)
        tool_calls.append(
            {
                "id": str(item.get("id") or f"agy_call_{len(tool_calls) + 1}"),
                "type": "function",
                "function": {
                    "name": str(function["name"]),
                    "arguments": arguments,
                },
            }
        )
        spans.append(match.span())

    if not spans:
        return [], text.strip()

    cleaned_parts: list[str] = []
    cursor = 0
    for start, end in spans:
        cleaned_parts.append(text[cursor:start])
        cursor = end
    cleaned_parts.append(text[cursor:])
    return tool_calls, "\n".join(part.strip() for part in cleaned_parts if part.strip())


def extract_final_response(text: str) -> str:
    matches = _FINAL_RESPONSE_RE.findall(text)
    return matches[-1].strip() if matches else text.strip()


@contextlib.contextmanager
def agy_prompt_argument(prompt: str, workdir: Path) -> Iterator[str]:
    """Yield an AGY prompt argument without exceeding Linux argv limits."""
    if len(prompt.encode("utf-8")) <= MAX_INLINE_PROMPT_BYTES:
        yield prompt
        return

    prompt_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix="hermes-request-",
            suffix=".txt",
            dir=workdir,
            delete=False,
        ) as handle:
            handle.write(prompt)
            prompt_path = Path(handle.name)
        prompt_path.chmod(0o600)
        yield (
            f"Read the complete Hermes backend request from @{prompt_path}. "
            "Follow it exactly. Do not use tools except reading that request file."
        )
    finally:
        if prompt_path is not None:
            prompt_path.unlink(missing_ok=True)


def run_agy(model: str, prompt: str) -> str:
    display_model = MODEL_MAP.get(model)
    if not display_model:
        raise HTTPException(status_code=404, detail=f"Unsupported model: {model}")

    command = os.getenv("AGY_BINARY", "").strip() or shutil.which("agy")
    if not command:
        raise HTTPException(status_code=503, detail="AGY CLI is not installed")

    timeout = int(os.getenv("AGY_BRIDGE_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS))
    workdir = Path(
        os.getenv("AGY_BRIDGE_WORKDIR", "~/.hermes/agy-bridge-workdir")
    ).expanduser()
    workdir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["AGY_CLI_DISABLE_AUTO_UPDATE"] = "true"
    try:
        with agy_prompt_argument(prompt, workdir) as prompt_argument:
            argv = [
                command,
                "--sandbox",
                "--model",
                display_model,
                "--print-timeout",
                f"{timeout}s",
                "--print",
                prompt_argument,
            ]
            result = subprocess.run(
                argv,
                cwd=workdir,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout + 15,
                check=False,
            )
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(status_code=504, detail="AGY request timed out") from exc

    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "AGY request failed").strip()
        raise HTTPException(status_code=502, detail=detail[-1000:])
    response = extract_final_response(result.stdout)
    if not response:
        raise HTTPException(status_code=502, detail="AGY returned an empty response")
    return response


def build_completion(payload: dict[str, Any], response_text: str) -> dict[str, Any]:
    model = str(payload.get("model") or DEFAULT_MODEL)
    tool_calls, content = extract_tool_calls(response_text)
    message: dict[str, Any] = {
        "role": "assistant",
        "content": content or None,
    }
    finish_reason = "stop"
    if tool_calls:
        message["tool_calls"] = tool_calls
        finish_reason = "tool_calls"

    return {
        "id": f"chatcmpl-agy-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


def stream_completion(completion: dict[str, Any]) -> Iterator[str]:
    choice = completion["choices"][0]
    message = choice["message"]
    base = {
        "id": completion["id"],
        "object": "chat.completion.chunk",
        "created": completion["created"],
        "model": completion["model"],
    }

    delta: dict[str, Any] = {"role": "assistant"}
    if message.get("content"):
        delta["content"] = message["content"]
    if message.get("tool_calls"):
        delta["tool_calls"] = [
            {
                "index": index,
                **tool_call,
            }
            for index, tool_call in enumerate(message["tool_calls"])
        ]
    first = {
        **base,
        "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
    }
    final = {
        **base,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": choice["finish_reason"],
            }
        ],
    }
    yield f"data: {json.dumps(first, ensure_ascii=False)}\n\n"
    yield f"data: {json.dumps(final, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "agy_available": bool(os.getenv("AGY_BINARY", "").strip() or shutil.which("agy")),
        "models": list(MODEL_MAP),
    }


@app.get("/v1/models")
def models() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {
                "id": model,
                "object": "model",
                "owned_by": "google-antigravity",
            }
            for model in MODEL_MAP
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Any:
    payload = await request.json()
    model = str(payload.get("model") or DEFAULT_MODEL)
    prompt = build_agy_prompt(payload)
    response_text = run_agy(model, prompt)
    completion = build_completion(payload, response_text)
    if payload.get("stream"):
        return StreamingResponse(
            stream_completion(completion),
            media_type="text/event-stream",
        )
    return completion


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9121)
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
