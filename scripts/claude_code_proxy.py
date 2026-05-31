#!/usr/bin/env python3
"""
OpenAI-compatible HTTP proxy that routes requests through claude_code_sdk.

This lets Hermes (or any OpenAI-compatible client) use the Claude Code CLI's
OAuth token instead of a paid API key.
"""

import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import AsyncGenerator

def _patch_sdk() -> None:
    try:
        import claude_code_sdk._internal.message_parser as _mp
        _orig = _mp.parse_message

        def _safe(data):
            try:
                return _orig(data)
            except Exception as exc:
                t = data.get("type", "?") if isinstance(data, dict) else "?"
                logging.getLogger(__name__).debug("sdk: skip unknown msg type %r: %s", t, exc)
                return None

        _mp.parse_message = _safe
    except Exception as e:
        logging.getLogger(__name__).warning("Could not patch claude_code_sdk: %s", e)

_patch_sdk()

from claude_code_sdk import (
    AssistantMessage, ClaudeCodeOptions, ResultMessage,
    SystemMessage, TextBlock, ThinkingBlock, query,
)
from claude_code_sdk._errors import MessageParseError
from claude_code_sdk.types import StreamEvent
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

HOST          = os.environ.get("CLAUDE_PROXY_HOST", "127.0.0.1")
PORT          = int(os.environ.get("CLAUDE_PROXY_PORT", "8765"))
SESSIONS_FILE = Path(os.environ.get(
    "CLAUDE_PROXY_SESSIONS",
    str(Path(__file__).parent / "claude_proxy_sessions.json"),
))
LOG_LEVEL = os.environ.get("CLAUDE_PROXY_LOG_LEVEL", "INFO")

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Claude Code Proxy", version="2.0.0")

_sessions: dict[str, str] = {}

def _load_sessions() -> None:
    global _sessions
    if SESSIONS_FILE.exists():
        try:
            _sessions = json.loads(SESSIONS_FILE.read_text())
        except Exception:
            _sessions = {}

def _save_sessions() -> None:
    SESSIONS_FILE.write_text(json.dumps(_sessions, indent=2))

def _session_key(messages: list[dict]) -> str:
    for m in messages:
        if m.get("role") == "user":
            c = m.get("content", "")
            if isinstance(c, list):
                c = " ".join(x.get("text", "") for x in c if isinstance(x, dict))
            return str(c)[:120].strip()
    return "default"

def _extract_prompt(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            c = m.get("content", "")
            if isinstance(c, list):
                return " ".join(
                    x.get("text", "") for x in c
                    if isinstance(x, dict) and x.get("type") == "text"
                )
            return str(c)
    return ""

def _extract_system(messages: list[dict]) -> str | None:
    for m in messages:
        if m.get("role") == "system":
            c = m.get("content", "")
            if isinstance(c, list):
                return " ".join(x.get("text", "") for x in c if isinstance(x, dict))
            return str(c)
    return None


async def _stream_claude(
    prompt: str,
    session_id: str | None,
    system: str | None,
) -> AsyncGenerator[tuple[str, str, str], None]:
    """
    Async generator yielding (chunk_text, model, new_session_id).
    Uses include_partial_messages=True for real-time streaming.
    Thinking blocks are streamed inside <think>...</think> tags.
    """
    if system and not session_id:
        prompt = f"<context>\n{system}\n</context>\n\n{prompt}"

    opts = ClaudeCodeOptions(
        permission_mode="acceptEdits",
        resume=session_id,
        continue_conversation=bool(session_id),
        include_partial_messages=True,
    )

    model = "claude-sonnet-4-6"
    sid = session_id or ""
    in_think = False

    async for msg in query(prompt=prompt, options=opts):
        if msg is None:
            continue
        try:
            if isinstance(msg, SystemMessage) and msg.subtype == "init":
                model = msg.data.get("model", model)
                if not sid:
                    sid = msg.data.get("session_id", "")

            elif isinstance(msg, StreamEvent):
                ev = msg.event
                etype = ev.get("type")

                if etype == "content_block_start":
                    btype = ev.get("content_block", {}).get("type")
                    if btype == "thinking" and not in_think:
                        in_think = True
                        yield ("<think>\n", model, sid)
                    elif btype == "text" and in_think:
                        in_think = False
                        yield ("\n</think>\n\n", model, sid)

                elif etype == "content_block_delta":
                    delta = ev.get("delta", {})
                    dtype = delta.get("type")
                    if dtype == "thinking_delta":
                        yield (delta.get("thinking", ""), model, sid)
                    elif dtype == "text_delta":
                        yield (delta.get("text", ""), model, sid)

                elif etype == "content_block_stop" and in_think:
                    in_think = False
                    yield ("\n</think>\n\n", model, sid)

            elif isinstance(msg, AssistantMessage):
                if in_think:
                    in_think = False
                    yield ("\n</think>\n\n", model, sid)
                for block in msg.content:
                    if isinstance(block, ThinkingBlock) and block.thinking:
                        yield (f"<think>\n{block.thinking}\n</think>\n\n", model, sid)
                    elif isinstance(block, TextBlock) and block.text:
                        yield (block.text, model, sid)

            elif isinstance(msg, ResultMessage):
                if msg.session_id:
                    sid = msg.session_id

        except MessageParseError as e:
            logger.debug("skip unparseable msg: %s", e)


async def _run_claude(
    prompt: str, session_id: str | None, system: str | None,
) -> tuple[str, str, str]:
    parts: list[str] = []
    model = "claude-sonnet-4-6"
    sid = session_id or ""
    async for chunk, m, s in _stream_claude(prompt, session_id, system):
        parts.append(chunk)
        model, sid = m, s
    return "".join(parts).strip() or "(no response)", model, sid


@app.get("/v1/models")
async def list_models():
    return {"object": "list", "data": [
        {"id": "claude-sonnet-4-6", "object": "model", "created": 0, "owned_by": "anthropic"},
        {"id": "claude-opus-4-7",   "object": "model", "created": 0, "owned_by": "anthropic"},
        {"id": "claude-haiku-4-5",  "object": "model", "created": 0, "owned_by": "anthropic"},
    ]}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages: list[dict] = body.get("messages", [])
    do_stream: bool = body.get("stream", False)

    prompt = _extract_prompt(messages)
    system = _extract_system(messages)

    if not prompt:
        return JSONResponse({"error": "no user message"}, status_code=400)

    key = _session_key(messages)
    sid = _sessions.get(key)

    logger.info("prompt=%r sys_len=%s session=%s stream=%s",
                prompt[:80], len(system) if system else 0, sid and sid[:8], do_stream)

    if do_stream:
        cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        now = int(time.time())

        async def sse():
            state = {"model": "claude-sonnet-4-6", "sid": sid or ""}
            try:
                async for chunk, m, s in _stream_claude(prompt, sid, system):
                    if not chunk:
                        continue
                    state["model"], state["sid"] = m, s
                    pkt = {"id": cid, "object": "chat.completion.chunk", "created": now,
                           "model": m, "choices": [{"index": 0,
                           "delta": {"role": "assistant", "content": chunk},
                           "finish_reason": None}]}
                    yield f"data: {json.dumps(pkt)}\n\n"
            except Exception as e:
                logger.error("stream error: %s", e, exc_info=True)
                err_pkt = {"id": cid, "object": "chat.completion.chunk", "created": now,
                           "model": state["model"], "choices": [{"index": 0,
                           "delta": {"content": f"\n\n[error: {e}]"}, "finish_reason": None}]}
                yield f"data: {json.dumps(err_pkt)}\n\n"
            finally:
                if state["sid"]:
                    _sessions[key] = state["sid"]
                    _save_sessions()
            stop = {"id": cid, "object": "chat.completion.chunk", "created": now,
                    "model": state["model"], "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
            yield f"data: {json.dumps(stop)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(sse(), media_type="text/event-stream")

    try:
        text, model, new_sid = await _run_claude(prompt, sid, system)
    except Exception as e:
        logger.error("claude error: %s", e, exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)

    if new_sid:
        _sessions[key] = new_sid
        _save_sessions()

    cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    return JSONResponse({
        "id": cid, "object": "chat.completion", "created": int(time.time()), "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    })


@app.delete("/v1/sessions/{key:path}")
async def delete_session(key: str):
    _sessions.pop(key, None)
    _save_sessions()
    return {"deleted": key}


if __name__ == "__main__":
    import uvicorn
    _load_sessions()
    logger.info("Claude Code proxy starting on %s:%s", HOST, PORT)
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
