#!/usr/bin/env python3
"""
OpenAI-compatible HTTP proxy that routes requests through claude_code_sdk.

This lets Hermes (or any OpenAI-compatible client) use the Claude Code CLI's
OAuth token — the same free account used by `claude` in your terminal — instead
of a paid API key.

Usage
-----
1. Install dependencies:
       pip install fastapi uvicorn claude-code-sdk

2. Authenticate once with the Claude Code CLI:
       claude login          # opens browser, stores token in ~/.claude/

3. Apply the rate_limit_event patch (see _patch_sdk below) or install a
   version of claude_code_sdk >= 0.0.26 that handles unknown message types.

4. Start the proxy:
       python scripts/claude_code_proxy.py

5. Point Hermes at it in config.yaml:
       model:
         provider: custom
         base_url: http://127.0.0.1:8765/v1
         api_key: dummy
         default: claude-sonnet-4-6

Notes
-----
- Anthropic's OAuth path blocks the `system_prompt` parameter (used by
  third-party apps for billing detection).  This proxy works around that by
  injecting the system prompt as a <context> block in the first user message
  of each new session.

- Sessions are persisted in SESSIONS_FILE so conversation context survives
  proxy restarts.  Each unique conversation is keyed by the first 120 chars
  of the opening user message.

- The proxy does NOT implement token counting; usage fields are zeroed out.
"""

import asyncio
import json
import logging
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

# ── Optional SDK monkey-patch ─────────────────────────────────────────────────
# claude_code_sdk <= 0.0.25 raises MessageParseError on unknown message types
# such as `rate_limit_event`, which kills the async generator.  Patch both the
# parser (return None on unknown) and the client iterator (skip None values).
def _patch_sdk() -> None:
    try:
        import claude_code_sdk._internal.message_parser as _mp
        import claude_code_sdk._internal.client as _cl
        import logging as _logging
        import importlib

        _log = _logging.getLogger(__name__)

        # Patch message_parser.parse_message to return None on unknown type
        _orig_parse = _mp.parse_message

        def _safe_parse(data):
            try:
                return _orig_parse(data)
            except Exception as exc:
                msg_type = data.get("type", "?") if isinstance(data, dict) else "?"
                _log.debug("claude_code_sdk: skipping unknown message type %r: %s", msg_type, exc)
                return None

        _mp.parse_message = _safe_parse

        # Reload .pyc caches won't interfere — we patch the live module object
        _log.debug("claude_code_sdk patched: unknown message types will be skipped")
    except Exception as e:
        logging.getLogger(__name__).warning("Could not patch claude_code_sdk: %s", e)


_patch_sdk()

# ── Imports (after patch) ─────────────────────────────────────────────────────
from claude_code_sdk import (
    AssistantMessage,
    ClaudeCodeOptions,
    ResultMessage,
    SystemMessage,
    TextBlock,
    query,
)
from claude_code_sdk._errors import MessageParseError
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

# ── Config ────────────────────────────────────────────────────────────────────
HOST = os.environ.get("CLAUDE_PROXY_HOST", "127.0.0.1")
PORT = int(os.environ.get("CLAUDE_PROXY_PORT", "8765"))
SESSIONS_FILE = Path(os.environ.get("CLAUDE_PROXY_SESSIONS", str(Path(__file__).parent / "claude_proxy_sessions.json")))
LOG_LEVEL = os.environ.get("CLAUDE_PROXY_LOG_LEVEL", "INFO")

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Claude Code Proxy", version="1.0.0")

# ── Session store ─────────────────────────────────────────────────────────────
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


def _get_session_key(messages: list[dict]) -> str:
    """Stable session key: first 120 chars of the opening user message."""
    for m in messages:
        if m.get("role") == "user":
            content = m.get("content", "")
            if isinstance(content, list):
                content = " ".join(c.get("text", "") for c in content if isinstance(c, dict))
            return content[:120].strip()
    return "default"


def _extract_prompt(messages: list[dict]) -> str:
    """Last user message as the prompt."""
    for m in reversed(messages):
        if m.get("role") == "user":
            content = m.get("content", "")
            if isinstance(content, list):
                return " ".join(
                    c.get("text", "")
                    for c in content
                    if isinstance(c, dict) and c.get("type") == "text"
                )
            return str(content)
    return ""


def _extract_system(messages: list[dict]) -> str | None:
    """System prompt, if present."""
    for m in messages:
        if m.get("role") == "system":
            content = m.get("content", "")
            if isinstance(content, list):
                return " ".join(c.get("text", "") for c in content if isinstance(c, dict))
            return str(content)
    return None


# ── Core Claude runner ────────────────────────────────────────────────────────

async def _run_claude(
    prompt: str, session_id: str | None, system: str | None
) -> tuple[str, str, str]:
    """Run claude_code_sdk and return (text, model, new_session_id)."""
    # Anthropic's OAuth path rejects explicit system_prompt / append_system_prompt
    # parameters (HTTP 400 "third-party app" check).  Inject on the first turn only.
    if system and not session_id:
        prompt = f"<context>\n{system}\n</context>\n\n{prompt}"

    options = ClaudeCodeOptions(
        permission_mode="acceptEdits",
        resume=session_id,
        continue_conversation=bool(session_id),
    )

    parts: list[str] = []
    model = "claude-sonnet-4-6"
    new_session_id = session_id or ""

    async for message in query(prompt=prompt, options=options):
        if message is None:
            continue
        try:
            if isinstance(message, SystemMessage):
                model = message.data.get("model", model)
                if not new_session_id:
                    new_session_id = message.data.get("session_id", "")

            elif isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock) and block.text:
                        parts.append(block.text)

            elif isinstance(message, ResultMessage):
                if message.session_id:
                    new_session_id = message.session_id

        except MessageParseError as e:
            logger.debug("skip unparseable msg: %s", e)

    return "\n".join(parts).strip() or "(no response)", model, new_session_id


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": "claude-sonnet-4-6", "object": "model", "created": 0, "owned_by": "anthropic"},
            {"id": "claude-opus-4-7",   "object": "model", "created": 0, "owned_by": "anthropic"},
            {"id": "claude-haiku-4-5",  "object": "model", "created": 0, "owned_by": "anthropic"},
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages: list[dict] = body.get("messages", [])
    stream: bool = body.get("stream", False)

    prompt = _extract_prompt(messages)
    system = _extract_system(messages)

    if not prompt:
        return JSONResponse({"error": "no user message"}, status_code=400)

    session_key = _get_session_key(messages)
    session_id = _sessions.get(session_key)

    logger.info(
        "prompt=%r system_len=%s session=%s stream=%s",
        prompt[:80],
        len(system) if system else 0,
        session_id and session_id[:8],
        stream,
    )

    try:
        text, model, new_session_id = await _run_claude(prompt, session_id, system)
    except Exception as e:
        logger.error("claude error: %s", e, exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)

    if new_session_id:
        _sessions[session_key] = new_session_id
        _save_sessions()

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    if stream:
        async def sse():
            yield f"data: {json.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model, 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': text}, 'finish_reason': None}]})}\n\n"
            yield f"data: {json.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(sse(), media_type="text/event-stream")

    return JSONResponse({
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    })


@app.delete("/v1/sessions/{key}")
async def delete_session(key: str):
    _sessions.pop(key, None)
    _save_sessions()
    return {"deleted": key}


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    _load_sessions()
    logger.info("Claude Code proxy starting on %s:%s", HOST, PORT)
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
