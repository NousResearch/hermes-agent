"""
CLI Bridge API — OpenAI-compatible endpoint for CLI-backed inference.

Endpoints:
- POST /v1/chat/completions - OpenAI-compatible chat (routes through CLI bridges)
- GET /v1/cli/status - Status of all CLI bridges
- GET /v1/cli/rate-stats - Rate limit stats for all CLIs

Model names:
- "auto" — router decides based on prompt analysis
- "claude-code" — force Claude Code CLI
- "gemini-cli" — force Gemini CLI

Supports streaming (SSE) when stream: true.
"""

import json
import time
import uuid
from typing import Optional

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse
from loguru import logger

router = APIRouter()


# Model name → CLI preference mapping
_MODEL_TO_CLI = {
    "auto": None,
    "claude-code": "claude_code",
    "gemini-cli": "gemini_cli",
    "gemini": "gemini_cli",
    "claude": "claude_code",
}


def _get_router():
    """Lazy import to avoid circular imports."""
    from Cosmos.integration.cli_bridge.capability_router import get_cli_router
    return get_cli_router


@router.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """
    OpenAI-compatible chat completions endpoint backed by CLI bridges.

    Accepts standard OpenAI request format:
    {
        "model": "auto" | "claude-code" | "gemini-cli",
        "messages": [{"role": "user", "content": "..."}],
        "stream": false,
        "max_tokens": 1000,
        "temperature": 0.7
    }

    Returns standard OpenAI response format.
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "Invalid JSON body", "type": "invalid_request_error"}},
        )

    model = body.get("model", "auto")
    messages = body.get("messages", [])
    stream = body.get("stream", False)
    max_tokens = body.get("max_tokens")
    temperature = body.get("temperature")

    if not messages:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "messages is required", "type": "invalid_request_error"}},
        )

    # Extract system prompt and user prompt from messages
    system_prompt = None
    user_prompt = ""

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "system":
            system_prompt = content
        elif role == "user":
            user_prompt = content
        elif role == "assistant":
            # Include assistant messages for context
            user_prompt += f"\n\nPrevious response: {content}"

    if not user_prompt:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "No user message found", "type": "invalid_request_error"}},
        )

    # Resolve preferred CLI from model name
    preferred_cli = _MODEL_TO_CLI.get(model)

    # Get the router
    get_router_fn = _get_router()
    cli_router = await get_router_fn()

    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    if stream:
        return StreamingResponse(
            _stream_response(
                cli_router, user_prompt, system_prompt, max_tokens,
                preferred_cli, model, request_id,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # Non-streaming response
    start = time.time()

    response = await cli_router.query_with_fallback(
        prompt=user_prompt,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        preferred_cli=preferred_cli,
    )

    latency = time.time() - start

    if not response.success:
        return JSONResponse(
            status_code=502,
            content={
                "error": {
                    "message": response.error,
                    "type": "cli_bridge_error",
                    "cli_name": response.cli_name,
                }
            },
        )

    # Return OpenAI-compatible response
    return JSONResponse(content={
        "id": request_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": response.model or model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response.content,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,  # CLI doesn't always report tokens
            "completion_tokens": response.tokens_used,
            "total_tokens": response.tokens_used,
        },
        "cli_bridge": {
            "cli_name": response.cli_name,
            "cost_usd": response.cost_usd,
            "latency_ms": response.latency_ms,
            "session_id": response.session_id,
        },
    })


async def _stream_response(
    cli_router, prompt, system_prompt, max_tokens,
    preferred_cli, model, request_id,
):
    """Generate SSE stream in OpenAI format."""
    created = int(time.time())

    async for delta in cli_router.query_streaming_with_fallback(
        prompt=prompt,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        preferred_cli=preferred_cli,
    ):
        chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": delta},
                    "finish_reason": None,
                }
            ],
        }
        yield f"data: {json.dumps(chunk)}\n\n"

    # Send final chunk
    final = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }
        ],
    }
    yield f"data: {json.dumps(final)}\n\n"
    yield "data: [DONE]\n\n"


@router.get("/v1/cli/status")
async def cli_status():
    """Get status of all CLI bridges."""
    try:
        get_router_fn = _get_router()
        cli_router = await get_router_fn()
        bridges = cli_router.get_available_bridges()
        return JSONResponse(content={
            "bridges": bridges,
            "total": len(bridges),
            "available": sum(1 for b in bridges if b.get("available")),
        })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
        )


@router.get("/v1/cli/rate-stats")
async def cli_rate_stats():
    """Get rate limit stats for all CLI bridges."""
    try:
        from Cosmos.integration.cli_bridge.rate_tracker import get_rate_tracker
        tracker = get_rate_tracker()
        return JSONResponse(content=tracker.get_all_stats())
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
        )
