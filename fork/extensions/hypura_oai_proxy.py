"""OpenAI-compatible HTTP proxy for Hypura serve.

Maps:
  GET  /v1/models            -> GET  {upstream}/api/tags
  POST /v1/chat/completions -> POST {upstream}/api/chat (NDJSON stream -> OpenAI SSE)

Hermes expects base_url like http://127.0.0.1:8090/v1

Environment:
  HYPURA_OAI_UPSTREAM     Upstream Hypura base URL (default: http://127.0.0.1:8080)
  HYPURA_DEFAULT_MODEL     Model id when request omits model (optional)
  HYPURA_OAI_PROXY_HOST    Bind host (default: 127.0.0.1)
  HYPURA_OAI_PROXY_PORT    Bind port (default: 8090)
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from typing import Any, AsyncIterator

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

logger = logging.getLogger(__name__)

app = FastAPI(title="Hypura OpenAI Proxy", version="0.1.0")


def _upstream() -> str:
    return os.getenv("HYPURA_OAI_UPSTREAM", "http://127.0.0.1:8080").rstrip("/")


def _timeout() -> httpx.Timeout:
    read_s = float(os.getenv("HYPURA_OAI_READ_TIMEOUT", "1800"))
    return httpx.Timeout(connect=30.0, read=read_s, write=600.0, pool=30.0)


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=_timeout(), follow_redirects=True)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/models")
async def list_models() -> JSONResponse:
    url = f"{_upstream()}/api/tags"
    async with _client() as client:
        try:
            r = await client.get(url)
        except httpx.RequestError as e:
            raise HTTPException(status_code=502, detail=f"upstream unreachable: {e}") from e
    if r.status_code != 200:
        raise HTTPException(status_code=r.status_code, detail=r.text[:2000])

    try:
        tags = r.json()
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=502, detail=f"invalid JSON from upstream: {e}") from e

    models = tags.get("models") or []
    data = []
    for m in models:
        mid = m.get("model") or m.get("name") or "unknown"
        data.append(
            {
                "id": mid,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "hypura",
            }
        )
    return JSONResponse(
        content={"object": "list", "data": data},
        status_code=200,
    )


def _build_hypura_chat_payload(body: dict[str, Any], default_model: str) -> dict[str, Any]:
    model = (body.get("model") or "").strip() or default_model
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise HTTPException(status_code=400, detail="messages required")

    stream = bool(body.get("stream", False))
    options: dict[str, Any] = {}
    if body.get("temperature") is not None:
        options["temperature"] = float(body["temperature"])
    if body.get("max_tokens") is not None:
        options["num_predict"] = int(body["max_tokens"])
    if body.get("top_p") is not None:
        options["top_p"] = float(body["top_p"])

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": stream,
        "options": options,
    }
    # Hypura accepts but ignores tools — strip to avoid oversized payloads
    if body.get("tools"):
        payload["tools"] = body["tools"]
    return payload


async def _resolve_default_model(client: httpx.AsyncClient) -> str:
    env = (os.getenv("HYPURA_DEFAULT_MODEL") or "").strip()
    if env:
        return env
    r = await client.get(f"{_upstream()}/api/tags")
    if r.status_code != 200:
        return "hypura"
    try:
        tags = r.json()
        models = tags.get("models") or []
        if models:
            m0 = models[0]
            return str(m0.get("model") or m0.get("name") or "hypura")
    except Exception:
        pass
    return "hypura"


def _openai_nonstream_response(
    hypura_obj: dict[str, Any],
    chatcmpl_id: str,
    created: int,
    request_model: str,
) -> dict[str, Any]:
    msg = hypura_obj.get("message") or {}
    content = msg.get("content") or ""
    prompt_t = hypura_obj.get("prompt_eval_count")
    comp_t = hypura_obj.get("eval_count")
    usage: dict[str, Any] = {}
    if prompt_t is not None:
        usage["prompt_tokens"] = int(prompt_t)
    if comp_t is not None:
        usage["completion_tokens"] = int(comp_t)
    if usage:
        usage["total_tokens"] = usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)

    return {
        "id": chatcmpl_id,
        "object": "chat.completion",
        "created": created,
        "model": request_model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": usage or None,
    }


async def _ndjson_to_sse(
    lines_iter: AsyncIterator[str],
    model: str,
) -> AsyncIterator[str]:
    chatcmpl_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    first = True
    last_usage: dict[str, Any] = {}

    async for line in lines_iter:
        line = line.strip()
        if not line:
            continue
        try:
            chunk = json.loads(line)
        except json.JSONDecodeError:
            logger.warning("skip bad NDJSON line: %s", line[:200])
            continue

        done = bool(chunk.get("done"))
        message = chunk.get("message") or {}
        piece = message.get("content") or ""
        hypura_model = chunk.get("model") or model

        if not done:
            delta: dict[str, Any] = {}
            if first:
                delta["role"] = "assistant"
                first = False
            if piece:
                delta["content"] = piece
            if not delta:
                continue
            payload = {
                "id": chatcmpl_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": hypura_model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
            }
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
        else:
            pt = chunk.get("prompt_eval_count")
            ct = chunk.get("eval_count")
            if pt is not None:
                last_usage["prompt_tokens"] = int(pt)
            if ct is not None:
                last_usage["completion_tokens"] = int(ct)
            if last_usage:
                last_usage["total_tokens"] = last_usage.get("prompt_tokens", 0) + last_usage.get(
                    "completion_tokens", 0
                )

            final_payload: dict[str, Any] = {
                "id": chatcmpl_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": hypura_model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            if last_usage:
                final_payload["usage"] = last_usage
            yield f"data: {json.dumps(final_payload, ensure_ascii=False)}\n\n"

    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions", response_model=None)
async def chat_completions(request: Request) -> JSONResponse | StreamingResponse:
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="invalid JSON body") from None

    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="body must be a JSON object")

    async with _client() as client:
        default_model = await _resolve_default_model(client)
    hypura_body = _build_hypura_chat_payload(body, default_model)
    req_model = hypura_body["model"]
    stream = hypura_body["stream"]
    url = f"{_upstream()}/api/chat"

    if not stream:
        async with _client() as client:
            try:
                r = await client.post(url, json=hypura_body)
            except httpx.RequestError as e:
                raise HTTPException(status_code=502, detail=f"upstream unreachable: {e}") from e
        if r.status_code != 200:
            raise HTTPException(status_code=r.status_code, detail=r.text[:2000])
        try:
            hypura_resp = r.json()
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=502, detail=f"invalid JSON: {e}") from e
        chatcmpl_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())
        out = _openai_nonstream_response(hypura_resp, chatcmpl_id, created, req_model)
        return JSONResponse(content={k: v for k, v in out.items() if v is not None})

    async def sse_body() -> AsyncIterator[bytes]:
        async with _client() as client:
            async with client.stream("POST", url, json=hypura_body) as r:
                if r.status_code != 200:
                    detail = (await r.aread()).decode("utf-8", errors="replace")[:2000]
                    raise HTTPException(status_code=r.status_code, detail=detail)

                async def line_source() -> AsyncIterator[str]:
                    async for line in r.aiter_lines():
                        yield line

                async for part in _ndjson_to_sse(line_source(), req_model):
                    yield part.encode("utf-8")

    return StreamingResponse(
        sse_body(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def main() -> None:
    import uvicorn

    host = os.getenv("HYPURA_OAI_PROXY_HOST", "127.0.0.1")
    port = int(os.getenv("HYPURA_OAI_PROXY_PORT", "8090"))
    logging.basicConfig(level=os.getenv("HYPURA_OAI_LOG_LEVEL", "INFO"))
    uvicorn.run("hypura_oai_proxy:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
