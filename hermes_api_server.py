"""Minimal FastAPI wrapper for Hermes Agent.

Endpoints:
- GET  /healthz
- GET  /status
- POST /send
"""

from __future__ import annotations

import os
import threading
from hmac import compare_digest
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, status as http_status
from pydantic import BaseModel, Field

from gateway.status import is_gateway_running
from hermes_cli.auth import has_usable_secret
from run_agent import AIAgent


class SendRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User message to Hermes")
    session_id: str | None = Field(default=None, description="Optional session identifier")
    model: str | None = Field(default=None, description="Optional model override for this request")


class SendResponse(BaseModel):
    ok: bool
    response: str
    model: str
    session_id: str | None = None


app = FastAPI(title="Hermes API", version="0.1.0", docs_url=None, redoc_url=None, openapi_url=None)
_agent_lock = threading.Lock()


def _require_api_key(authorization: str | None = Header(default=None)) -> None:
    api_key = os.getenv("API_SERVER_KEY", "")
    if not has_usable_secret(api_key, min_length=16):
        raise HTTPException(status_code=http_status.HTTP_503_SERVICE_UNAVAILABLE, detail="API server is not configured")
    token = authorization[7:].strip() if authorization and authorization.startswith("Bearer ") else ""
    if not compare_digest(token, api_key):
        raise HTTPException(status_code=http_status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/status")
def status(_: None = Depends(_require_api_key)) -> dict[str, Any]:
    gateway_running = False
    gateway_error = None
    try:
        gateway_running = bool(is_gateway_running())
    except Exception as e:
        gateway_error = str(e)

    return {
        "status": "ok",
        "gateway_running": gateway_running,
        "gateway_error": gateway_error,
    }


@app.post("/send", response_model=SendResponse)
def send(req: SendRequest, _: None = Depends(_require_api_key)) -> SendResponse:
    chosen_model = req.model or os.getenv("HERMES_API_MODEL") or ""
    kwargs: dict[str, Any] = {}
    if chosen_model:
        kwargs["model"] = chosen_model
    if req.session_id:
        kwargs["session_id"] = req.session_id

    try:
        with _agent_lock:
            agent = AIAgent(**kwargs)
            result = agent.chat(req.message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"send failed: {e}") from e

    return SendResponse(
        ok=True,
        response=result,
        model=chosen_model or "(config default)",
        session_id=req.session_id,
    )


def main() -> None:
    import uvicorn

    api_key = os.getenv("API_SERVER_KEY", "")
    if not has_usable_secret(api_key, min_length=16):
        raise RuntimeError("API_SERVER_KEY must be a non-placeholder secret of at least 16 characters")

    host = os.getenv("HERMES_API_HOST", "127.0.0.1")
    port = int(os.getenv("HERMES_API_PORT", "8080"))
    uvicorn.run("hermes_api_server:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
