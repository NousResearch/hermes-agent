"""Prompt Optimizer plugin — backend API routes.

Mounted at /api/plugins/prompt-optimizer/ by the dashboard plugin system
(the same FastAPI app the desktop ``hermes serve`` backend runs).

Why this backend exists
-----------------------
The desktop UI plugin used to call the gateway JSON-RPC ``llm.oneshot`` via
``host.request``, but the desktop renderer's JsonRpcGatewayClient enforces a
hard 30s request timeout (``DEFAULT_GATEWAY_REQUEST_TIMEOUT_MS`` in
``apps/desktop/src/hermes.ts``). Prompt optimization on slow providers
regularly exceeds 30s (measured 28–52s), so every click failed with
"request timed out after 30s: llm.oneshot".

``host.rest`` on the other hand accepts an explicit ``timeoutMs`` (passed
through Electron main → HTTP request), so the plugin now POSTs here with a
5-minute window and this module runs the one-shot call with the same 5-minute
deadline, inheriting the live session's model when the session is resolvable.

Model inheritance mirrors tui_gateway/server.py ``_main_runtime_from_agent``:
the one-shot rides the same provider/model/credentials as the session the
user is looking at, falling back to the configured auxiliary backend when no
live agent is found (e.g. session already closed). Never touches session
history, so prompt caching is unaffected.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

log = logging.getLogger(__name__)

router = APIRouter()

# Mirror of tui_gateway/server.py:_main_runtime_from_agent — builds the aux
# client's main_runtime override from a live agent so the optimization uses
# the same model the user is chatting with.
_AGENT_RUNTIME_FIELDS = ("provider", "model", "base_url", "api_key", "api_mode", "auth_mode")


def _main_runtime_from_agent(agent: Any) -> Optional[Dict[str, Any]]:
    if agent is None:
        return None
    runtime: Dict[str, Any] = {}
    for field in _AGENT_RUNTIME_FIELDS:
        value = getattr(agent, field, None)
        if isinstance(value, str) and value.strip():
            runtime[field] = value.strip()
        elif field == "api_key" and callable(value):
            runtime[field] = value
    return runtime or None


def _session_agent(session_id: Optional[str]) -> Any:
    """Resolve the live agent for a session id, if the in-process tui_gateway
    session table has it. Fails open (None) on any import/access hiccup so a
    core refactor degrades to the auxiliary backend instead of breaking the
    plugin."""
    if not session_id:
        return None
    try:
        from tui_gateway import methods_session as ms

        session = ms._sessions.get(session_id)
        return session.get("agent") if session else None
    except Exception:
        log.warning("prompt-optimizer: session agent lookup failed", exc_info=True)
        return None


class OptimizeRequest(BaseModel):
    input: str
    instructions: str = ""
    session_id: Optional[str] = None
    max_tokens: int = 2000
    temperature: float = 0.3
    timeout: float = 300.0


class OptimizeResponse(BaseModel):
    text: str


@router.post("/optimize", response_model=OptimizeResponse)
def optimize(req: OptimizeRequest) -> OptimizeResponse:
    if not (req.input or "").strip():
        raise HTTPException(status_code=400, detail="empty input")

    from agent.oneshot import run_oneshot

    try:
        text = run_oneshot(
            instructions=req.instructions,
            user_input=req.input,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            timeout=req.timeout,
            main_runtime=_main_runtime_from_agent(_session_agent(req.session_id)),
        )
    except Exception as exc:
        log.warning("prompt-optimizer: oneshot failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"one-shot generation failed: {exc}") from exc

    if not (text or "").strip():
        raise HTTPException(status_code=502, detail="empty result from model")

    return OptimizeResponse(text=text)
