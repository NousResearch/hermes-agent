"""
Cosmos External Gateway Routes - "The Window"

API endpoints for external agent communication with the collective.
Rate-limited, sandboxed, with full audit logging.

Routes:
    POST /api/gateway/query     - External agents send queries here
    GET  /api/gateway/stats     - Gateway statistics (admin only)
    GET  /api/gateway/audit     - Audit log (admin only)
    POST /api/gateway/block     - Block a client (admin only)
    POST /api/gateway/unblock   - Unblock a client (admin only)
    POST /api/gateway/shutdown  - Emergency shutdown (admin only)
    POST /api/gateway/enable    - Re-enable after shutdown (admin only)
"""

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
from loguru import logger

# Guard gateway import
try:
    from Cosmos.core.external_gateway import get_external_gateway
    GATEWAY_AVAILABLE = True
except ImportError:
    GATEWAY_AVAILABLE = False
    get_external_gateway = None

router = APIRouter(prefix="/api/gateway", tags=["External Gateway"])


# =============================================================================
# REQUEST MODELS
# =============================================================================

class GatewayQueryRequest(BaseModel):
    input: str
    context: Optional[str] = None

class BlockClientRequest(BaseModel):
    client_id: str
    reason: str = "Manually blocked by admin"

class UnblockClientRequest(BaseModel):
    client_id: str


# =============================================================================
# ROUTES
# =============================================================================

@router.post("/query")
async def gateway_query(request: Request, body: GatewayQueryRequest):
    """
    External agents send queries here.

    This is the main public-facing endpoint. Rate-limited to 5 req/min per IP.
    All inputs pass through injection defense. All outputs are scrubbed of secrets.
    """
    if not GATEWAY_AVAILABLE:
        return JSONResponse(
            status_code=503,
            content={"error": "Gateway not available", "status": "unavailable"},
        )

    gateway = get_external_gateway()

    # Extract client info
    client_ip = request.client.host if request.client else "unknown"
    user_agent = request.headers.get("user-agent", "")

    try:
        result = await gateway.handle_request(
            input_text=body.input,
            client_ip=client_ip,
            user_agent=user_agent,
        )

        # Determine HTTP status based on gateway result
        status = result.get("status", "error")
        if status == "ok":
            return JSONResponse(content=result)
        elif status == "rate_limited":
            return JSONResponse(status_code=429, content=result)
        elif status == "blocked":
            return JSONResponse(status_code=403, content=result)
        elif status == "disabled":
            return JSONResponse(status_code=503, content=result)
        elif status == "invalid":
            return JSONResponse(status_code=400, content=result)
        else:
            return JSONResponse(status_code=500, content=result)

    except Exception as e:
        logger.error(f"Gateway query error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Internal gateway error", "status": "error"},
        )


@router.get("/stats")
async def gateway_stats():
    """
    Gateway statistics. Admin endpoint.

    Returns total requests, blocks, unique clients, threat distribution, etc.
    """
    if not GATEWAY_AVAILABLE:
        return JSONResponse(
            status_code=503,
            content={"error": "Gateway not available"},
        )

    gateway = get_external_gateway()
    return JSONResponse(content={
        "success": True,
        "stats": gateway.get_stats(),
    })


@router.get("/audit")
async def gateway_audit(last_n: int = 100):
    """
    Audit log of recent requests and responses. Admin endpoint.

    Args:
        last_n: Number of recent entries to return (default 100, max 1000).
    """
    if not GATEWAY_AVAILABLE:
        return JSONResponse(
            status_code=503,
            content={"error": "Gateway not available"},
        )

    last_n = min(last_n, 1000)
    gateway = get_external_gateway()
    audit = gateway.get_audit_log(last_n=last_n)
    return JSONResponse(content={
        "success": True,
        "count": len(audit),
        "audit": audit,
    })


@router.post("/block")
async def gateway_block_client(body: BlockClientRequest):
    """Block a client from the gateway. Admin endpoint."""
    if not GATEWAY_AVAILABLE:
        return JSONResponse(
            status_code=503,
            content={"error": "Gateway not available"},
        )

    gateway = get_external_gateway()
    gateway.block_client(body.client_id, body.reason)
    return JSONResponse(content={
        "success": True,
        "message": f"Client {body.client_id} blocked: {body.reason}",
    })


@router.post("/unblock")
async def gateway_unblock_client(body: UnblockClientRequest):
    """Unblock a previously blocked client. Admin endpoint."""
    if not GATEWAY_AVAILABLE:
        return JSONResponse(
            status_code=503,
            content={"error": "Gateway not available"},
        )

    gateway = get_external_gateway()
    gateway.unblock_client(body.client_id)
    return JSONResponse(content={
        "success": True,
        "message": f"Client {body.client_id} unblocked",
    })


@router.post("/shutdown")
async def gateway_shutdown():
    """Emergency shutdown of the gateway. Admin endpoint."""
    if not GATEWAY_AVAILABLE:
        return JSONResponse(
            status_code=503,
            content={"error": "Gateway not available"},
        )

    gateway = get_external_gateway()
    await gateway.emergency_shutdown()
    return JSONResponse(content={
        "success": True,
        "message": "Gateway has been shut down. Use /api/gateway/enable to re-enable.",
    })


@router.post("/enable")
async def gateway_enable():
    """Re-enable the gateway after shutdown. Admin endpoint."""
    if not GATEWAY_AVAILABLE:
        return JSONResponse(
            status_code=503,
            content={"error": "Gateway not available"},
        )

    gateway = get_external_gateway()
    await gateway.enable()
    return JSONResponse(content={
        "success": True,
        "message": "Gateway re-enabled and accepting requests.",
    })
