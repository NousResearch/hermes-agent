"""
COSMOS Mesh Dashboard — API Routes
===================================
Serves the mesh dashboard UI and provides the /api/mesh/status endpoint
that the dashboard polls for real-time COSMOS network state.
"""
import time
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse
from loguru import logger

router = APIRouter()


def _get_templates():
    try:
        from Cosmos.web.server import templates
        return templates
    except Exception:
        from fastapi.templating import Jinja2Templates
        return Jinja2Templates(directory=str(Path(__file__).parent.parent / "templates"))


def _get_node():
    """Get the running COSMOS node, if available."""
    try:
        from Cosmos.network.cosmos_node import get_cosmos_node
        return get_cosmos_node()
    except Exception:
        return None


def _get_latent_router():
    """Get latent router directly if node not running (for local dev)."""
    node = _get_node()
    if node and node._latent_router:
        return node._latent_router
    return None


# ── Dashboard Page ─────────────────────────────────────────

@router.get("/mesh", response_class=HTMLResponse)
async def mesh_dashboard(request: Request):
    """Serve the COSMOS Mesh Protocol dashboard."""
    tpl = _get_templates()
    return tpl.TemplateResponse("mesh_dashboard.html", {"request": request})


# ── API Endpoints ──────────────────────────────────────────

@router.get("/api/mesh/status")
async def mesh_status():
    """
    Get full COSMOS mesh status for the dashboard.

    Returns node info, peer connections, latent router stats,
    PoI consensus, attestation chain, and swarm memory state.
    """
    node = _get_node()

    if node:
        # Full live status from running node
        status = node.get_status()

        # Enrich latent router data with recent routes for the feed
        if node._latent_router:
            route_stats = node._latent_router.get_route_stats()
            status["latent_router"] = route_stats

        return JSONResponse(status)

    # Fallback: return skeleton data so dashboard doesn't break
    return JSONResponse({
        "node_name": "not-running",
        "version": "2.0.0",
        "identity": "0" * 16,
        "gpu_fingerprint": "0" * 16,
        "port": 9999,
        "connected_peers": [],
        "peer_count": 0,
        "local_bots": [],
        "all_bots": {},
        "mesh_root": "0" * 16,
        "mesh_peers": 0,
        "mesh_sequence": 0,
        "pending_approvals": 0,
        "poi": {"active_rounds": 0, "completed_proofs": 0},
        "latent_router": {
            "total_routes": 0,
            "avg_confidence": 0.0,
            "methods": {},
            "bot_distribution": {},
            "models": {},
            "recent_routes": [],
        },
        "attestation": {
            "node": "not-running",
            "gpu_model": "",
            "chain_length": 0,
            "last_seal": "0" * 32,
            "models_attested": [],
            "remote_nodes_tracked": [],
            "trust_scores": {},
        },
        "swarm_memory": {
            "total_crystals": 0,
            "by_status": {},
            "total_verifications": 0,
            "graph_edges": 0,
            "unique_tags": [],
        },
    })


@router.get("/api/mesh/nodes")
async def mesh_nodes():
    """Get list of all known nodes (connected and disconnected)."""
    node = _get_node()
    if not node:
        return JSONResponse({"nodes": []})

    nodes = []
    # Local node
    nodes.append({
        "name": node.node_name,
        "status": "online",
        "bots": node.get_local_bots(),
        "identity": node._identity[:8].hex() if node._identity else "",
    })
    # Peers
    for peer_name, peer in node._peers.items():
        nodes.append({
            "name": peer_name,
            "status": "connected" if peer.verified else "unverified",
            "bots": peer.remote_bots,
            "last_heartbeat": peer.last_heartbeat,
        })

    return JSONResponse({"nodes": nodes})


@router.get("/api/mesh/latent/route")
async def latent_route_test(prompt: str = "What is the capital of France?"):
    """Test latent routing — see which model would be selected for a prompt."""
    node = _get_node()
    if not node or not node._latent_router:
        return JSONResponse({"error": "Latent router not available"}, status_code=503)

    decision = node._latent_router.route(prompt(node.get_all_bots().keys()))
    return JSONResponse({
        "prompt": prompt,
        "selected_bot": decision.selected_bot,
        "confidence": round(decision.confidence, 4),
        "method": decision.method,
        "all_scores": {k: round(v, 4) for k, v in decision.all_scores.items()},
        "query_dimensions": {k: round(v, 3) for k, v in decision.query_dimensions.items()},
    })


@router.post("/api/mesh/query")
async def mesh_query(request: Request):
    """
    Send a prompt through the COSMOS mesh — latent routes to best model,
    runs inference, returns result with routing metadata.
    """
    import asyncio
    body = await request.json()
    prompt = body.get("prompt", "").strip()
    if not prompt:
        return JSONResponse({"error": "prompt required"}, status_code=400)

    node = _get_node()
    if not node:
        return JSONResponse({"error": "COSMOS node not running"}, status_code=503)

    # Latent route
    available = list(node.get_all_bots().keys())
    decision = node._latent_router.route(prompt, available)

    result = {
        "prompt": prompt,
        "routed_to": decision.selected_bot,
        "confidence": round(decision.confidence, 4),
        "method": decision.method,
        "dimensions": {k: round(v, 3) for k, v in decision.query_dimensions.items() if v > 0.05},
        "all_scores": {k: round(v, 4) for k, v in decision.all_scores.items()},
        "response": None,
        "inference_ms": 0,
    }

    # Run inference through the mesh
    t0 = time.time()
    try:
        bot_name = decision.selected_bot
        if bot_name in (node._local_bots or {}):
            resp = await asyncio.wait_for(
                node._local_bots[bot_name](prompt, 1000), timeout=60
            )
        else:
            resp = await asyncio.wait_for(
                node.query_remote_bot(bot_name, prompt, 1000, 60), timeout=60
            )
        result["response"] = resp
        result["inference_ms"] = round((time.time() - t0) * 1000)

        # Record feedback for the latent router
        if resp:
            quality = min(1.0, len(resp) / 200)  # Simple quality heuristic
            node._latent_router.record_outcome(
                decision, quality, result["inference_ms"]
            )
    except Exception as e:
        result["response"] = f"[Error: {str(e)[:200]}]"
        result["inference_ms"] = round((time.time() - t0) * 1000)

    return JSONResponse(result)
