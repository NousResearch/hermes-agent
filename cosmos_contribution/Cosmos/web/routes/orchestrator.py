"""
Cosmos Token Orchestrator Routes

API endpoints for monitoring and controlling the dynamic token orchestrator.

Routes:
    GET  /api/orchestrator/dashboard    - Real-time token usage dashboard
    GET  /api/orchestrator/agents       - Per-agent budget status
    POST /api/orchestrator/rebalance    - Trigger manual rebalance
    POST /api/orchestrator/tandem       - Start Grok+Kimi tandem session
    GET  /api/orchestrator/tandem/{id}  - Get tandem session status
    GET  /api/orchestrator/efficiency   - Efficiency metrics
"""

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
from loguru import logger

# Guard orchestrator import
try:
    from Cosmos.core.token_orchestrator import get_token_orchestrator
    ORCHESTRATOR_AVAILABLE = True
except ImportError:
    ORCHESTRATOR_AVAILABLE = False
    get_token_orchestrator = None

router = APIRouter(prefix="/api/orchestrator", tags=["Token Orchestrator"])


# =============================================================================
# REQUEST MODELS
# =============================================================================

class TandemRequest(BaseModel):
    task: str
    task_type: str = "chat"

class TandemHandoffRequest(BaseModel):
    session_id: str
    context_summary: str


# =============================================================================
# ROUTES
# =============================================================================

@router.get("/dashboard")
async def orchestrator_dashboard():
    """
    Real-time token usage dashboard.

    Returns comprehensive view of all agent budgets, active tandem sessions,
    efficiency metrics, and daily budget status.
    """
    if not ORCHESTRATOR_AVAILABLE:
        return JSONResponse(
            status_code=503,
            content={"error": "Token orchestrator not available"},
        )

    try:
        orchestrator = get_token_orchestrator()
        dashboard = orchestrator.get_dashboard()
        return JSONResponse(content={
            "success": True,
            "dashboard": dashboard,
        })
    except Exception as e:
        logger.error(f"Orchestrator dashboard error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Dashboard error: {str(e)}"},
        )


@router.get("/agents")
async def orchestrator_agents():
    """
    Per-agent budget status.

    Returns individual budget, usage, efficiency score, and tier for each agent.
    """
    if not ORCHESTRATOR_AVAILABLE:
        return JSONResponse(
            status_code=503,
            content={"error": "Token orchestrator not available"},
        )

    try:
        orchestrator = get_token_orchestrator()
        dashboard = orchestrator.get_dashboard()
        return JSONResponse(content={
            "success": True,
            "agents": dashboard.get("agents", {}),
            "total_agents": len(dashboard.get("agents", {})),
        })
    except Exception as e:
        logger.error(f"Orchestrator agents error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Agent status error: {str(e)}"},
        )


@router.post("/rebalance")
async def orchestrator_rebalance():
    """
    Trigger manual budget rebalance.

    Redistributes token budgets based on current usage patterns.
    Normally runs automatically every 5 minutes.
    """
    if not ORCHESTRATOR_AVAILABLE:
        return JSONResponse(
            status_code=503,
            content={"error": "Token orchestrator not available"},
        )

    try:
        orchestrator = get_token_orchestrator()
        await orchestrator.rebalance()
        dashboard = orchestrator.get_dashboard()
        return JSONResponse(content={
            "success": True,
            "message": "Rebalance completed",
            "dashboard": dashboard,
        })
    except Exception as e:
        logger.error(f"Orchestrator rebalance error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Rebalance error: {str(e)}"},
        )


@router.post("/tandem")
async def orchestrator_tandem(body: TandemRequest):
    """
    Start a Grok+Kimi tandem session.

    Grok leads for: research, real-time data, X/Twitter, humor, controversy.
    Kimi leads for: long documents, complex reasoning, synthesis, planning.
    Auto-detects primary based on task content.

    Returns the tandem session with initial response from primary agent.
    """
    if not ORCHESTRATOR_AVAILABLE:
        return JSONResponse(
            status_code=503,
            content={"error": "Token orchestrator not available"},
        )

    try:
        orchestrator = get_token_orchestrator()
        session = await orchestrator.start_tandem(
            task=body.task,
            task_type=body.task_type,
        )

        return JSONResponse(content={
            "success": True,
            "session": {
                "session_id": session.session_id,
                "primary": session.primary,
                "secondary": session.secondary,
                "primary_budget": session.primary_budget,
                "secondary_budget": session.secondary_budget,
                "task_type": session.task_type,
                "started_at": session.started_at.isoformat(),
            },
        })
    except Exception as e:
        logger.error(f"Orchestrator tandem start error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Tandem start error: {str(e)}"},
        )


@router.get("/tandem/{session_id}")
async def orchestrator_tandem_status(session_id: str):
    """Get status of a specific tandem session."""
    if not ORCHESTRATOR_AVAILABLE:
        return JSONResponse(
            status_code=503,
            content={"error": "Token orchestrator not available"},
        )

    try:
        orchestrator = get_token_orchestrator()
        sessions = orchestrator._tandem_sessions
        if session_id not in sessions:
            return JSONResponse(
                status_code=404,
                content={"error": f"Tandem session {session_id} not found"},
            )

        session = sessions[session_id]
        return JSONResponse(content={
            "success": True,
            "session": {
                "session_id": session.session_id,
                "primary": session.primary,
                "secondary": session.secondary,
                "shared_context": session.shared_context[:200] + "..."
                if len(session.shared_context) > 200 else session.shared_context,
                "primary_budget": session.primary_budget,
                "secondary_budget": session.secondary_budget,
                "handoff_count": session.handoff_count,
                "total_tokens_used": session.total_tokens_used,
                "started_at": session.started_at.isoformat(),
                "task_type": session.task_type,
            },
        })
    except Exception as e:
        logger.error(f"Orchestrator tandem status error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Tandem status error: {str(e)}"},
        )


@router.post("/tandem/handoff")
async def orchestrator_tandem_handoff(body: TandemHandoffRequest):
    """
    Execute a handoff within a tandem session.

    Passes compressed context from primary to secondary agent.
    Returns the secondary agent's response.
    """
    if not ORCHESTRATOR_AVAILABLE:
        return JSONResponse(
            status_code=503,
            content={"error": "Token orchestrator not available"},
        )

    try:
        orchestrator = get_token_orchestrator()
        response = await orchestrator.tandem_handoff(
            session_id=body.session_id,
            context_summary=body.context_summary,
        )
        return JSONResponse(content={
            "success": True,
            "response": response,
            "session_id": body.session_id,
        })
    except Exception as e:
        logger.error(f"Orchestrator tandem handoff error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Tandem handoff error: {str(e)}"},
        )


@router.get("/efficiency")
async def orchestrator_efficiency():
    """
    Efficiency metrics for the collective.

    Returns per-agent efficiency scores, top performer, and overall rating.
    """
    if not ORCHESTRATOR_AVAILABLE:
        return JSONResponse(
            status_code=503,
            content={"error": "Token orchestrator not available"},
        )

    try:
        orchestrator = get_token_orchestrator()
        dashboard = orchestrator.get_dashboard()

        # Extract efficiency-specific data
        agents = dashboard.get("agents", {})
        efficiency_data = {}
        for agent_id, info in agents.items():
            efficiency_data[agent_id] = {
                "efficiency_score": info.get("efficiency_score", 1.0),
                "requests_count": info.get("requests_count", 0),
                "used_tokens": info.get("used_tokens", 0),
                "tier": info.get("tier", "unknown"),
            }

        return JSONResponse(content={
            "success": True,
            "efficiency_rating": dashboard.get("efficiency_rating", 1.0),
            "top_performer": dashboard.get("top_performer", "none"),
            "agents": efficiency_data,
            "daily_budget_remaining": dashboard.get("total_remaining", 0),
        })
    except Exception as e:
        logger.error(f"Orchestrator efficiency error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Efficiency metrics error: {str(e)}"},
        )
