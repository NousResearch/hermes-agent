"""
Claude Teams API Routes - AGI v1.9 Integration

Cosmos is the ORCHESTRATOR. Claude teams are WORKERS.

Endpoints:
- POST /api/claude/delegate - Delegate single task
- POST /api/claude/team - Create team for complex task
- POST /api/claude/plan - Create orchestration plan
- POST /api/claude/plan/{plan_id}/execute - Execute plan
- POST /api/claude/hybrid - Hybrid deliberation
- GET /api/claude/teams - list Claude teams
- GET /api/claude/switches - Get agent switch states
- POST /api/claude/switches/{agent} - Set agent switch
- POST /api/claude/switches/bulk - Set multiple switches
- POST /api/claude/priority - Set model priority
- GET /api/claude/stats - Integration statistics
- GET /api/claude/mcp/tools - MCP tools available
- GET /api/claude/delegations - Delegation history
- POST /api/claude/quick/research - Quick research
- POST /api/claude/quick/code - Quick coding
- POST /api/claude/quick/analyze - Quick analysis
- POST /api/claude/quick/critique - Quick critique
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter()


# ============================================
# REQUEST MODELS
# ============================================

class ClaudeDelegateRequest(BaseModel):
    task: str
    task_type: str = "analysis"
    model: str = "sonnet"
    timeout: float = 120.0
    context: Optional[dict] = None
    constraints: Optional[list[str]] = None


class ClaudeTeamRequest(BaseModel):
    task: str
    team_name: str = "task_force"
    team_purpose: Optional[str] = None
    roles: Optional[list[str]] = None
    model: str = "sonnet"
    timeout: float = 300.0


class OrchestrationPlanRequest(BaseModel):
    name: str
    tasks: list[dict]
    mode: str = "sequential"


# ============================================
# DELEGATION ENDPOINTS
# ============================================

@router.post("/api/claude/delegate")
async def claude_delegate(request: ClaudeDelegateRequest):
    """Cosmos delegates to Claude."""
    try:
        from Cosmos.integration.claude_teams import get_swarm_team_fusion
        from Cosmos.integration.claude_teams.swarm_team_fusion import DelegationType
        from Cosmos.integration.claude_teams.agent_sdk_bridge import ClaudeModel

        fusion = get_swarm_team_fusion()

        result = await fusion.delegate(
            task=request.task,
            delegation_type=DelegationType(request.task_type),
            model=ClaudeModel(request.model),
            context=request.context,
            constraints=request.constraints,
            timeout=request.timeout,
        )

        return JSONResponse({
            "success": True,
            "delegation_id": result.request_id,
            "status": result.status,
            "result": result.result,
            "orchestrator": "cosmos",
            "worker": f"claude_{request.model}",
        })
    except Exception as e:
        logger.error(f"Claude delegation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/claude/team")
async def claude_create_team_task(request: ClaudeTeamRequest):
    """Create a Claude team and delegate a complex task."""
    try:
        from Cosmos.integration.claude_teams import get_swarm_team_fusion
        from Cosmos.integration.claude_teams.team_coordinator import TeamRole
        from Cosmos.integration.claude_teams.agent_sdk_bridge import ClaudeModel

        fusion = get_swarm_team_fusion()

        roles = None
        if request.roles:
            roles = [TeamRole(r) for r in request.roles]

        result = await fusion.delegate_to_team(
            task=request.task,
            team_name=request.team_name,
            team_purpose=request.team_purpose or f"Complete: {request.task[:100]}",
            roles=roles,
            model=ClaudeModel(request.model),
            timeout=request.timeout,
        )

        return JSONResponse({
            "success": True,
            "team_result": result,
            "orchestrator": "cosmos",
        })
    except Exception as e:
        logger.error(f"Claude team error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/claude/plan")
async def claude_create_plan(request: OrchestrationPlanRequest):
    """Create a multi-step orchestration plan."""
    try:
        from Cosmos.integration.claude_teams import get_swarm_team_fusion
        from Cosmos.integration.claude_teams.swarm_team_fusion import OrchestrationMode

        fusion = get_swarm_team_fusion()

        plan = await fusion.create_orchestration_plan(
            name=request.name,
            tasks=request.tasks,
            mode=OrchestrationMode(request.mode),
        )

        return JSONResponse({
            "success": True,
            "plan_id": plan.plan_id,
            "plan_name": plan.name,
            "steps": len(plan.steps),
            "mode": plan.mode.value,
            "orchestrator": "cosmos",
        })
    except Exception as e:
        logger.error(f"Claude plan creation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/claude/plan/{plan_id}/execute")
async def claude_execute_plan(plan_id: str):
    """Execute an orchestration plan."""
    try:
        from Cosmos.integration.claude_teams import get_swarm_team_fusion

        fusion = get_swarm_team_fusion()
        results = await fusion.execute_plan(plan_id)

        return JSONResponse({
            "success": True,
            "plan_id": plan_id,
            "results": results,
            "orchestrator": "cosmos",
        })
    except Exception as e:
        logger.error(f"Claude plan execution error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/claude/hybrid")
async def claude_hybrid_deliberation(question: str, team_id: Optional[str] = None):
    """Run hybrid deliberation - Cosmos swarm + Claude team."""
    try:
        from Cosmos.integration.claude_teams import get_team_coordinator

        coordinator = get_team_coordinator()
        result = await coordinator.hybrid_deliberation(
            topic=question,
            claude_team_id=team_id,
            include_cosmos=True,
        )

        return JSONResponse({
            "success": True,
            "hybrid_result": result,
            "participants": ["cosmos_swarm", "claude_team"],
        })
    except Exception as e:
        logger.error(f"Hybrid deliberation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# TEAM MANAGEMENT
# ============================================

@router.get("/api/claude/teams")
async def claude_list_teams():
    """list all Claude teams."""
    try:
        from Cosmos.integration.claude_teams import get_team_coordinator

        coordinator = get_team_coordinator()
        teams = coordinator.get_teams()

        return JSONResponse({
            "success": True,
            "teams": teams,
            "orchestrator": "cosmos",
        })
    except Exception as e:
        logger.error(f"list teams error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# AGENT SWITCHES
# ============================================

@router.get("/api/claude/switches")
async def claude_get_switches():
    """Get current agent switch states."""
    try:
        from Cosmos.integration.claude_teams import get_swarm_team_fusion

        fusion = get_swarm_team_fusion()
        switches = fusion.get_agent_switches()

        return JSONResponse({
            "success": True,
            "switches": switches,
            "description": "Agent switches - Cosmos controls which Claude agents are active",
        })
    except Exception as e:
        logger.error(f"Get switches error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/claude/switches/{agent}")
async def claude_set_switch(agent: str, enabled: bool = True):
    """Set an agent switch."""
    try:
        from Cosmos.integration.claude_teams import get_swarm_team_fusion

        fusion = get_swarm_team_fusion()
        success = fusion.set_agent_switch(agent, enabled)

        if not success:
            raise HTTPException(status_code=404, detail=f"Unknown agent: {agent}")

        return JSONResponse({
            "success": True,
            "agent": agent,
            "enabled": enabled,
            "message": f"Cosmos has {'enabled' if enabled else 'disabled'} {agent}",
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Set switch error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/claude/switches/bulk")
async def claude_set_switches_bulk(switches: dict[str, bool]):
    """Set multiple agent switches at once."""
    try:
        from Cosmos.integration.claude_teams import get_swarm_team_fusion

        fusion = get_swarm_team_fusion()
        results = {}
        for agent, enabled in switches.items():
            results[agent] = fusion.set_agent_switch(agent, enabled)

        return JSONResponse({
            "success": True,
            "results": results,
            "current_switches": fusion.get_agent_switches(),
        })
    except Exception as e:
        logger.error(f"Bulk switch error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/claude/priority")
async def claude_set_priority(priority: list[str]):
    """Set model priority order for fallback."""
    try:
        from Cosmos.integration.claude_teams import get_swarm_team_fusion

        fusion = get_swarm_team_fusion()
        fusion.set_model_priority(priority)

        return JSONResponse({
            "success": True,
            "priority": priority,
            "message": "Model priority updated",
        })
    except Exception as e:
        logger.error(f"Set priority error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# STATS & INFO
# ============================================

@router.get("/api/claude/stats")
async def claude_integration_stats():
    """Get Claude Teams integration statistics."""
    try:
        from Cosmos.integration.claude_teams import get_swarm_team_fusion

        fusion = get_swarm_team_fusion()
        stats = fusion.get_stats()

        return JSONResponse({
            "success": True,
            "stats": stats,
            "integration": "AGI v1.9 - Claude Teams Fusion",
            "description": "Cosmos orchestrates, Claude executes",
        })
    except Exception as e:
        logger.error(f"Claude stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/claude/mcp/tools")
async def claude_mcp_tools(team_id: Optional[str] = None):
    """list MCP tools available to Claude teams."""
    try:
        from Cosmos.integration.claude_teams import get_mcp_server

        mcp = get_mcp_server()
        tools = mcp.list_tools(team_id)

        return JSONResponse({
            "success": True,
            "tools": tools,
            "description": "Cosmos tools exposed to Claude via MCP",
        })
    except Exception as e:
        logger.error(f"MCP tools error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/claude/delegations")
async def claude_recent_delegations(limit: int = 10):
    """Get recent delegation history."""
    try:
        from Cosmos.integration.claude_teams import get_swarm_team_fusion

        fusion = get_swarm_team_fusion()
        delegations = fusion.get_recent_delegations(limit)

        return JSONResponse({
            "success": True,
            "delegations": delegations,
            "orchestrator": "cosmos",
        })
    except Exception as e:
        logger.error(f"Delegations history error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# QUICK DELEGATION ENDPOINTS
# ============================================

@router.post("/api/claude/quick/research")
async def claude_quick_research(topic: str, model: str = "haiku"):
    """Quick research delegation."""
    try:
        from Cosmos.integration.claude_teams import get_swarm_team_fusion
        from Cosmos.integration.claude_teams.agent_sdk_bridge import ClaudeModel

        fusion = get_swarm_team_fusion()
        result = await fusion.quick_research(topic, ClaudeModel(model))

        return JSONResponse({"success": True, "research": result, "model": model})
    except Exception as e:
        logger.error(f"Quick research error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/claude/quick/code")
async def claude_quick_code(task: str, model: str = "sonnet"):
    """Quick coding delegation."""
    try:
        from Cosmos.integration.claude_teams import get_swarm_team_fusion
        from Cosmos.integration.claude_teams.agent_sdk_bridge import ClaudeModel

        fusion = get_swarm_team_fusion()
        result = await fusion.quick_code(task, ClaudeModel(model))

        return JSONResponse({"success": True, "code": result, "model": model})
    except Exception as e:
        logger.error(f"Quick code error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/claude/quick/analyze")
async def claude_quick_analyze(data: str, model: str = "sonnet"):
    """Quick analysis delegation."""
    try:
        from Cosmos.integration.claude_teams import get_swarm_team_fusion
        from Cosmos.integration.claude_teams.agent_sdk_bridge import ClaudeModel

        fusion = get_swarm_team_fusion()
        result = await fusion.quick_analyze(data, ClaudeModel(model))

        return JSONResponse({"success": True, "analysis": result, "model": model})
    except Exception as e:
        logger.error(f"Quick analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/claude/quick/critique")
async def claude_quick_critique(work: str, model: str = "opus"):
    """Quick critique delegation - uses Opus for depth."""
    try:
        from Cosmos.integration.claude_teams import get_swarm_team_fusion
        from Cosmos.integration.claude_teams.agent_sdk_bridge import ClaudeModel

        fusion = get_swarm_team_fusion()
        result = await fusion.quick_critique(work, ClaudeModel(model))

        return JSONResponse({"success": True, "critique": result, "model": model})
    except Exception as e:
        logger.error(f"Quick critique error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
