"""
FORGE API Routes
=================

Endpoints for the FORGE (Cosmos Orchestrated Rapid Generation Engine)
development orchestration system.

Endpoints:
  POST /api/forge/quick        - Quick-mode plan+execute
  POST /api/forge/research     - Multi-model parallel research
  POST /api/forge/deliberate   - PROPOSE-CRITIQUE-REFINE-VOTE on a plan
  POST /api/forge/project      - Initialize a new FORGE project
  GET  /api/forge/progress     - Get project progress
  GET  /api/forge/cost         - Get cost breakdown
  GET  /api/forge/status       - FORGE system status
"""

import asyncio
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from loguru import logger


router = APIRouter(prefix="/api/forge")


# =============================================================================
# REQUEST MODELS
# =============================================================================

class QuickRequest(BaseModel):
    task: str
    workspace: str = "/workspace/Cosmos"

class ResearchRequest(BaseModel):
    topic: str
    workspace: str = "/workspace/Cosmos"

class DeliberateRequest(BaseModel):
    objective: str
    context: str = ""
    models: list = None
    workspace: str = "/workspace/Cosmos"

class ProjectInitRequest(BaseModel):
    name: str
    description: str
    workspace: str = "/workspace/Cosmos"

class AddPhaseRequest(BaseModel):
    name: str
    description: str
    goals: list
    success_criteria: list
    workspace: str = "/workspace/Cosmos"


# =============================================================================
# HELPER
# =============================================================================

def _get_engine(workspace: str = "/workspace/Cosmos"):
    """Get or create a ForgeEngine instance."""
    from Cosmos.core.forge.forge_engine import ForgeEngine
    return ForgeEngine(workspace)


# =============================================================================
# ENDPOINTS
# =============================================================================

@router.get("/status")
async def forge_status():
    """Get FORGE system status."""
    try:
        from Cosmos.core.forge import __version__
        from Cosmos.core.collective.persistent_agent import get_shadow_agents

        agents = get_shadow_agents()

        return {
            "status": "operational",
            "version": __version__,
            "engine": "FORGE - Cosmos Orchestrated Rapid Generation Engine",
            "capabilities": {
                "multi_model_deliberation": True,
                "propose_critique_refine_vote": True,
                "seven_layer_memory": True,
                "provider_fallback_chains": True,
                "cost_tracking": True,
                "rollback_support": True,
                "wave_parallel_execution": True,
            },
            "available_agents": agents if agents else [
                "grok", "gemini", "claude", "deepseek", "kimi", "phi",
                "huggingface", "swarm_mind", "qwen_coder"
            ],
            "model_task_map": {
                "planning": ["claude", "grok", "gemini"],
                "coding": ["claude", "deepseek", "qwen_coder"],
                "review": ["gemini", "grok", "claude"],
                "research": ["grok", "gemini", "kimi"],
                "debugging": ["deepseek", "claude", "phi"],
                "testing": ["phi", "deepseek", "claude"],
            },
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"FORGE status error: {e}")
        return {"status": "error", "error": str(e)}


@router.post("/quick")
async def forge_quick(req: QuickRequest):
    """
    Quick-mode: plan a task with swarm deliberation.
    Uses fewer models for speed, skips full research and verification.
    """
    try:
        engine = _get_engine(req.workspace)
        result = await engine.quick(req.task)
        return {
            "status": "success",
            "task": req.task,
            "result": result,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"FORGE quick error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/research")
async def forge_research(req: ResearchRequest):
    """
    Multi-model parallel research on a topic.
    4 agents investigate stack, architecture, pitfalls, and features simultaneously.
    """
    try:
        engine = _get_engine(req.workspace)
        findings = await engine.research(req.topic)
        return {
            "status": "success",
            "topic": req.topic,
            "findings": findings,
            "agents_used": ["grok", "gemini", "kimi", "claude"],
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"FORGE research error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/deliberate")
async def forge_deliberate(req: DeliberateRequest):
    """
    Full PROPOSE-CRITIQUE-REFINE-VOTE deliberation on an objective.
    Multiple models propose, critique, refine, and vote on the plan.
    Returns consensus score and final plan.
    """
    try:
        engine = _get_engine(req.workspace)
        result = await engine.deliberate(
            req.objective,
            context=req.context,
            models=req.models,
        )
        return {
            "status": "success",
            "objective": req.objective,
            "deliberation": result,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"FORGE deliberate error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/project")
async def forge_init_project(req: ProjectInitRequest):
    """Initialize a new FORGE project."""
    try:
        engine = _get_engine(req.workspace)
        project = engine.state.init_project(req.name, req.description)
        return {
            "status": "success",
            "project": {
                "name": project.name,
                "description": project.description,
                "milestone": project.milestone,
                "created_at": project.created_at,
            },
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"FORGE project init error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/phase")
async def forge_add_phase(req: AddPhaseRequest):
    """Add a phase to the current project."""
    try:
        engine = _get_engine(req.workspace)
        if not engine.state.project:
            engine.state.load_project()
        if not engine.state.project:
            raise HTTPException(status_code=404, detail="No project initialized. Use POST /api/forge/project first.")

        phase = engine.state.add_phase(
            req.name, req.description, req.goals, req.success_criteria
        )
        return {
            "status": "success",
            "phase": {
                "id": phase.id,
                "name": phase.name,
                "goals": phase.goals,
                "status": phase.status,
            },
            "timestamp": datetime.now().isoformat(),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"FORGE add phase error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/progress")
async def forge_progress():
    """Get project progress summary."""
    try:
        engine = _get_engine()
        progress = engine.get_progress()
        return {
            "status": "success",
            "progress": progress,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"FORGE progress error: {e}")
        return {"status": "error", "error": str(e)}


@router.get("/cost")
async def forge_cost():
    """Get cost breakdown by phase and model."""
    try:
        engine = _get_engine()
        report = engine.get_cost_report()
        return {
            "status": "success",
            "cost_report": report,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"FORGE cost error: {e}")
        return {"status": "error", "error": str(e)}
