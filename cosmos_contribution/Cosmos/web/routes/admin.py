"""
Admin Routes - Workers, Staging, Cognition, Heartbeat, DLQ Monitoring

Endpoints:
- GET /api/workers/status - Parallel worker system status
- POST /api/workers/init-tasks - Initialize development tasks
- POST /api/workers/start - Start parallel workers
- GET /api/staging/files - list staging files
- GET /api/evolution/status - Evolution loop status (separate from engine)
- GET /api/cognition/status - Cognitive system status
- GET /api/heartbeat - Swarm health vitals
- GET /api/heartbeat/history - Heartbeat history
- GET /api/metrics/dlq - Dead Letter Queue metrics
- GET /api/metrics/dlq/pending - Pending DLQ entries
- GET /api/metrics/dlq/recent - Recent DLQ entries
- POST /api/metrics/dlq/purge - Purge resolved DLQ entries
"""

import logging
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

router = APIRouter()


def _get_shared():
    """Import shared state from server module lazily."""
    from Cosmos.web import server
    return server


# ============================================
# PARALLEL WORKER API
# ============================================

@router.get("/api/workers/status")
async def get_workers_status():
    """Get parallel worker system status."""
    from Cosmos.core.agent_spawner import get_spawner
    spawner = get_spawner()
    return {
        "spawner": spawner.get_status(),
        "tasks": [
            {
                "id": t.task_id,
                "type": t.task_type.value,
                "agent": t.assigned_to,
                "status": t.status,
                "description": t.description
            }
            for t in spawner.task_queue
        ],
        "discoveries": spawner.shared_state.get("discoveries", [])[-10:],
        "proposals": len(spawner.shared_state.get("proposals", [])),
    }


@router.post("/api/workers/init-tasks")
async def init_tasks():
    """Initialize the 20 development tasks."""
    from Cosmos.core.agent_spawner import initialize_development_tasks
    status = initialize_development_tasks()
    return {"status": "initialized", "info": status}


@router.post("/api/workers/start")
async def start_workers():
    """Start the parallel worker system."""
    try:
        from Cosmos.core.parallel_workers import start_parallel_workers
        manager = await start_parallel_workers()
        return {"status": "started", "info": manager.get_status()}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.get("/api/staging/files")
async def get_staging_files():
    """list files in the staging directory."""
    import os
    staging_dir = Path("/workspace/Cosmos/cosmos/staging")
    files = []
    if staging_dir.exists():
        for root, dirs, filenames in os.walk(staging_dir):
            for f in filenames:
                path = Path(root) / f
                files.append({
                    "path": str(path.relative_to(staging_dir)),
                    "size": path.stat().st_size,
                    "modified": path.stat().st_mtime
                })
    return {"files": files[:50]}


# ============================================
# EVOLUTION LOOP STATUS
# ============================================

@router.get("/api/evolution/loop-status")
async def get_evolution_loop_status():
    """Get evolution loop status."""
    from Cosmos.core.agent_spawner import get_spawner
    from Cosmos.core.evolution_loop import get_evolution_loop
    loop = get_evolution_loop()
    spawner = get_spawner()
    return {
        "running": loop.running,
        "evolution_cycle": loop.evolution_cycle,
        "last_discussion": loop.last_discussion.isoformat() if loop.last_discussion else None,
        "spawner": spawner.get_status()
    }


# ============================================
# HUMAN-LIKE COGNITION SYSTEMS
# ============================================

@router.get("/api/cognition/status")
async def get_cognition_status():
    """Get cognitive system status."""
    try:
        from Cosmos.core.temporal_awareness import get_temporal_awareness
        from Cosmos.core.spontaneous_cognition import get_spontaneous_cognition
        from Cosmos.core.capability_registry import get_capability_registry

        temporal = get_temporal_awareness()
        cognition = get_spontaneous_cognition()
        registry = get_capability_registry()

        return {
            "temporal": temporal.get_status(),
            "emotional_state": cognition.get_emotional_state(),
            "recent_thoughts": [t.to_any() for t in cognition.get_recent_thoughts(5)],
            "capabilities_count": len(registry.capabilities),
            "available_capabilities": len(registry.get_available()),
        }
    except Exception as e:
        return {"error": str(e)}


# ============================================
# SWARM HEARTBEAT
# ============================================

@router.get("/api/heartbeat")
async def get_heartbeat_status():
    """Get current swarm health vitals."""
    try:
        from Cosmos.core.swarm_heartbeat import get_current_vitals
        vitals = await get_current_vitals()
        return vitals.to_any() if vitals else {"error": "No vitals available"}
    except Exception as e:
        return {"error": str(e)}


@router.get("/api/heartbeat/history")
async def get_heartbeat_history():
    """Get recent heartbeat history."""
    try:
        from Cosmos.core.swarm_heartbeat import get_heartbeat
        heartbeat = get_heartbeat()
        return {"history": [v.to_any() for v in heartbeat.health_history[-20:]]}
    except Exception as e:
        return {"error": str(e)}


# ============================================
# DEAD LETTER QUEUE (AGI v1.9.1)
# ============================================

@router.get("/api/metrics/dlq")
async def get_dlq_metrics():
    """Get Dead Letter Queue metrics - failed signal counts, retry rates, breakdowns."""
    try:
        from Cosmos.core.dlq import get_dlq
        dlq = get_dlq()
        return dlq.get_metrics()
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@router.get("/api/metrics/dlq/pending")
async def get_dlq_pending(limit: int = 50):
    """Get pending (unresolved) DLQ entries awaiting retry."""
    try:
        from Cosmos.core.dlq import get_dlq
        dlq = get_dlq()
        return {"pending": dlq.get_pending_entries(limit=min(limit, 200))}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@router.get("/api/metrics/dlq/recent")
async def get_dlq_recent(limit: int = 50):
    """Get recent DLQ entries (resolved and unresolved)."""
    try:
        from Cosmos.core.dlq import get_dlq
        dlq = get_dlq()
        return {"entries": dlq.get_recent_entries(limit=min(limit, 200))}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@router.post("/api/metrics/dlq/purge")
async def purge_dlq(older_than_hours: int = 24):
    """Purge resolved DLQ entries older than the given age."""
    try:
        from Cosmos.core.dlq import get_dlq
        dlq = get_dlq()
        removed = dlq.purge_resolved(older_than_hours=older_than_hours)
        return {"purged": removed, "remaining": len(dlq._entries)}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
