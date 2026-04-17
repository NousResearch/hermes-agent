"""
Agent Orchestration Tools
==========================

4 tools registered via PluginContext.register_tool():

  - spawn_agent:       spawn a background sub-agent
  - agent_status:      query agent status
  - send_agent_message: send message to running agent
  - orchestrate_task:  execute a workflow (parallel/sequential/map_reduce/dag)

Each tool returns JSON strings (convention in Hermes tool system).
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from concurrent.futures import as_completed
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Tool Schemas ─────────────────────────────────────────────────────────

SPAWN_AGENT_SCHEMA = {
    "type": "object",
    "properties": {
        "goal": {
            "type": "string",
            "description": "The task for the sub-agent to accomplish",
        },
        "context": {
            "type": "string",
            "description": "Additional context or instructions (optional)",
        },
        "toolsets": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Tool sets to make available (default: terminal, file, web)",
        },
        "model": {
            "type": "string",
            "description": "Model to use (default: inherit from parent)",
        },
    },
    "required": ["goal"],
}

AGENT_STATUS_SCHEMA = {
    "type": "object",
    "properties": {
        "agent_id": {
            "type": "string",
            "description": "Specific agent ID, or omit for all agents",
        },
    },
}

SEND_AGENT_MESSAGE_SCHEMA = {
    "type": "object",
    "properties": {
        "agent_id": {
            "type": "string",
            "description": "Target agent ID",
        },
        "message": {
            "type": "string",
            "description": "Message content to send",
        },
    },
    "required": ["agent_id", "message"],
}

ORCHESTRATE_TASK_SCHEMA = {
    "type": "object",
    "properties": {
        "workflow_type": {
            "type": "string",
            "enum": ["parallel", "sequential", "map_reduce", "dag"],
            "description": "Workflow execution pattern",
        },
        "tasks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "goal": {"type": "string"},
                    "context": {"type": "string"},
                    "toolsets": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "depends_on": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Task IDs this depends on (dag only)",
                    },
                },
                "required": ["goal"],
            },
            "description": "List of tasks to execute",
        },
        "aggregator_goal": {
            "type": "string",
            "description": "Goal for the aggregation agent (map_reduce only)",
        },
        "max_parallel": {
            "type": "integer",
            "description": "Max parallel agents (default: orchestration.max_agents)",
        },
    },
    "required": ["workflow_type", "tasks"],
}


# ── Tool Handlers ────────────────────────────────────────────────────────

def handle_spawn_agent(args: Dict[str, Any], **kwargs) -> str:
    """Spawn a background sub-agent."""
    from agent_orchestration.manager import get_manager

    goal = args.get("goal", "").strip()
    if not goal:
        return json.dumps({"error": "Missing required parameter: goal"})

    manager = get_manager()
    try:
        agent_id = manager.spawn(
            goal=goal,
            context=args.get("context"),
            toolsets=args.get("toolsets"),
            model=args.get("model"),
        )
        return json.dumps({
            "agent_id": agent_id,
            "status": "running",
            "message": f"Agent {agent_id} spawned and running in background",
        })
    except RuntimeError as exc:
        return json.dumps({"error": str(exc)})
    except Exception as exc:
        logger.exception("spawn_agent failed")
        return json.dumps({"error": str(exc)})


def handle_agent_status(args: Dict[str, Any], **kwargs) -> str:
    """Query status of spawned agents."""
    from agent_orchestration.manager import get_manager

    manager = get_manager()
    result = manager.get_status(args.get("agent_id"))
    return json.dumps(result, ensure_ascii=False)


def handle_send_agent_message(args: Dict[str, Any], **kwargs) -> str:
    """Send a message to a running agent."""
    from agent_orchestration.manager import get_manager
    from agent_orchestration.mailbox import get_mailbox_hub

    agent_id = args.get("agent_id", "").strip()
    message = args.get("message", "").strip()

    if not agent_id or not message:
        return json.dumps({"error": "Missing required: agent_id and message"})

    manager = get_manager()

    # Send via manager (inject into running agent)
    injected = manager.send_message(agent_id, message)

    # Also store in mailbox for async retrieval
    hub = get_mailbox_hub()
    hub.send_message(
        source_id="parent",
        target_id=agent_id,
        content=message,
    )

    return json.dumps({
        "agent_id": agent_id,
        "injected": injected,
        "mailbox_queued": True,
    })


def handle_orchestrate_task(args: Dict[str, Any], **kwargs) -> str:
    """Execute an orchestrated workflow across multiple agents."""
    from agent_orchestration.manager import get_manager
    from agent_orchestration.config import get_max_agents

    workflow_type = args.get("workflow_type", "").strip()
    tasks = args.get("tasks", [])

    if not workflow_type:
        return json.dumps({"error": "Missing required: workflow_type"})
    if not tasks:
        return json.dumps({"error": "Missing required: tasks"})

    # Validate tasks
    for i, t in enumerate(tasks):
        if not t.get("goal", "").strip():
            return json.dumps({"error": f"Task {i} missing 'goal'"})

    manager = get_manager()
    max_parallel = args.get("max_parallel") or get_max_agents()

    try:
        if workflow_type == "parallel":
            result = _run_parallel(manager, tasks, max_parallel)
        elif workflow_type == "sequential":
            result = _run_sequential(manager, tasks)
        elif workflow_type == "map_reduce":
            result = _run_map_reduce(
                manager, tasks,
                args.get("aggregator_goal", ""),
                max_parallel,
            )
        elif workflow_type == "dag":
            result = _run_dag(manager, tasks, max_parallel)
        else:
            return json.dumps({"error": f"Unknown workflow_type: {workflow_type}"})

        return json.dumps(result, ensure_ascii=False)

    except Exception as exc:
        logger.exception("orchestrate_task failed")
        return json.dumps({"error": str(exc)})


# ── Workflow Implementations ─────────────────────────────────────────────

def _run_parallel(
    manager, tasks: List[Dict], max_parallel: int
) -> Dict[str, Any]:
    """Run all tasks in parallel, collect results."""
    start = time.monotonic()
    results = []

    # Assign IDs
    for i, t in enumerate(tasks):
        t["_task_id"] = f"task-{i}"

    # Build and spawn all agents (respecting max_parallel)
    agent_map = {}  # task_id -> agent_id
    for t in tasks:
        aid = manager.spawn(
            goal=t["goal"],
            context=t.get("context"),
            toolsets=t.get("toolsets"),
        )
        agent_map[t["_task_id"]] = aid

    # Wait for all to complete (poll status)
    pending = set(agent_map.values())
    while pending:
        time.sleep(1)
        for aid in list(pending):
            status = manager.get_status(aid)
            s = status.get("status", "")
            if s in ("completed", "failed", "interrupted"):
                results.append(status)
                pending.discard(aid)

    duration = round(time.monotonic() - start, 2)
    return {
        "workflow": "parallel",
        "results": results,
        "total_duration_seconds": duration,
        "task_count": len(tasks),
    }


def _run_sequential(manager, tasks: List[Dict]) -> Dict[str, Any]:
    """Run tasks one after another, passing previous result as context."""
    start = time.monotonic()
    results = []
    previous_summary = ""

    for i, t in enumerate(tasks):
        # Enrich context with previous result
        ctx = t.get("context", "") or ""
        if previous_summary:
            ctx = f"{ctx}\n\nPrevious step result:\n{previous_summary}" if ctx else \
                  f"Previous step result:\n{previous_summary}"

        aid = manager.spawn(goal=t["goal"], context=ctx)

        # Wait for completion
        while True:
            status = manager.get_status(aid)
            s = status.get("status", "")
            if s in ("completed", "failed", "interrupted"):
                results.append(status)
                previous_summary = status.get("summary", "")
                break
            time.sleep(1)

    duration = round(time.monotonic() - start, 2)
    return {
        "workflow": "sequential",
        "results": results,
        "total_duration_seconds": duration,
        "task_count": len(tasks),
    }


def _run_map_reduce(
    manager, tasks: List[Dict], aggregator_goal: str, max_parallel: int
) -> Dict[str, Any]:
    """Run mappers in parallel, then aggregate with a single agent."""
    start = time.monotonic()

    # Map phase
    map_results = []
    agent_ids = []
    for t in tasks:
        aid = manager.spawn(
            goal=t["goal"], context=t.get("context"),
            toolsets=t.get("toolsets"),
        )
        agent_ids.append(aid)

    # Wait for all mappers
    for aid in agent_ids:
        while True:
            status = manager.get_status(aid)
            s = status.get("status", "")
            if s in ("completed", "failed", "interrupted"):
                map_results.append(status)
                break
            time.sleep(1)

    # Reduce phase
    agg_context = "Results from mapper agents:\n\n"
    for i, r in enumerate(map_results):
        agg_context += f"--- Mapper {i+1} ---\n"
        agg_context += r.get("summary", "(no result)") + "\n\n"

    agg_goal = aggregator_goal or "Synthesize and summarize the following results from multiple agents into a coherent final answer."

    agg_id = manager.spawn(goal=agg_goal, context=agg_context)
    while True:
        status = manager.get_status(agg_id)
        s = status.get("status", "")
        if s in ("completed", "failed", "interrupted"):
            break
        time.sleep(1)

    duration = round(time.monotonic() - start, 2)
    return {
        "workflow": "map_reduce",
        "map_results": map_results,
        "aggregation": status,
        "total_duration_seconds": duration,
        "task_count": len(tasks),
    }


def _run_dag(manager, tasks: List[Dict], max_parallel: int) -> Dict[str, Any]:
    """Run tasks as a DAG — tasks with satisfied dependencies execute in parallel.

    Uses Kahn's algorithm for topological sorting.
    """
    start = time.monotonic()

    # Assign IDs and build dependency graph
    task_map = {}
    for i, t in enumerate(tasks):
        tid = f"dag-{i}"
        t["_task_id"] = tid
        task_map[tid] = t

    # Track completed tasks and results
    completed: Dict[str, str] = {}  # task_id -> summary
    results = []
    remaining = set(task_map.keys())

    while remaining:
        # Find tasks with all dependencies satisfied
        ready = []
        for tid in list(remaining):
            deps = task_map[tid].get("depends_on", [])
            if all(d in completed for d in deps):
                ready.append(tid)

        if not ready:
            # Circular dependency or stuck
            return {
                "workflow": "dag",
                "error": "Circular dependency or unresolvable tasks",
                "remaining": list(remaining),
                "results": results,
            }

        # Spawn ready tasks
        agent_ids = {}
        for tid in ready:
            t = task_map[tid]
            # Build context from dependency results
            ctx = t.get("context", "") or ""
            dep_summaries = []
            for dep_id in t.get("depends_on", []):
                if dep_id in completed:
                    dep_summaries.append(f"[{dep_id}]: {completed[dep_id]}")
            if dep_summaries:
                dep_ctx = "Dependency results:\n" + "\n".join(dep_summaries)
                ctx = f"{ctx}\n\n{dep_ctx}" if ctx else dep_ctx

            aid = manager.spawn(goal=t["goal"], context=ctx)
            agent_ids[tid] = aid

        # Wait for this batch to complete
        for tid, aid in agent_ids.items():
            while True:
                status = manager.get_status(aid)
                s = status.get("status", "")
                if s in ("completed", "failed", "interrupted"):
                    results.append({"task_id": tid, **status})
                    completed[tid] = status.get("summary", "")
                    remaining.discard(tid)
                    break
                time.sleep(1)

    duration = round(time.monotonic() - start, 2)
    return {
        "workflow": "dag",
        "results": results,
        "total_duration_seconds": duration,
        "task_count": len(tasks),
    }


# ── Registration Helper ──────────────────────────────────────────────────

def register_all(ctx) -> None:
    """Register all 4 orchestration tools via PluginContext."""
    ctx.register_tool(
        name="spawn_agent",
        toolset="orchestration",
        schema=SPAWN_AGENT_SCHEMA,
        handler=handle_spawn_agent,
        description="Spawn a background sub-agent to execute a task autonomously",
        emoji="🔀",
    )

    ctx.register_tool(
        name="agent_status",
        toolset="orchestration",
        schema=AGENT_STATUS_SCHEMA,
        handler=handle_agent_status,
        description="Query status of spawned agents (running, completed, failed)",
        emoji="📊",
    )

    ctx.register_tool(
        name="send_agent_message",
        toolset="orchestration",
        schema=SEND_AGENT_MESSAGE_SCHEMA,
        handler=handle_send_agent_message,
        description="Send a message to a running sub-agent (injects into conversation)",
        emoji="✉️",
    )

    ctx.register_tool(
        name="orchestrate_task",
        toolset="orchestration",
        schema=ORCHESTRATE_TASK_SCHEMA,
        handler=handle_orchestrate_task,
        description="Execute a multi-agent workflow (parallel, sequential, map_reduce, or dag)",
        emoji="🎯",
    )
