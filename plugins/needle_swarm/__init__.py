"""plugins/needle_swarm/__init__.py - Hermes Plugin Integration for Needle Swarm.

Registers Hermes tools:
- needle_single_dispatch
- needle_swarm_fanout
- needle_growth_cycle
- needle_hindsight_recall

Registers Hermes slash command:
- /needle
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from hermes_constants import get_hermes_home
from .orchestrator import Orchestrator

logger = logging.getLogger(__name__)

_ORCHESTRATOR: Optional[Orchestrator] = None


def get_orchestrator(ctx: Optional[Any] = None) -> Orchestrator:
    global _ORCHESTRATOR
    if _ORCHESTRATOR is None:
        data_dir = get_hermes_home() / "needle_swarm"

        # Build LLM facade callback using Hermes ctx.llm if available
        def _brain_llm_callback(prompt: str) -> str:
            if ctx and hasattr(ctx, "llm") and ctx.llm:
                try:
                    res = ctx.llm.chat(prompt)
                    if isinstance(res, str):
                        return res
                    if isinstance(res, dict):
                        return res.get("content", str(res))
                except Exception as e:
                    logger.warning("ctx.llm call failed in needle_swarm: %s", e)
            return f"[Needle Swarm Brain Fallback] Handled request: '{prompt[:60]}...'"

        _ORCHESTRATOR = Orchestrator(data_dir=data_dir, brain_llm_callback=_brain_llm_callback)
    return _ORCHESTRATOR


# --- Tool Handlers ---

def _tool_needle_single_dispatch(utterance: str, mode_cluster: str = "default", task_id: str = "") -> str:
    orch = get_orchestrator()
    if mode_cluster:
        orch.enter_mode(mode_cluster, exclusive=False)
    res = orch.handle_single(utterance)
    return json.dumps(res, indent=2)


def _tool_needle_swarm_fanout(query: str, mode_cluster: str = "security", task_id: str = "") -> str:
    orch = get_orchestrator()
    if mode_cluster:
        orch.enter_mode(mode_cluster, exclusive=False)
    swarm_res = orch.handle_swarm(query)
    return json.dumps({
        "query": query,
        "action_signals": swarm_res.action_signals,
        "memory_signals": swarm_res.memory_signals,
        "playbook_signals": swarm_res.playbook_signals,
        "bonsai_context": swarm_res.as_bonsai_context(),
    }, indent=2)


def _tool_needle_growth_cycle(task_id: str = "") -> str:
    orch = get_orchestrator()
    growth_summary = orch.run_growth_cycle()
    return json.dumps(growth_summary, indent=2)


def _tool_needle_hindsight_recall(query: str, top_k: int = 5, task_id: str = "") -> str:
    orch = get_orchestrator()
    recalled = orch.hindsight_store.recall(query, top_k=top_k)
    return json.dumps({"query": query, "recalled_observations": recalled}, indent=2)


# --- Slash Command Handler ---

def _handle_needle_command(raw_args: str) -> str:
    orch = get_orchestrator()
    args = (raw_args or "").strip().split(maxsplit=1)
    sub = args[0].lower() if args else "status"

    if sub == "status":
        loaded = orch.working_set.get_loaded_ids()
        mb = orch.working_set.current_mb()
        max_mb = orch.working_set.max_mb
        skills_count = len(orch.registry.list_skills())
        return (
            f"🌲 Needle Swarm Status:\n"
            f"  - Registered Skills: {skills_count}\n"
            f"  - Loaded Skills ({len(loaded)}): {', '.join(loaded) if loaded else 'none'}\n"
            f"  - Memory Budget: {mb:.1f} / {max_mb:.1f} MB"
        )
    elif sub == "growth":
        summary = orch.run_growth_cycle()
        return f"🌲 Needle Swarm Growth Cycle Completed:\n```json\n{json.dumps(summary, indent=2)}\n```"
    elif sub == "mode":
        cluster = args[1].strip() if len(args) > 1 else "default"
        loaded = orch.enter_mode(cluster)
        return f"🌲 Switched to cluster '{cluster}'. Loaded {len(loaded)} skills: {', '.join(loaded)}"

    return "Usage: /needle [status | growth | mode <cluster>]"


def register(ctx: Any) -> None:
    # Initialize singleton with Hermes ctx
    get_orchestrator(ctx)

    # Register Tools
    ctx.register_tool(
        name="needle_single_dispatch",
        toolset="needle_swarm",
        schema={
            "name": "needle_single_dispatch",
            "description": "Route a single user utterance to the best fast Needle micro-model skill, escalating to LLM if needed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "utterance": {"type": "string", "description": "The user utterance or task query."},
                    "mode_cluster": {"type": "string", "description": "Active skill cluster (e.g. coding, security)."},
                },
                "required": ["utterance"],
            },
        },
        handler=lambda args, **kw: _tool_needle_single_dispatch(
            utterance=args.get("utterance", ""),
            mode_cluster=args.get("mode_cluster", "default"),
            task_id=kw.get("task_id", ""),
        ),
    )

    ctx.register_tool(
        name="needle_swarm_fanout",
        toolset="needle_swarm",
        schema={
            "name": "needle_swarm_fanout",
            "description": "Parallel fan-out query across all loaded Needle specialist skills (action, memory, playbook).",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The event stream utterance or query."},
                    "mode_cluster": {"type": "string", "description": "Active swarm cluster."},
                },
                "required": ["query"],
            },
        },
        handler=lambda args, **kw: _tool_needle_swarm_fanout(
            query=args.get("query", ""),
            mode_cluster=args.get("mode_cluster", "security"),
            task_id=kw.get("task_id", ""),
        ),
    )

    ctx.register_tool(
        name="needle_growth_cycle",
        toolset="needle_swarm",
        schema={
            "name": "needle_growth_cycle",
            "description": "Trigger the out-of-band growth dreaming loop to retrain skills and propose new ones from escalations.",
            "parameters": {"type": "object", "properties": {}},
        },
        handler=lambda args, **kw: _tool_needle_growth_cycle(task_id=kw.get("task_id", "")),
    )

    ctx.register_tool(
        name="needle_hindsight_recall",
        toolset="needle_swarm",
        schema={
            "name": "needle_hindsight_recall",
            "description": "Recall consolidated facts and observations from the Hindsight RAG store.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query for memory recall."},
                    "top_k": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        },
        handler=lambda args, **kw: _tool_needle_hindsight_recall(
            query=args.get("query", ""),
            top_k=args.get("top_k", 5),
            task_id=kw.get("task_id", ""),
        ),
    )

    # Register Slash Command
    ctx.register_command(
        name="needle",
        handler=_handle_needle_command,
        description="Needle Swarm status, mode switches, and growth triggers",
        args_hint="[status | growth | mode <cluster>]",
    )
