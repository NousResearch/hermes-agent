"""Wire Kynver plan-progress todo store and operating prompt blocks into Hermes."""

from __future__ import annotations

import logging
from typing import Any, List, Mapping, Optional

from agent.operating_prompt import register_operating_prompt_hook

from .agentos_bridge import KynverAgentOSClient
from .operating_config import load_operating_linkage
from .substrate import allow_local_fallback, substrate_active
from .todo_store import KynverTodoStore

logger = logging.getLogger(__name__)

_PROMPT_HOOK_REGISTERED = False


def _ensure_prompt_hook() -> None:
    global _PROMPT_HOOK_REGISTERED
    if _PROMPT_HOOK_REGISTERED:
        return
    register_operating_prompt_hook(get_prompt_blocks)
    _PROMPT_HOOK_REGISTERED = True


def configure_agent(
    agent: Any,
    agent_cfg: Mapping[str, Any],
    *,
    platform: Optional[str] = None,
) -> None:
    """Replace the default local todo store when Kynver substrate is active."""

    _ensure_prompt_hook()

    agent._kynver_active = False
    agent._kynver_degraded = False

    if not substrate_active(config=agent_cfg):
        return

    client = KynverAgentOSClient()
    linkage = load_operating_linkage()
    fallback_ok = allow_local_fallback(agent_cfg)

    agent._kynver_client = client
    agent._kynver_active = True
    agent._todo_store = KynverTodoStore(
        client,
        linkage=linkage,
        allow_fallback=fallback_ok,
    )
    agent._todo_store_provider = "kynver"
    agent._kynver_degraded = bool(getattr(agent._todo_store, "degraded", False))

    logger.info(
        "Kynver todo store active (plan_id=%s; in_progress uses progress-focus, not running)",
        linkage.plan_id or "(none)",
    )


def get_prompt_blocks(agent: Any) -> List[str]:
    blocks: List[str] = []
    if getattr(agent, "_kynver_degraded", False):
        blocks.append(
            "[Kynver: degraded mode — todo/current-focus may use local Hermes fallback]"
        )
    provider = getattr(agent, "_todo_store_provider", "local")
    if provider == "kynver" and getattr(agent, "_kynver_active", False):
        blocks.append(
            "[Kynver: session todos sync to AgentOS plan progress; "
            "in_progress is current focus, not harness running lease]"
        )
    return blocks
