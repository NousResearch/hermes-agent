"""A background run does not continue on a LOCAL model after a billing/credits refusal.

When a scheduled job's provider is out of credits (xAI's 403 ``spending-limit``, a 402), a local
fallback (LM Studio, Ollama, llama.cpp or vLLM on loopback or the LAN) is often the one chain
entry that still answers, so every fire ran the whole job on it: on a small home server that is
hours of grinding for work nobody is waiting on. Declining the switch ends the run on the billing
wall, where its owner holds it instead: cron parks the job (``cron/billing_hold.py``) and the
Kanban dispatcher spaces the task's retries (the worker exits with the rate-limit code). A cloud
fallback still takes over, other failure reasons still reach a local entry, and interactive
sessions keep the whole chain. ``fallback.background_local_when_billing_blocked: true`` restores
the local switch for background runs.
"""

from __future__ import annotations

import os
from typing import Any
from urllib.parse import urlparse

from agent.error_classifier import FailoverReason

CONFIG_KEY = "background_local_when_billing_blocked"


def _delegation_root(agent: Any) -> Any:
    seen: set[int] = set()
    while id(agent) not in seen:
        seen.add(id(agent))
        ref = getattr(agent, "_delegate_parent_ref", None)
        parent = ref() if callable(ref) else None
        if parent is None:
            break
        agent = parent
    return agent


def is_background_run(agent: Any) -> bool:
    """Nobody is waiting on this turn: a cron run (its delegate children follow their root), or a
    process the Kanban dispatcher spawned for a task (``HERMES_KANBAN_TASK``, the process identity
    ``hermes_cli.cli_single_query._single_query_exit_code`` maps provider walls by)."""
    if getattr(_delegation_root(agent), "platform", None) == "cron":
        return True
    return bool(os.environ.get("HERMES_KANBAN_TASK"))


def is_local_model_endpoint(base_url: Any) -> bool:
    """An HTTP endpoint on loopback, the LAN, a container host or a Tailscale peer. Non-HTTP
    routes (``moa://``, ``acp://``) are never local servers."""
    url = str(base_url or "")
    if urlparse(url).scheme not in ("http", "https"):
        return False
    from agent.model_metadata import is_local_endpoint

    return is_local_endpoint(url)


def _local_allowed_by_config() -> bool:
    try:
        from hermes_cli.config import load_config

        section = (load_config() or {}).get("fallback")
    except Exception:
        return False
    return isinstance(section, dict) and section.get(CONFIG_KEY) is True


def declines_local_fallback(agent: Any, reason: "FailoverReason | None", fb_base_url: Any) -> bool:
    """True when this switch is a background run leaving a billing-refused provider for a local
    model, and the operator has not opted back in."""
    return (
        reason == FailoverReason.billing
        and is_local_model_endpoint(fb_base_url)
        and is_background_run(agent)
        and not _local_allowed_by_config()
    )
