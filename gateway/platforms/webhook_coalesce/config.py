"""Per-route event coalescing for the webhook adapter (opt-in ``coalesce`` route block).

Rapid distinct events on one logical entity (five pushes to a PR, a burst of ticket edits) each
carry a fresh delivery ID, so idempotency cannot suppress them and every event wakes an agent
run. Coalescing groups events by a payload-derived key and debounces them: only the LATEST event
of a group is dispatched once the quiet window passes, bounded by ``max_wait_seconds`` past the
group's first event so a steady stream cannot starve dispatch (one durable run per entity, stale
heads superseded). #92066
"""

from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_WINDOW_SECONDS = 30.0
DEFAULT_MAX_WAIT_SECONDS = 300.0
_PLACEHOLDER_RE = re.compile(r"\{[a-zA-Z0-9_.]+\}")


def validate_coalesce_config(route_name: str, route: dict) -> None:
    """Startup validation of a route's ``coalesce`` block; raises ValueError so a typo'd route fails
    fast instead of silently dispatching every event. Agent-mode routes only."""
    coalesce = route.get("coalesce")
    if coalesce is None:
        return
    for exclusive in ("deliver_only", "cron_job"):
        if route.get(exclusive):
            raise ValueError(f"[webhook] Route '{route_name}' combines {exclusive} with coalesce. Coalescing only "
                             f"applies to agent-mode routes.")
    key = coalesce.get("key") if isinstance(coalesce, dict) else None
    if not isinstance(key, str) or not key.strip():
        raise ValueError(f"[webhook] Route '{route_name}' has a coalesce block without a non-empty string 'key'. "
                         f"Set coalesce.key to a payload field (e.g. 'pull_request.number') or a template "
                         f"(e.g. '{{repository.full_name}}#{{number}}').")
    for field in ("window_seconds", "max_wait_seconds"):
        value = coalesce.get(field)
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0):
            raise ValueError(f"[webhook] Route '{route_name}' coalesce.{field} must be a positive number, got {value!r}.")


def prompt_with_note(prompt: str, count: int) -> str:
    if count <= 1:
        return prompt
    return (f"{prompt}\n\n(Note: {count} webhook events for this item arrived in quick succession "
            f"and were coalesced — this is the most recent one; earlier events are superseded.)")


def group_key( route_name: str, key_template: str, payload: dict, event_type: str, render) -> Optional[str]:
    """``"{route}|{rendered key}"`` for a bare dotted field or a brace template; ``None`` when a field did
    not resolve — unrelated entities must not collapse into one shared group (review finding on #92066)."""
    template = key_template.strip()
    if "{" not in template:
        template = "{" + template + "}"
    rendered = render(template, payload, event_type, route_name)
    if _PLACEHOLDER_RE.search(rendered):
        return None
    return f"{route_name}|{rendered}"
