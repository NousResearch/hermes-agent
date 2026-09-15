"""Per-route event coalescing for the webhook adapter (opt-in ``coalesce`` route block).

Rapid distinct events on one logical entity (five pushes to a PR, a burst of ticket edits) each
carry a fresh delivery ID, so idempotency cannot suppress them and every event wakes an agent
run. Coalescing groups events by a payload-derived key and debounces them: only the LATEST event
of a group is dispatched once the quiet window passes, bounded by ``max_wait_seconds`` past the
group's first event so a steady stream cannot starve dispatch (one durable run per entity, stale
heads superseded). #92066
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

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


@dataclass
class PendingEvent:
    """Newest event of a coalesce group plus the group's bookkeeping."""
    payload: Any
    prompt: str
    delivery_id: str
    first_at: float
    count: int
    dispatch_kwargs: Dict[str, Any]

    def prompt_with_note(self) -> str:
        if self.count <= 1:
            return self.prompt
        return (f"{self.prompt}\n\n(Note: {self.count} webhook events for this item arrived in quick succession "
                f"and were coalesced — this is the most recent one; earlier events are superseded.)")


class WebhookCoalescer:
    """Debounce-and-supersede buffer. ``dispatch(payload, prompt, delivery_id, **kwargs)`` is the adapter's
    agent-run spawner. Only ever driven from the aiohttp event loop → no locking."""

    def __init__(self, dispatch: Callable[..., Any], render: Callable[[str, dict, str, str], str]):
        self._dispatch = dispatch
        self._render = render
        self._pending: Dict[str, PendingEvent] = {}
        self._timers: Dict[str, asyncio.Task] = {}

    @property
    def pending(self) -> Dict[str, PendingEvent]:
        return self._pending

    def group_key(self, route_name: str, key_template: str, payload: dict, event_type: str) -> Optional[str]:
        """``"{route}|{rendered key}"`` for a bare dotted field or a brace template; ``None`` when a field did
        not resolve — unrelated entities must not collapse into one shared group (review finding on #92066)."""
        template = key_template.strip()
        if "{" not in template:
            template = "{" + template + "}"
        rendered = self._render(template, payload, event_type, route_name)
        if _PLACEHOLDER_RE.search(rendered):
            return None
        return f"{route_name}|{rendered}"

    def enqueue(self, *, route_name: str, coalesce: dict, payload: dict, event_type: str, prompt: str,
                delivery_id: str, now: float, **dispatch_kwargs) -> bool:
        """Buffer one event; False when the key did not resolve and the caller must dispatch immediately."""
        group_key = self.group_key(route_name, coalesce["key"], payload, event_type)
        if group_key is None:
            logger.info("[webhook] coalesce key %r unresolved for delivery %s on route %s — dispatching immediately",
                        coalesce["key"], delivery_id, route_name)
            return False
        window = float(coalesce.get("window_seconds", DEFAULT_WINDOW_SECONDS))
        max_wait = float(coalesce.get("max_wait_seconds", DEFAULT_MAX_WAIT_SECONDS))
        existing = self._pending.get(group_key)
        first_at = existing.first_at if existing else now
        count = existing.count + 1 if existing else 1
        dispatch_kwargs = {**dispatch_kwargs, "route_name": route_name, "event_type": event_type}
        self._pending[group_key] = PendingEvent(payload=payload, prompt=prompt, delivery_id=delivery_id,
                                                first_at=first_at, count=count, dispatch_kwargs=dispatch_kwargs)
        if existing is not None:
            logger.info("[webhook] coalesced delivery %s superseded by %s (group=%s, %d events)",
                        existing.delivery_id, delivery_id, group_key, count)
        old = self._timers.pop(group_key, None)
        if old is not None and not old.done():
            old.cancel()
        # Quiet window from now, capped at max_wait past the group's FIRST event.
        delay = min(window, max(0.0, first_at + max_wait - now))
        self._timers[group_key] = asyncio.create_task(self._timer(group_key, delay))
        return True

    async def _timer(self, group_key: str, delay: float) -> None:
        try:
            if delay > 0:
                await asyncio.sleep(delay)
        except asyncio.CancelledError:
            return  # superseded by a newer event's timer — the pending entry stays
        self._timers.pop(group_key, None)
        entry = self._pending.pop(group_key, None)
        if entry is not None:
            self._settle(group_key, entry)

    def _settle(self, group_key: str, entry: PendingEvent) -> Any:
        logger.info("[webhook] coalesce settled group=%s events=%d delivery=%s", group_key, entry.count,
                    entry.delivery_id)
        return self._dispatch(entry.payload, entry.prompt_with_note(), entry.delivery_id, time.time(),
                              **entry.dispatch_kwargs)

    async def flush(self) -> None:
        """Dispatch every pending group now and wait until each run is handed to the runner, so an adapter
        disconnect/reconnect drops nothing. A hard process kill still loses the current window's buffer."""
        for task in self._timers.values():
            if not task.done():
                task.cancel()
        self._timers.clear()
        pending, self._pending = self._pending, {}
        handed_off = []
        for group_key, entry in pending.items():
            try:
                handed_off.append(self._settle(group_key, entry))
            except Exception:
                logger.exception("[webhook] failed to flush coalesced group %s", group_key)
        awaitables = [t for t in handed_off if isinstance(t, asyncio.Future)]
        if awaitables:
            await asyncio.gather(*awaitables, return_exceptions=True)
