"""Platform-neutral lifecycle observation for one Gateway turn.

The core emits a deliberately small, closed event vocabulary to the selected
platform adapter. Adapters may translate those events at the edge, but prompts,
tool arguments/results, exception text, protocol payloads, and transport state
never enter this contract.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Literal, Optional

logger = logging.getLogger(__name__)

_SAFE_TOOL_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_.:-]{0,63}")
_TOOL_ID_MAP_CAP = 256

TurnPhase = Literal[
    "turn_started",
    "session_resolved",
    "turn_liveness",
    "tool_started",
    "tool_finished",
    "turn_finished",
]
TurnOutcome = Literal["success", "failed", "cancelled", "timed_out"]
ToolStatus = Literal["executing", "completed", "failed"]


@dataclass(frozen=True)
class TurnLifecycleEvent:
    """Allowlisted metadata for one platform-neutral Gateway lifecycle event."""

    phase: TurnPhase
    platform: str
    profile: str
    channel_id: str
    session_id: str
    turn_id: str
    started_at: str
    triggering_event_id: Optional[str] = None
    is_new_session: Optional[bool] = None
    tool_call_id: Optional[str] = None
    tool_name: Optional[str] = None
    tool_status: Optional[ToolStatus] = None
    outcome: Optional[TurnOutcome] = None


def classify_turn_outcome(
    result: Any,
    *,
    timed_out: bool = False,
    exception_type: Optional[type[BaseException]] = None,
) -> TurnOutcome:
    """Classify terminal state without exposing result or exception content."""

    if timed_out:
        return "timed_out"
    if exception_type is not None and issubclass(
        exception_type, asyncio.CancelledError
    ):
        return "cancelled"
    if exception_type is not None or not isinstance(result, dict):
        return "failed"
    if bool(result.get("interrupted")):
        return "cancelled"
    if bool(result.get("failed")) or bool(result.get("error")):
        return "failed"
    return "success"


class GatewayTurnObserver:
    """Fail-open lifecycle dispatcher scoped to one logical Gateway turn.

    ``route`` is opaque dispatch context (normally the selected platform
    adapter). It is passed separately from :class:`TurnLifecycleEvent` so the
    event remains serializable, platform-neutral, and privacy-auditable.
    """

    def __init__(
        self,
        *,
        platform: str,
        profile: str,
        channel_id: str,
        session_id: str,
        triggering_event_id: Optional[str],
        is_new_session: bool,
        route: Any,
        loop: asyncio.AbstractEventLoop,
        is_current: Callable[[], bool],
    ) -> None:
        self.platform = str(platform or "")
        self.profile = str(profile or "")
        self.channel_id = str(channel_id or "")
        self.session_id = str(session_id or "")
        self.triggering_event_id = (
            str(triggering_event_id) if triggering_event_id is not None else None
        )
        self.is_new_session = bool(is_new_session)
        self.route = route
        self.loop = loop
        self.is_current = is_current
        self.turn_id = uuid.uuid4().hex
        self.started_at = (
            datetime
            .now(timezone.utc)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z")
        )
        self._active = False
        self._started = False
        self._terminal_dispatched = False
        self._liveness_task: Optional[asyncio.Task] = None
        self._tool_id_counter = 0
        self._tool_ids: OrderedDict[bytes, str] = OrderedDict()

    @property
    def active(self) -> bool:
        return self._active and not self._terminal_dispatched

    def _event(self, phase: TurnPhase, **kwargs: Any) -> TurnLifecycleEvent:
        return TurnLifecycleEvent(
            phase=phase,
            platform=self.platform,
            profile=self.profile,
            channel_id=self.channel_id,
            session_id=self.session_id,
            turn_id=self.turn_id,
            started_at=self.started_at,
            **kwargs,
        )

    def _dispatch(self, event: TurnLifecycleEvent) -> bool:
        """Synchronously offer an event to the selected adapter, fail-open."""

        try:
            handler = getattr(self.route, "on_turn_lifecycle", None)
            if not callable(handler):
                return False
            return bool(handler(event))
        except Exception:
            logger.debug("Gateway turn observer failed open", exc_info=True)
            return False

    def start(self, *, liveness_interval: float = 10.0) -> bool:
        """Dispatch turn start and immediately arm liveness when consumed."""

        if self._started:
            return self.active
        self._started = True
        self._active = self._dispatch(
            self._event(
                "turn_started",
                triggering_event_id=self.triggering_event_id,
            )
        )
        if self._active and liveness_interval > 0:
            self._liveness_task = self.loop.create_task(
                self._liveness_loop(liveness_interval)
            )
        return self._active

    def session_resolved(self) -> bool:
        if not self.active:
            return False
        return self._dispatch(
            self._event("session_resolved", is_new_session=self.is_new_session)
        )

    async def _liveness_loop(self, interval: float) -> None:
        try:
            while self.active:
                await asyncio.sleep(interval)
                if not self.active or not self.is_current():
                    return
                self._dispatch(self._event("turn_liveness"))
        except asyncio.CancelledError:
            return
        except Exception:
            logger.debug("Gateway turn liveness observer failed open", exc_info=True)

    def _dispatch_if_active(self, event: TurnLifecycleEvent) -> None:
        if self.active and self.is_current():
            self._dispatch(event)

    def _dispatch_threadsafe(self, event: TurnLifecycleEvent) -> None:
        """Preserve worker-thread tool callbacks without leaking tool content."""

        if not self.active or not self.is_current():
            return
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None
        if running_loop is self.loop:
            self._dispatch_if_active(event)
            return
        try:
            self.loop.call_soon_threadsafe(self._dispatch_if_active, event)
        except Exception:
            logger.debug("Gateway turn observer scheduling failed open", exc_info=True)

    def _safe_tool_call_id(self, value: Any) -> str:
        """Return a bounded turn-local correlation id without retaining input."""

        raw = str(value or "")[:4096].encode("utf-8", errors="replace")
        digest = hashlib.blake2s(raw, digest_size=16).digest()
        existing = self._tool_ids.get(digest)
        if existing is not None:
            self._tool_ids.move_to_end(digest)
            return existing
        self._tool_id_counter += 1
        safe_id = f"tool-{self._tool_id_counter}"
        self._tool_ids[digest] = safe_id
        while len(self._tool_ids) > _TOOL_ID_MAP_CAP:
            self._tool_ids.popitem(last=False)
        return safe_id

    @staticmethod
    def _safe_tool_name(value: Any) -> str:
        """Allow only compact display labels; arbitrary values become ``tool``."""

        text = str(value or "")
        return text if _SAFE_TOOL_NAME.fullmatch(text) else "tool"

    def tool_started(self, call_id: Any, tool_name: Any, _args: Any = None) -> None:
        if not self.active or not self.is_current():
            return
        self._dispatch_threadsafe(
            self._event(
                "tool_started",
                tool_call_id=self._safe_tool_call_id(call_id),
                tool_name=self._safe_tool_name(tool_name),
                tool_status="executing",
            )
        )

    def tool_finished(
        self,
        call_id: Any,
        tool_name: Any,
        _args: Any = None,
        result: Any = None,
    ) -> None:
        if not self.active or not self.is_current():
            return
        try:
            from agent.display import _detect_tool_failure

            failed, _ = _detect_tool_failure(str(tool_name), result)
        except Exception:
            failed = False
        self._dispatch_threadsafe(
            self._event(
                "tool_finished",
                tool_call_id=self._safe_tool_call_id(call_id),
                tool_name=self._safe_tool_name(tool_name),
                tool_status="failed" if failed else "completed",
            )
        )

    def finish(
        self,
        result: Any,
        *,
        timed_out: bool = False,
        exception_type: Optional[type[BaseException]] = None,
    ) -> bool:
        """Dispatch exactly one local terminal event and stop liveness."""

        if self._terminal_dispatched or not self._started:
            return False
        self._terminal_dispatched = True
        self._active = False
        task = self._liveness_task
        self._liveness_task = None
        if task is not None and not task.done():
            task.cancel()
        return self._dispatch(
            self._event(
                "turn_finished",
                outcome=classify_turn_outcome(
                    result,
                    timed_out=timed_out,
                    exception_type=exception_type,
                ),
            )
        )
