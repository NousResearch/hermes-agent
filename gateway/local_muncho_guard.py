"""Visible-send guard helpers for the Local Muncho runtime."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from copy import copy
from dataclasses import is_dataclass, replace
from typing import Any, Mapping

from agent.local_muncho.policy import visible_send_action
from agent.local_muncho.runtime import get_current_runtime
from agent.local_muncho.types import VisibleSendDecision, VisibleSendIntent


class LocalMunchoVisibleSendBlocked(RuntimeError):
    pass


async def guard_visible_send(
    intent: VisibleSendIntent,
    *,
    runtime: Any = None,
) -> VisibleSendDecision:
    runtime = runtime or get_current_runtime()
    if not runtime.enabled():
        return VisibleSendDecision(True, reason="runtime disabled")

    assertion = runtime.assert_active_lease(
        action=visible_send_action(intent.kind),
        approval_class="visible_send",
    )
    if not assertion.allowed:
        return VisibleSendDecision(False, reason=assertion.reason)

    validated_kinds = {"final", "error", "degraded_status", "send_message"}
    if intent.text and intent.kind in validated_kinds:
        validation = runtime.validate_final_output(intent.text, evidence=())
        if not validation.allowed:
            recheck = runtime.assert_active_lease(action="visible_send:replacement")
            if recheck.allowed:
                return VisibleSendDecision(
                    True,
                    reason=validation.reason,
                    replacement_text=validation.replacement_text,
                )
            return VisibleSendDecision(False, reason=recheck.reason)

    return VisibleSendDecision(True, reason=assertion.reason)


async def send_with_muncho_guard(
    send_coro_factory: Callable[[], Awaitable[Any]],
    *,
    intent: VisibleSendIntent,
    runtime: Any = None,
) -> Any:
    decision = await guard_visible_send(intent, runtime=runtime)
    if not decision.allowed:
        raise LocalMunchoVisibleSendBlocked(decision.reason)
    return await send_coro_factory()


def _runtime_streaming_config(runtime: Any) -> Mapping[str, Any]:
    config = getattr(runtime, "config", {})
    if not isinstance(config, Mapping):
        return {}
    streaming = config.get("streaming")
    return streaming if isinstance(streaming, Mapping) else {}


def guard_stream_consumer_config(config: Any, *, runtime: Any = None) -> Any:
    """Return a guarded stream config copy without touching the consumer core.

    Disabled runtime returns the original object by identity.  Enabled runtime
    can clamp callers to buffer-only/edit delivery using muncho_runtime.streaming.
    """

    runtime = runtime or get_current_runtime()
    if not runtime.enabled():
        return config

    streaming = _runtime_streaming_config(runtime)
    if not streaming.get("buffer_only", False) and streaming.get(
        "drafts_enabled",
        True,
    ):
        return config

    updates: dict[str, Any] = {}
    if streaming.get("buffer_only", False):
        updates["buffer_only"] = True
    if not streaming.get("drafts_enabled", True):
        updates["transport"] = "edit"
    if not updates:
        return config

    if is_dataclass(config) and not isinstance(config, type):
        return replace(config, **updates)

    guarded = copy(config)
    for key, value in updates.items():
        if hasattr(guarded, key):
            setattr(guarded, key, value)
    return guarded
