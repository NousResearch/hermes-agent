"""Typed outcomes for plugin-triggered gateway message injection.

``PluginContext.inject_message()`` historically answered ``True`` as soon as a
dispatch coroutine was *scheduled*, which is indistinguishable from an unknown
session, a rotated session, revoked authorization, or a missing adapter. This
module owns the typed contract that lets a caller tell adoption apart from a
silent drop, so neither the plugin registry nor ``gateway/run.py`` grows another
authority subsystem.

Settlement law
--------------
Every outcome here is either **settled** (a terminal fact: the event was adopted,
or it was refused before reaching the session) or **indeterminate** (delivery is
unknown and may yet be, or already have been, adopted). Refusal is the strong
claim, and it is only made when the host can prove the event never crossed into
``adapter.handle_message``:

* an observation timeout is indeterminate -- the caller stopped waiting, the host
  did not stop working;
* cancellation or an adapter exception *after* that boundary is likewise
  indeterminate, because the adapter may already have run its side effect;
* cancellation that provably won the race against adapter entry stays a
  terminal, retry-safe refusal.

Collapsing any of these into a retry-safe ``accepted=False`` would let a webhook
answer "failed" and re-send a wake that already landed, delivering it twice.

``correlation_id`` is a bounded opaque tag echoed into the dispatched event's
metadata so a caller can *recognize* its event. It is deliberately **not** a
durable idempotency key: nothing in this path deduplicates on it, and two
injections carrying the same id both reach the session. Retry safety therefore
comes from :attr:`GatewayInjectionResult.safe_to_retry`, not from the tag.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import threading
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Reasons a gateway message injection can resolve to. ``adopted`` is the only
# outcome that means the event reached the stored session. Everything else is a
# refusal a caller can act on -- except ``timeout``, which is not an outcome at
# all but the absence of one, and ``cancelled``/``internal_error``, which are
# refusals only while delivery is still disprovable (see ``INDETERMINATE_REASONS``
# and ``DELIVERY_SENSITIVE_REASONS``).
GATEWAY_INJECTION_REASONS = (
    "adopted",
    "unknown_session",
    "unauthorized",
    "no_adapter",
    "gateway_draining",
    "internal_error",
    "not_scheduled",
    "injection_denied",
    "no_gateway",
    "invalid_request",
    "unsupported",
    "cancelled",
    "timeout",
    "cli_queued",
)

# Reasons that describe an unfinished dispatch rather than a decided one. A
# result carrying one of these is falsy but MUST NOT be read as a denial.
INDETERMINATE_REASONS = frozenset({"timeout"})

# Reasons whose settlement depends on how far dispatch got. Losing a dispatch to
# cancellation or an exception is a proven refusal only while the event is still
# on the host's side of ``adapter.handle_message``; past that boundary the
# adapter may already have run its side effect, so the outcome is
# delivery-unknown. See :class:`InjectionDelivery`.
DELIVERY_SENSITIVE_REASONS = frozenset({"cancelled", "internal_error"})

# Terminal refusals that still cannot be blindly retried, because the event may
# already have reached the adapter before the failure surfaced.
_UNSAFE_RETRY_REASONS = frozenset({"internal_error"})


class InjectionDelivery:
    """Tracks the one boundary that makes a refusal provable.

    ``refused`` claims the event will never reach the session. That is only
    true while dispatch has not entered ``adapter.handle_message``. Both sides
    of the race arbitrate through this object: dispatch claims the boundary with
    :meth:`enter_adapter`, a caller reserves cancellation with
    :meth:`request_cancel`, and whoever gets there second is told it lost.
    """

    __slots__ = ("_lock", "_adapter_entered", "_cancel_requested")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._adapter_entered = False
        self._cancel_requested = False

    def enter_adapter(self) -> bool:
        """Claim the adoption boundary; False when a cancel already won it."""
        with self._lock:
            if self._cancel_requested:
                return False
            self._adapter_entered = True
            return True

    def request_cancel(self) -> bool:
        """Reserve cancellation; False when the adapter already has the event."""
        with self._lock:
            if self._adapter_entered:
                return False
            self._cancel_requested = True
            return True

    @property
    def delivery_unknown(self) -> bool:
        """Whether the host can no longer prove the event was not delivered."""
        with self._lock:
            return self._adapter_entered


MAX_INJECTION_CORRELATION_ID = 128


@dataclass(frozen=True)
class GatewayInjectionResult:
    """What gateway dispatch decided about one injected message.

    ``accepted`` is truthy only for ``reason == "adopted"``, i.e. the event was
    handed to the adapter that owns the stored session. ``session_id`` is the
    host's identity for that session -- never anything the caller supplied.

    ``settled`` distinguishes a decided outcome from an observation that ran out
    of patience. Read :attr:`refused` before treating a falsy result as denial,
    and :attr:`safe_to_retry` before re-issuing the injection.
    """

    accepted: bool
    reason: str
    session_id: Optional[str] = None
    session_key: Optional[str] = None
    correlation_id: Optional[str] = None
    settled: bool = True

    def __bool__(self) -> bool:
        return self.accepted

    @property
    def indeterminate(self) -> bool:
        """The dispatch has not resolved; it may still be adopted."""
        return not self.settled

    @property
    def refused(self) -> bool:
        """A terminal denial: the host proved the event never crossed into the
        adapter, so it will never reach the session."""
        return self.settled and not self.accepted

    @property
    def safe_to_retry(self) -> bool:
        """Whether re-issuing this injection cannot duplicate a delivered wake.

        False for anything indeterminate. ``correlation_id`` does not make a
        retry safe -- it is metadata, not durable idempotent admission -- so an
        indeterminate result must be reconciled through the originating
        :class:`GatewayInjectionHandle` instead of blindly re-sent.
        """
        return self.refused and self.reason not in _UNSAFE_RETRY_REASONS


class GatewayInjectionHandle:
    """Deferred :class:`GatewayInjectionResult` for an awaited injection.

    Await it on the gateway loop; call :meth:`result` from another thread (the
    shape a plugin's own HTTP listener needs when it must answer a request only
    after dispatch resolved). Both paths return a result object rather than
    raising, so a caller never has to guess whether a failure means "refused" or
    "crashed".

    The handle is also the **reconciliation path**. When :meth:`result` times
    out it returns an indeterminate result and leaves the dispatch running; the
    same handle can be polled with :meth:`settled` or read again with
    :meth:`result` to learn the eventual outcome. Retrying the injection instead
    would deliver the wake twice.
    """

    __slots__ = ("_future", "_correlation_id", "_session_key", "_delivery")

    def __init__(
        self,
        future: Any,
        *,
        correlation_id: Optional[str] = None,
        session_key: Optional[str] = None,
        delivery: Optional[InjectionDelivery] = None,
    ) -> None:
        self._future = future
        self._correlation_id = correlation_id
        self._session_key = session_key
        self._delivery = delivery

    @classmethod
    def resolved(cls, result: GatewayInjectionResult) -> "GatewayInjectionHandle":
        """Wrap an outcome that was decided before anything was scheduled."""
        future: Any = concurrent.futures.Future()
        future.set_result(result)
        return cls(
            future,
            correlation_id=result.correlation_id,
            session_key=result.session_key,
        )

    @property
    def correlation_id(self) -> Optional[str]:
        return self._correlation_id

    @property
    def settled(self) -> bool:
        """Whether the eventual outcome is now readable without blocking."""
        return bool(self._future.done())

    def cancel(self) -> bool:
        """Cancel the pending dispatch; a later read reports ``cancelled``.

        Cancelling once the adapter already holds the event does not undo the
        wake, so the reported outcome becomes delivery-unknown rather than a
        retry-safe refusal.
        """
        if self._delivery is not None:
            self._delivery.request_cancel()
        return bool(self._future.cancel())

    def _failure(self, reason: str) -> GatewayInjectionResult:
        settled = reason not in INDETERMINATE_REASONS
        if (
            settled
            and reason in DELIVERY_SENSITIVE_REASONS
            and self._delivery is not None
            and self._delivery.delivery_unknown
        ):
            settled = False
        return GatewayInjectionResult(
            False,
            reason,
            session_key=self._session_key,
            correlation_id=self._correlation_id,
            settled=settled,
        )

    def _coerce(self, value: Any) -> GatewayInjectionResult:
        if isinstance(value, GatewayInjectionResult):
            return value
        return self._failure("internal_error")

    async def _resolve(self) -> GatewayInjectionResult:
        future = self._future
        if isinstance(future, concurrent.futures.Future):
            future = asyncio.wrap_future(future)
        try:
            return self._coerce(await future)
        except asyncio.CancelledError:
            # A cancelled *dispatch* is an outcome; a cancelled *caller* is not.
            if self._future.cancelled():
                return self._failure("cancelled")
            raise
        except Exception:
            logger.warning("Gateway injection dispatch failed", exc_info=True)
            return self._failure("internal_error")

    def __await__(self):
        return self._resolve().__await__()

    def result(self, timeout: Optional[float] = None) -> GatewayInjectionResult:
        """Block until dispatch resolves. Never call this on the gateway loop.

        On timeout the dispatch is left running and an *indeterminate* result is
        returned (``settled=False``, ``refused=False``). Read this handle again
        to reconcile; do not re-inject.
        """
        future = self._future
        if not isinstance(future, concurrent.futures.Future):
            raise RuntimeError(
                "GatewayInjectionHandle.result() would deadlock on the gateway "
                "loop; await the handle instead"
            )
        try:
            return self._coerce(future.result(timeout))
        except concurrent.futures.TimeoutError:
            # Not a refusal: we stopped watching, the dispatch did not stop.
            return self._failure("timeout")
        except concurrent.futures.CancelledError:
            return self._failure("cancelled")
        except Exception:
            logger.warning("Gateway injection dispatch failed", exc_info=True)
            return self._failure("internal_error")


def validate_correlation_id(value: Any) -> bool:
    """Correlation ids are opaque bounded tags, never routing information."""
    return (
        isinstance(value, str)
        and 0 < len(value) <= MAX_INJECTION_CORRELATION_ID
        and value.isprintable()
    )


def submit_to_gateway(
    manager: Any,
    *,
    session_key: str,
    content: str,
    plugin_id: str,
    await_dispatch: bool,
    correlation_id: Optional[str],
) -> Any:
    """Hand an already-authorized injection to the live gateway injector.

    The caller owns the config grant check; this owns only the mechanics of
    talking to whatever injector the running host published.
    """

    def _refuse(reason: str) -> Any:
        result = GatewayInjectionResult(
            False, reason, session_key=session_key, correlation_id=correlation_id
        )
        return GatewayInjectionHandle.resolved(result) if await_dispatch else False

    if not manager.has_gateway_message_injector:
        logger.warning("inject_message: no live gateway is available")
        return _refuse("no_gateway")

    kwargs: Dict[str, Any] = {
        "session_key": session_key,
        "content": content,
        "plugin_id": plugin_id,
    }
    # Only widen the injector call when the caller asked for the new behavior,
    # so a host that published the original three-kwarg injector keeps working.
    extra = [
        name
        for name, wanted in (
            ("await_dispatch", await_dispatch),
            ("correlation_id", correlation_id is not None),
        )
        if wanted
    ]
    if extra and not manager.gateway_injector_accepts(*extra):
        logger.warning(
            "inject_message: the live gateway injector does not support "
            "dispatch-outcome reporting for plugin %s",
            plugin_id,
        )
        return _refuse("unsupported")
    if await_dispatch:
        kwargs["await_dispatch"] = True
    if correlation_id is not None:
        kwargs["correlation_id"] = correlation_id

    try:
        outcome = manager.inject_gateway_message(**kwargs)
    except Exception:
        logger.warning(
            "inject_message: gateway scheduling failed for plugin %s",
            plugin_id,
            exc_info=True,
        )
        return _refuse("internal_error")

    if not await_dispatch:
        return bool(outcome)
    if isinstance(outcome, GatewayInjectionHandle):
        return outcome
    if isinstance(outcome, GatewayInjectionResult):
        return GatewayInjectionHandle.resolved(outcome)
    return _refuse("unsupported")


# Host-authenticated identity fields, read task-locally so concurrent sessions
# in one process never observe each other's coordinates.
_CALL_CONTEXT_FIELDS = {
    "session_key": "HERMES_SESSION_KEY",
    "session_id": "HERMES_SESSION_ID",
    "platform": "HERMES_SESSION_PLATFORM",
    "source": "HERMES_SESSION_SOURCE",
    "chat_id": "HERMES_SESSION_CHAT_ID",
    "chat_type": "HERMES_SESSION_CHAT_TYPE",
    "chat_name": "HERMES_SESSION_CHAT_NAME",
    "thread_id": "HERMES_SESSION_THREAD_ID",
    "user_id": "HERMES_SESSION_USER_ID",
    "user_name": "HERMES_SESSION_USER_NAME",
    "scope_id": "HERMES_SESSION_SCOPE_ID",
    "message_id": "HERMES_SESSION_MESSAGE_ID",
}


def build_call_context(process_profile: str) -> Dict[str, str]:
    """Identity of the turn that invoked a plugin, as the host knows it.

    ``profile`` follows the same task-local rule as every other field: the
    session's own ``HERMES_SESSION_PROFILE`` when a session is bound, falling
    back to the process profile only when nothing is bound (plain CLI, cron,
    tests). A multiplexed gateway serves several profiles from one process, so
    reading the process profile while a session is bound would hand a plugin the
    right chat and the wrong policy domain.
    """
    try:
        from gateway.session_context import get_session_env
    except Exception:
        return {"profile": process_profile, **{k: "" for k in _CALL_CONTEXT_FIELDS}}

    context = {
        key: get_session_env(name, "") for key, name in _CALL_CONTEXT_FIELDS.items()
    }
    context["profile"] = (
        get_session_env("HERMES_SESSION_PROFILE", "") or process_profile
    )
    return context
