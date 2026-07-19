"""Direct NeMo Relay integration for Hermes shared client metrics."""

from __future__ import annotations

import atexit
import logging
import threading
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Callable

from hermes_cli import __version__

from . import relay_runtime
from .shared_metrics import SharedMetricsStore
from .shared_metrics_contract import (
    MODEL_CALL_SCOPE,
    SCHEMA_KEY,
    SCHEMA_VERSION,
    SUBSCRIBER_NAME,
    model_call_fields,
    model_call_outcome,
)
from .shared_metrics_subscriber import SharedMetricsSubscriber

logger = logging.getLogger(__name__)

HANDLED_HOOKS = frozenset({
    "on_session_start",
    "on_session_end",
    "on_session_finalize",
    "on_session_reset",
    "pre_api_request",
    "post_api_request",
    "api_request_error",
})

_RUNTIME_FAILED = object()
_RUNTIME: _Runtime | object | None = None
_RUNTIME_LOCK = threading.RLock()


@dataclass
class _ModelCall:
    handle: Any
    task_id: str
    fields: dict[str, str]


@dataclass
class _MetricsSession:
    session_id: str
    relay_session: relay_runtime.RelaySession
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    closing: bool = False
    model_calls: dict[str, _ModelCall] = field(default_factory=dict)


class _Runtime:
    """Own shared-metrics state layered on the Hermes core Relay host."""

    def __init__(self, host: relay_runtime.RelayRuntime | None = None) -> None:
        resolved_host = host or relay_runtime.get_runtime()
        if resolved_host is None:
            raise RuntimeError("Hermes core Relay runtime is unavailable")
        self.host: relay_runtime.RelayRuntime = resolved_host
        self.relay = self.host.relay
        self._sessions_lock = threading.RLock()
        self._sessions: dict[str, _MetricsSession] = {}
        self.subscriber = SharedMetricsSubscriber(SharedMetricsStore(), __version__)
        self.relay.subscribers.register(SUBSCRIBER_NAME, self.subscriber)
        self._registered = True
        atexit.register(self.shutdown)

    def ensure_session(self, event: dict[str, Any]) -> _MetricsSession | None:
        session_id = str(event.get("session_id") or "")
        if not session_id:
            return None
        relay_session = self.host.ensure_session(event)
        if relay_session is None:
            return None
        with self._sessions_lock:
            session = self._sessions.get(session_id)
            if session is None:
                session = _MetricsSession(
                    session_id=session_id,
                    relay_session=relay_session,
                )
                self._sessions[session_id] = session
        with session.lock:
            if session.closing:
                return None
        return session

    def _run_in_session(
        self,
        session: _MetricsSession,
        callback: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        return self.host.run_in_session(
            session.relay_session,
            callback,
            *args,
            **kwargs,
        )

    def start_model_call(self, event: dict[str, Any]) -> None:
        session = self.ensure_session(event)
        if session is None:
            return
        request_id = str(event.get("api_request_id") or "")
        if not request_id:
            return
        fields = model_call_fields(event)
        model_family = fields["model_family"]
        with session.lock:
            if session.closing:
                return
            existing = session.model_calls.get(request_id)
            if existing is not None:
                existing.fields = fields
                return
            handle = self._run_in_session(
                session,
                self.relay.llm.call,
                MODEL_CALL_SCOPE,
                self.relay.LLMRequest({}, {}),
                handle=session.relay_session.handle,
                metadata={SCHEMA_KEY: SCHEMA_VERSION},
                model_name=model_family,
            )
            session.model_calls[request_id] = _ModelCall(
                handle=handle,
                task_id=str(event.get("task_id") or ""),
                fields=fields,
            )

    def end_model_call(self, event: dict[str, Any], outcome: str | None = None) -> None:
        session = self._session(event)
        if session is None:
            return
        request_id = str(event.get("api_request_id") or "")
        with session.lock:
            if session.closing:
                return
            model_call = session.model_calls.get(request_id)
            if model_call is None:
                return
            fields = model_call_fields(event)
            model_call.fields = fields
            self._finish_model_call(
                session,
                request_id,
                outcome or model_call_outcome(event),
            )

    def end_pending_model_calls(self, event: dict[str, Any]) -> None:
        session = self._session(event)
        if session is None:
            return
        with session.lock:
            if session.closing:
                return
            self._end_pending_model_calls(session, event)

    def close_session(self, event: dict[str, Any]) -> None:
        session = self._session(event)
        if session is None:
            return
        failures: list[str] = []
        with session.lock:
            if session.closing:
                return
            session.closing = True
            self._end_pending_model_calls(session, event)
        try:
            self.relay.subscribers.flush()
        except Exception as exc:
            failures.append(f"subscriber flush failed: {exc}")
        self._export()
        with self._sessions_lock:
            if self._sessions.get(session.session_id) is session:
                self._sessions.pop(session.session_id, None)
        if failures:
            logger.warning(
                "Hermes shared-metrics session %s closed with errors: %s",
                session.session_id,
                "; ".join(failures),
            )

    def shutdown(self) -> None:
        with self._sessions_lock:
            session_ids = list(self._sessions)
        for session_id in session_ids:
            self._safe(self.close_session, {"session_id": session_id})
        if not self._registered:
            return
        self._safe(self.relay.subscribers.flush)
        self._export()
        self._safe(self.relay.subscribers.deregister, SUBSCRIBER_NAME)
        self._registered = False
        try:
            atexit.unregister(self.shutdown)
        except Exception:
            pass

    def _session(self, event: dict[str, Any]) -> _MetricsSession | None:
        session_id = str(event.get("session_id") or "")
        with self._sessions_lock:
            return self._sessions.get(session_id)

    def _finish_model_call(
        self,
        session: _MetricsSession,
        request_id: str,
        outcome: str,
    ) -> None:
        model_call = session.model_calls.pop(request_id, None)
        if model_call is None:
            return
        try:
            self._run_in_session(
                session,
                self.relay.llm.call_end,
                model_call.handle,
                {**model_call.fields, "outcome": outcome},
                metadata={SCHEMA_KEY: SCHEMA_VERSION},
            )
        except Exception:
            logger.warning(
                "Hermes shared-metrics model call close failed", exc_info=True
            )

    def _end_pending_model_calls(
        self,
        session: _MetricsSession,
        event: dict[str, Any],
    ) -> None:
        task_id = str(event.get("task_id") or "")
        request_ids = [
            request_id
            for request_id, model_call in session.model_calls.items()
            if not task_id or model_call.task_id == task_id
        ]
        outcome = "cancelled" if event.get("interrupted") else "failed"
        for request_id in request_ids:
            self._finish_model_call(session, request_id, outcome)

    def _export(self) -> None:
        self._safe(self.subscriber.store.create_and_export_package)

    @staticmethod
    def _safe(callback: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        try:
            return callback(*args, **kwargs)
        except Exception:
            logger.warning("Hermes shared metrics operation failed", exc_info=True)
            return None


@lru_cache(maxsize=1)
def enabled() -> bool:
    """Return the process-lifetime Hermes shared-metrics policy."""
    try:
        from hermes_cli.config import load_config_readonly

        config = load_config_readonly() or {}
    except Exception:
        logger.debug("Unable to read Hermes shared-metrics policy", exc_info=True)
        return False
    if not isinstance(config, dict):
        return False
    telemetry = config.get("telemetry")
    if not isinstance(telemetry, dict):
        return False
    shared_metrics = telemetry.get("shared_metrics")
    return isinstance(shared_metrics, dict) and shared_metrics.get("enabled") is True


def handles_hook(hook_name: str) -> bool:
    return hook_name in HANDLED_HOOKS and enabled()


def observe_lifecycle(hook_name: str, **kwargs: Any) -> None:
    """Project one Hermes lifecycle event into the core Relay integration."""
    if not handles_hook(hook_name):
        return
    runtime = _get_runtime()
    if runtime is None:
        return
    try:
        if hook_name == "on_session_start":
            runtime.ensure_session(kwargs)
        elif hook_name == "pre_api_request":
            runtime.start_model_call(kwargs)
        elif hook_name == "post_api_request":
            runtime.end_model_call(kwargs, "success")
        elif hook_name == "api_request_error":
            if kwargs.get("retryable") is False:
                runtime.end_model_call(kwargs, "failed")
        elif hook_name == "on_session_end":
            runtime.end_pending_model_calls(kwargs)
        elif hook_name in {"on_session_finalize", "on_session_reset"}:
            runtime.close_session(kwargs)
    except Exception:
        logger.warning(
            "Hermes shared metrics hook failed: %s", hook_name, exc_info=True
        )


def prepare_session_start() -> None:
    """Register the subscriber before any producer opens the session scope."""
    if enabled():
        _get_runtime()


def _get_runtime() -> _Runtime | None:
    global _RUNTIME
    with _RUNTIME_LOCK:
        if isinstance(_RUNTIME, _Runtime):
            return _RUNTIME
        if _RUNTIME is _RUNTIME_FAILED:
            return None
        try:
            _RUNTIME = _Runtime()
        except Exception:
            logger.warning("Hermes shared metrics initialization failed", exc_info=True)
            _RUNTIME = _RUNTIME_FAILED
            return None
        return _RUNTIME


def _reset_for_tests() -> None:
    """Reset process-global state for isolated tests."""
    global _RUNTIME
    with _RUNTIME_LOCK:
        if isinstance(_RUNTIME, _Runtime):
            _RUNTIME.shutdown()
        _RUNTIME = None
        enabled.cache_clear()
