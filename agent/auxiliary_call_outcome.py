"""How one auxiliary logical call ended, for its Relay ``hermes.logical_llm_call`` scope.

The scope's ``outcome`` + ``error_class`` become the shared-metrics ``model_route`` row of an
auxiliary call, under the same contract as a primary call: a failure carries the classifier's
``FailoverReason`` for the error that ended it, a Hermes-side abort (``/stop``, Ctrl+C, an
interrupt, shutdown) is ``cancelled`` rather than ``failed``, and a success keeps the last
attempt error it recovered from (a retried timeout, a fallback past a 429). Per attempt,
:class:`_AuxiliaryCancellationDecision` settles whether an explicit cancel or the provider timeout
ended it.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Attribute an aux "invalid response" error carries when the HTTP-200 body held a provider
# ``error`` object instead of choices (aggregators relay upstream failures this way).
PROVIDER_ERROR_ATTR = "provider_error"


class _EmbeddedProviderError(Exception):
    """An HTTP-200 ``error`` object reshaped so ``classify_api_error`` reads it as an SDK body."""

    def __init__(self, error: dict[str, Any]) -> None:
        super().__init__(str(error.get("message") or ""))
        self.body = {"error": error}


def embedded_provider_error(response: Any) -> dict[str, Any] | None:
    """The ``error`` object of an HTTP-200 body that carried no choices, else None."""
    error = response.get("error") if isinstance(response, dict) else getattr(response, "error", None)
    if isinstance(error, dict):
        return error or None
    if error is None or not (hasattr(error, "message") or hasattr(error, "code")):
        return None
    fields = {"message": getattr(error, "message", None), "code": getattr(error, "code", None)}
    return {k: v for k, v in fields.items() if v is not None} or None


def is_cancellation(error: BaseException) -> bool:
    """Hermes aborted the call (explicit cancel, interrupt, Ctrl+C, task cancel, shutdown):
    everything that is not an ``Exception``, plus the ``InterruptedError`` aux streams raise."""
    return not isinstance(error, Exception) or isinstance(error, InterruptedError)


def error_class(error: BaseException, *, provider: str = "", model: str = "") -> str:
    """The classifier's reason for one aux attempt error; ``unknown`` when it cannot say.
    Never raises: telemetry must not replace the caller's exception."""
    if not isinstance(error, Exception):
        return "unknown"
    try:
        from agent.error_classifier import classify_api_error

        embedded = getattr(error, PROVIDER_ERROR_ATTR, None)
        target = _EmbeddedProviderError(embedded) if isinstance(embedded, dict) else error
        return classify_api_error(target, provider=provider, model=model).reason.value
    except Exception:
        logger.debug("Auxiliary error classification failed", exc_info=True)
        return "unknown"


def _captured_aux_cancel_requested(cancel_check: Callable[[], Any]) -> bool:
    """Read a request-thread cancellation source without leaking its failures."""
    try:
        return bool(cancel_check())
    except Exception:
        logging.getLogger("agent.auxiliary_client").debug("captured aux cancel check failed", exc_info=True)
        return False


class _AuxiliaryCancellationDecision:
    """Atomically choose explicit cancellation or provider timeout per attempt."""

    def __init__(self, source_cancel_check: Callable[[], Any]) -> None:
        self._source_cancel_check = source_cancel_check
        self._lock = threading.Lock()
        self._outcome = "active"

    def __call__(self) -> bool:
        with self._lock:
            if self._outcome == "active" and _captured_aux_cancel_requested(self._source_cancel_check):
                self._outcome = "cancelled"
            return self._outcome == "cancelled"

    def begin_timeout_cleanup(self) -> bool:
        """Return whether timeout won and destructive cleanup is permitted."""
        with self._lock:
            if self._outcome == "active":
                cancelled = _captured_aux_cancel_requested(self._source_cancel_check)
                self._outcome = "cancelled" if cancelled else "timed_out"
            return self._outcome == "timed_out"
