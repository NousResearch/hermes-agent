"""Call-site API for messaging-gateway and cron shared metrics.

Platform connection health, outbound delivery outcomes, first-reply latency and cron runs happen on
the gateway event loop, adapter tasks or the cron ticker, never inside a model or tool call. Every
recorder takes RAW runtime objects, derives only closed classes / buckets / public platform names,
and hands the recording to one background worker so the caller's loop never waits on the first
runtime build or a config read. The worker runs in a copy of the caller's context, so the profile
the caller was scoped to (gateway multiplex, cron ``_profile_cron_scope``) owns the row.

This module is imported by ``gateway/platforms/base.py`` at class-definition time: keep imports lazy.
"""

from __future__ import annotations

import contextvars
import functools
import logging
import threading
import time
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from typing import Any, Callable

logger = logging.getLogger(__name__)

_executor: ThreadPoolExecutor | None = None
_executor_lock = threading.Lock()


def _submit(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Future | None:
    """Fire-and-forget: run ``fn`` on the metrics worker in the caller's context (profile scope)."""
    global _executor
    try:
        with _executor_lock:
            if _executor is None:
                _executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hermes-gateway-metrics")
        return _executor.submit(contextvars.copy_context().run, _guarded, fn, *args, **kwargs)
    except Exception:  # interpreter shutdown refuses new work: the metric is simply dropped
        logger.debug("Gateway shared metric not scheduled", exc_info=True)
        return None


def _guarded(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
    try:
        fn(*args, **kwargs)
    except Exception:
        logger.debug("Gateway shared metric not recorded", exc_info=True)


def drain(timeout: float = 5.0) -> None:
    """Wait for queued recordings (tests and live probes; the worker is FIFO)."""
    future = _submit(lambda: None)
    if future is not None:
        future.result(timeout=timeout)


def _emit(mark_name: str, build: Callable[..., dict[str, str] | None], **raw: Any) -> None:
    from . import shared_metrics_contract as contract
    from .shared_metrics_events import _emit as emit

    emit(getattr(contract, mark_name), build, **raw)


def _platform(value: Any) -> str:
    from .shared_metrics_contract import adapter_platform

    return adapter_platform(getattr(value, "platform", value))


# ---- classification (closed classes from exception types, HTTP statuses and Hermes's own codes) ----

_STATUS_CLASSES = {401: "auth", 403: "forbidden", 413: "too_long", 429: "rate_limited"}
# Adapter fatal codes are Hermes-authored identifiers (``telegram_auth_error``); first match wins, so
# "missing credentials" reads as configuration, not a rejected credential.
_CODE_CLASSES = (
    (("missing", "dependency", "npm", "config", "conflict", "lock", "bind"), "config"),
    (("auth", "credential", "token", "unauthor", "forbidden", "login"), "auth"),
    (("ratelimit", "rate_limit", "flood", "too_many", "toomany", "429"), "rate_limited"),
    (("network", "connect", "timeout", "socket", "dns", "ssl", "lost", "exited", "stream"), "network"),
)
_SEND_KIND_CLASSES = {
    "too_long": "too_long", "forbidden": "forbidden", "rate_limited": "rate_limited", "transient": "network",
}


def _status_code(exc: BaseException | None) -> int | None:
    # Stored values only, never a property: a deprecated one warns (websockets' ``code``), and a
    # classifier must not run library code against the adapter's live exception.
    from inspect import getattr_static

    for holder in (exc, getattr_static(exc, "response", None)):
        for attr in ("status_code", "status", "code"):
            value = getattr_static(holder, attr, None)
            if isinstance(value, int) and not isinstance(value, bool) and 100 <= value <= 599:
                return value
    return None


def _code_class(code: Any) -> str | None:
    text = str(code or "").strip().lower()
    if not text:
        return None
    return next((cls for needles, cls in _CODE_CLASSES if any(n in text for n in needles)), "other")


def _exception_class(exc: BaseException | None) -> str:
    """Connection error class from the exception's type and HTTP status only (never its message)."""
    if exc is None:
        return "other"
    status = _status_code(exc)
    if status in (401, 403):
        return "auth"
    if status == 429:
        return "rate_limited"
    if isinstance(exc, ImportError):
        return "config"
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return "network"
    names = "".join(cls.__name__.lower() for cls in type(exc).__mro__[:-1])
    # Type names say auth / rate / network; "configuration" is only ever a Hermes fatal code.
    return next((cls for needles, cls in _CODE_CLASSES[1:] if any(n in names for n in needles)), "other")


def connect_error_class(*, exc: BaseException | None = None, fatal_code: Any = None) -> str:
    if fatal_code:
        return _code_class(fatal_code) or "other"
    return _exception_class(exc)


def delivery_failure_class(result: Any = None, exc: BaseException | None = None) -> str:
    """Why an outbound delivery failed, from the SendResult's typed fields or the exception type."""
    if exc is not None:
        status = _STATUS_CLASSES.get(_status_code(exc) or 0)
        if status is not None:
            return status
        cls = _exception_class(exc)
        return cls if cls in {"auth", "network", "rate_limited"} else "other"
    raw = getattr(result, "raw_response", None)
    status = raw.get("status_code", raw.get("status")) if isinstance(raw, dict) else None
    if isinstance(status, int) and status in _STATUS_CLASSES:
        return _STATUS_CLASSES[status]
    if getattr(result, "retry_after", None) is not None:
        return "rate_limited"
    kind = getattr(result, "error_kind", None)
    if kind is None:
        from gateway.platforms.base import classify_send_error

        kind = classify_send_error(None, str(getattr(result, "error", "") or ""))
    if kind in _SEND_KIND_CLASSES:
        return _SEND_KIND_CLASSES[kind]
    return "network" if getattr(result, "retryable", False) else "other"


# ---- hermes.platform.health ----

def platform_health_fields(*, platform: Any, event: str, error_class: str) -> dict[str, str] | None:
    from . import shared_metrics_contract as contract

    if event not in contract.PLATFORM_HEALTH_EVENTS:
        return None
    ok = event in {"connect_ok", "reconnect"}
    return {
        "error_class": "none" if ok else error_class if error_class in contract.PLATFORM_ERROR_CLASSES else "other",
        "event": event,
        "platform": _platform(platform),
    }


def _record_health(platform: Any, event: str, *, exc: BaseException | None = None, fatal_code: Any = None) -> None:
    # Classified on the worker: whatever an exception carries, the caller re-raising it never sees us.
    error_class = "none" if event in {"connect_ok", "reconnect"} else connect_error_class(exc=exc, fatal_code=fatal_code)
    _emit("PLATFORM_HEALTH_MARK", platform_health_fields, platform=platform, event=event, error_class=error_class)


def _in_home(home: Any, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
    """Run ``fn`` scoped to ``home`` (the profile that owns the row), or the submitting context's scope."""
    if home is None:
        fn(*args, **kwargs)
        return
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    token = set_hermes_home_override(str(home))
    try:
        fn(*args, **kwargs)
    finally:
        reset_hermes_home_override(token)


def record_platform_connect(
    adapter: Any, platform: Any, *, is_reconnect: bool, ok: bool, exc: BaseException | None = None,
) -> None:
    """One adapter connect attempt (cold start or reconnect watcher), from the runner's single
    connect seam. A failed attempt is ``connect_failed`` whichever path made it."""
    event = ("reconnect" if is_reconnect else "connect_ok") if ok else "connect_failed"
    fatal_code = None if ok or exc is not None else getattr(adapter, "fatal_error_code", None)
    _submit(_record_health, platform, event, exc=exc, fatal_code=fatal_code)


# The user turned the platform off (relay opt-out revokes its credential): not a lost connection.
_USER_CHOSEN_FATAL_CODES = frozenset({"relay_disabled"})


def record_platform_disconnect(adapter: Any, *, hermes_home: Any = None) -> None:
    """A live adapter lost its connection after startup (its fatal-error handler owned it).
    ``hermes_home``: the owning profile, for handlers that run outside its scope (multiplexed
    secondaries)."""
    fatal_code = getattr(adapter, "fatal_error_code", None)
    if fatal_code in _USER_CHOSEN_FATAL_CODES:
        return
    _submit(_in_home, hermes_home, _record_health, adapter, "disconnect", fatal_code=fatal_code)


# ---- hermes.platform.delivery + hermes.gateway.reply_latency ----

def delivery_fields(*, platform: Any, failure_class: str) -> dict[str, str]:
    from . import shared_metrics_contract as contract

    return {
        "failure_class": failure_class if failure_class in contract.DELIVERY_FAILURE_CLASSES else "other",
        "outcome": "sent" if failure_class == "none" else "failed",
        "platform": _platform(platform),
    }


def records_delivery(send: Callable[..., Any]) -> Callable[..., Any]:
    """Count one logical outbound delivery (all retries and the plain-text fallback included) per
    call of the wrapped ``_send_with_retry`` (replies, busy acks, command replies alike)."""

    @functools.wraps(send)
    async def wrapper(self, *args: Any, **kwargs: Any) -> Any:
        chat_id = kwargs.get("chat_id", args[0] if args else None)
        try:
            result = await send(self, *args, **kwargs)
        except Exception as exc:
            _submit(_record_delivery, self, exc=exc, chat_id=chat_id)
            raise
        if isinstance(getattr(result, "success", None), bool):
            _submit(_record_delivery, self, result, chat_id=chat_id)
        return result

    return wrapper


def _record_delivery(
    adapter: Any, result: Any = None, *, exc: BaseException | None = None, chat_id: Any = None,
) -> None:
    failure_class = "none" if exc is None and result.success else delivery_failure_class(result, exc)
    _emit("PLATFORM_DELIVERY_MARK", delivery_fields, platform=_source_platform(adapter, chat_id),
          failure_class=failure_class)


def stops_reply_clock(send_or_edit: Callable[..., Any]) -> Callable[..., Any]:
    """Stream consumer transport: the first delivered visible text (frame, draft or message)."""

    @functools.wraps(send_or_edit)
    async def wrapper(self, *args: Any, **kwargs: Any) -> Any:
        ok = await send_or_edit(self, *args, **kwargs)
        if ok is True and getattr(self, "_last_sent_text", None):
            stop_reply_clock(getattr(self, "adapter", None), getattr(self, "chat_id", None))
        return ok

    return wrapper


_REPLY_THRESHOLDS = ((2.0, "lt_2s"), (5.0, "2s_to_5s"), (15.0, "5s_to_15s"), (60.0, "15s_to_60s"))
_REPLY_CLOCK_MAX = 1024
_REPLY_CLOCK_MAX_AGE = 3600.0
# (platform value, chat id) -> (monotonic start, owning Hermes home). Keyed without the profile: the
# send side may run outside the turn's scope, so the start side records whose row it is.
_reply_clocks: OrderedDict[tuple[str, str], tuple[float, str]] = OrderedDict()
_reply_lock = threading.Lock()
# One connector adapter fronting several platforms: the turn started under the platform the message
# came in on (``discord``), its reply leaves through this adapter.
_FRONTING_PLATFORMS = frozenset({"relay"})


def _source_platform(adapter: Any, chat_id: Any) -> Any:
    """The platform a fronting adapter's chat really lives on (the relay connector learned it from
    the inbound, else its primary platform); the adapter's own platform otherwise. Read-only."""
    platform = getattr(adapter, "platform", None)
    if str(getattr(platform, "value", platform) or "") not in _FRONTING_PLATFORMS or chat_id in (None, ""):
        return adapter
    resolve = getattr(adapter, "_chat_platform", None)
    try:
        return (resolve(str(chat_id)) if callable(resolve) else None) or adapter
    except Exception:
        return adapter


def _clock_key(platform: Any, chat_id: Any) -> tuple[str, str] | None:
    value = str(getattr(platform, "value", platform) or "")
    return (value, str(chat_id)) if value and chat_id not in (None, "") else None


def start_reply_clock(source: Any, *, internal: bool = False) -> None:
    """An inbound user message was accepted as a new agent turn (internal events never are)."""
    key = None if internal else _clock_key(getattr(source, "platform", None), getattr(source, "chat_id", None))
    if key is None:
        return
    from hermes_constants import get_hermes_home

    with _reply_lock:
        _reply_clocks.pop(key, None)
        _reply_clocks[key] = (time.monotonic(), str(get_hermes_home()))
        while len(_reply_clocks) > _REPLY_CLOCK_MAX:
            _reply_clocks.popitem(last=False)


def stop_reply_clock(adapter: Any, chat_id: Any, result: Any = None) -> None:
    """First outbound text for the chat's pending turn (stream first chunk or the final reply; busy
    acks and command replies never stop it). ``result``: a failed send is not a reply."""
    key = _clock_key(getattr(adapter, "platform", None), chat_id)
    if key is None or (result is not None and getattr(result, "success", False) is not True):
        return
    with _reply_lock:
        started, started_key = _reply_clocks.pop(key, None), key
        if started is None and key[0] in _FRONTING_PLATFORMS:
            fronted = [k for k in _reply_clocks if k[1] == key[1]]
            started_key = fronted[0] if len(fronted) == 1 else key
            started = _reply_clocks.pop(started_key) if len(fronted) == 1 else None
    # A turn that ended without a reply leaves its clock behind; an unrelated send hours later
    # must not be read as that turn's reply.
    if started is not None and time.monotonic() - started[0] <= _REPLY_CLOCK_MAX_AGE:
        # Labelled by the platform the message came in on, not the connector that carried the reply.
        platform = started_key[0] if started_key[0] not in _FRONTING_PLATFORMS else None
        _submit(_in_home, started[1], _emit_reply_latency, adapter, chat_id, platform, time.monotonic() - started[0])


def _emit_reply_latency(adapter: Any, chat_id: Any, platform: Any, seconds: float) -> None:
    _emit("REPLY_LATENCY_MARK", reply_latency_fields,
          platform=platform or _source_platform(adapter, chat_id), seconds=seconds)


def reply_latency_fields(*, platform: Any, seconds: float) -> dict[str, str]:
    from .shared_metrics_contract import _bucket

    return {
        "first_response_bucket": _bucket(max(0.0, seconds), _REPLY_THRESHOLDS, "gte_60s"),
        "platform": _platform(platform),
    }


# ---- hermes.cron.run ----

_CRON_NOTES_MAX = 512
# execution id -> delivery kind, noted where the job dict is in hand; popped once at the terminal write.
_cron_delivery: OrderedDict[str, str] = OrderedDict()
_cron_skipped: set[str] = set()
_cron_lock = threading.Lock()
_LOCAL_DELIVERY = frozenset({"local", ""})


def cron_delivery_kind(job: Any) -> str:
    """Where a job's output goes, as a closed kind; targets themselves never leave the machine."""
    if not isinstance(job, dict):
        return "other"
    deliver = job.get("deliver")
    parts = deliver if isinstance(deliver, (list, tuple)) else str(deliver or "").split(",")
    kinds = {str(p).strip().lower().split(":", 1)[0] for p in parts} - {""}
    if not kinds or kinds <= _LOCAL_DELIVERY:
        return "local"
    if kinds & {"none", "off"}:
        return "none"
    return "webhook" if kinds == {"webhook"} else "platform"


def note_cron_execution(job: Any) -> None:
    """Remember the delivery kind of an execution about to run (or be refused) in this process."""
    execution_id = job.get("execution_id") if isinstance(job, dict) else None
    if not execution_id:
        return
    with _cron_lock:
        _cron_delivery[str(execution_id)] = cron_delivery_kind(job)
        while len(_cron_delivery) > _CRON_NOTES_MAX:
            _cron_delivery.popitem(last=False)


def note_cron_skipped(job: Any) -> None:
    """A gate (pre-run script ``wakeAgent=false`` / no output) decided this run does no agent work."""
    execution_id = job.get("execution_id") if isinstance(job, dict) else None
    if execution_id:
        with _cron_lock:
            if len(_cron_skipped) < _CRON_NOTES_MAX:
                _cron_skipped.add(str(execution_id))


def _iso_seconds(start: Any, end: Any) -> float:
    try:
        return max(0.0, (datetime.fromisoformat(str(end)) - datetime.fromisoformat(str(start))).total_seconds())
    except (TypeError, ValueError):
        return 0.0


def cron_run_fields(*, outcome: str, delivery_kind: str, seconds: float) -> dict[str, str]:
    from . import shared_metrics_contract as contract

    return {
        "delivery_kind": delivery_kind if delivery_kind in contract.CRON_DELIVERY_KINDS else "other",
        "duration_bucket": contract.duration_bucket(int(seconds * 1000)),
        "outcome": outcome if outcome in contract.CRON_RUN_OUTCOMES else "failed",
    }


def record_cron_finish(record: Any, delivery_outcome: Any = None) -> None:
    """One execution reached its (write-once) terminal row. A row that never started running was
    refused before any side effect (claim lost, dispatch limit, executor shutdown): ``skipped``,
    unless a failure notice was delivered for it."""
    if not isinstance(record, dict) or not record.get("id"):
        return
    execution_id = str(record["id"])
    with _cron_lock:
        delivery_kind = _cron_delivery.pop(execution_id, "other")
        gated = execution_id in _cron_skipped
        _cron_skipped.discard(execution_id)
    started = record.get("started_at")
    success = record.get("status") == "completed"
    if gated or (not started and not success and delivery_outcome is None):
        outcome = "skipped"
    else:
        outcome = "success" if success else "failed"
    _submit(_emit, "CRON_RUN_MARK", cron_run_fields, outcome=outcome, delivery_kind=delivery_kind,
            seconds=_iso_seconds(started, record.get("finished_at")) if started else 0.0)


def record_cron_missed(job: Any) -> None:
    """A scheduled occurrence the scheduler dropped without running it (catch-up disabled after
    downtime, or a one-shot past its grace window)."""
    _submit(_emit, "CRON_RUN_MARK", cron_run_fields, outcome="missed",
            delivery_kind=cron_delivery_kind(job), seconds=0.0)
