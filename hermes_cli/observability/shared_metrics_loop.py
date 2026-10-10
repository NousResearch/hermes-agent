"""Learning-loop, delegation and execution-backend shared metrics.

Memory operations, curator runs, delegate_task fan-outs and terminal/browser/execute_code backend
calls. Producers call the ``record_*`` functions with RAW values; every builder here maps them onto
the closed enums and buckets in ``shared_metrics_contract``. Recording is a no-op unless shared
metrics are enabled for the owning profile and never raises into the caller.

This module imports only the stdlib at load time so tool modules can import it on their hot path.
"""

from __future__ import annotations

import contextlib
import contextvars
import json
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def _bound_home(hermes_home: Any) -> Iterator[None]:
    """Record in the profile captured by the producer (curator/delegation threads carry no binding)."""
    if not hermes_home:
        yield
        return
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    token = set_hermes_home_override(hermes_home)
    try:
        yield
    finally:
        reset_hermes_home_override(token)


def _emit(mark_attr: str, build: Callable[..., dict[str, str] | None], *, hermes_home: Any = None, **raw: Any) -> None:
    try:
        from . import shared_metrics_contract as contract
        from .shared_metrics_events import _emit as emit_mark

        with _bound_home(hermes_home):
            emit_mark(getattr(contract, mark_attr), build, **raw)
    except Exception:
        logger.debug("Shared-metrics %s not recorded", mark_attr, exc_info=True)


def _norm(value: Any) -> str:
    return str(value or "").strip().lower()


def _parsed(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        return result
    try:
        data = json.loads(result) if isinstance(result, (str, bytes)) else None
    except (TypeError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _current_home() -> str | None:
    try:
        from hermes_constants import get_hermes_home

        return str(get_hermes_home())
    except Exception:
        return None


# ---- memory ----------------------------------------------------------------------------------

_MEMORY_OP_WORDS = {
    **dict.fromkeys(
        ("add", "conclude", "create", "ingest", "remember", "retain", "save", "store", "upload"), "add"
    ),
    **dict.fromkeys(("edit", "replace", "update"), "replace"),
    **dict.fromkeys(("patch",), "patch"),
    **dict.fromkeys(("delete", "forget", "remove"), "remove"),
    **dict.fromkeys(("probe", "query", "reason", "reasoning", "recall", "related", "search"), "search"),
    **dict.fromkeys(("context", "get", "list", "profile", "read"), "read"),
}
# A batch can carry many operations; one call never exports more rows than this.
_MAX_BATCH_OPS = 25


def memory_op(value: Any) -> str:
    return _MEMORY_OP_WORDS.get(_norm(value), "other")


def memory_provider_op(tool_name: Any, args: Any) -> str:
    """The operation a provider tool performs: its ``action`` argument, else a verb in its name."""
    action = args.get("action") if isinstance(args, dict) else None
    if isinstance(action, str) and _norm(action) in _MEMORY_OP_WORDS:
        return _MEMORY_OP_WORDS[_norm(action)]
    for word in reversed(_norm(tool_name).replace("-", "_").split("_")):
        if word in _MEMORY_OP_WORDS:
            return _MEMORY_OP_WORDS[word]
    return "other"


def memory_provider_name(value: Any) -> str:
    from .shared_metrics_contract import MEMORY_PROVIDERS

    name = _norm(value) or "builtin"
    return name if name in MEMORY_PROVIDERS else "plugin"


def _memory_origin() -> str:
    from tools.skill_provenance import is_background_review

    return "background_review" if is_background_review() else "foreground"


def memory_op_fields(*, op: Any, provider: Any, outcome: Any, origin: Any, failure_class: Any) -> dict[str, str]:
    from .shared_metrics_contract import MEMORY_OP_FAILURE_CLASSES, MEMORY_OP_ORIGINS, MEMORY_OP_OUTCOMES, MEMORY_OPS

    outcome_value, origin_value, class_value = _norm(outcome), _norm(origin), _norm(failure_class)
    outcome_value = outcome_value if outcome_value in MEMORY_OP_OUTCOMES else "failed"
    return {
        "failure_class": "none" if outcome_value == "success" else (
            class_value if class_value in MEMORY_OP_FAILURE_CLASSES - {"none"} else "unknown"
        ),
        "op": op if op in MEMORY_OPS else "other",
        "origin": origin_value if origin_value in MEMORY_OP_ORIGINS else "foreground",
        "outcome": outcome_value,
        "provider": memory_provider_name(provider),
    }


def _record_memory_ops(ops: Iterable[str], *, provider: Any, outcome: str, failure_class: str) -> None:
    try:
        origin = _memory_origin()
    except Exception:
        origin = "foreground"
    for op in list(ops)[:_MAX_BATCH_OPS]:
        _emit(
            "MEMORY_OP_MARK", memory_op_fields, op=op, provider=provider, outcome=outcome, origin=origin,
            failure_class=failure_class,
        )


def builtin_memory_ops(action: Any, operations: Any) -> list[str]:
    if isinstance(operations, list) and operations:
        return [memory_op(op.get("action") if isinstance(op, dict) else None) for op in operations]
    return [memory_op(action)]


def record_builtin_memory_call(action: Any, operations: Any, *, outcome: str, failure_class: str = "unknown") -> None:
    """One row per operation of a built-in ``memory`` tool call (a batch applies all or none).
    ``failure_class`` is the store's closed refusal/failure name, ``none`` on success."""
    _record_memory_ops(
        builtin_memory_ops(action, operations), provider="builtin", outcome=outcome, failure_class=failure_class,
    )


def _provider_result_failed(result: Any) -> bool:
    if not isinstance(result, str):
        return False
    head = result.lstrip()[:200]
    return head.startswith('{"error"') or '"success": false' in head


def record_provider_memory_call(provider: Any, tool_name: Any, args: Any, result: Any = None, *, raised: bool = False) -> None:
    """One row per memory-provider tool call (plugin providers expose their own tools)."""
    failed = "exception" if raised else "provider_error" if _provider_result_failed(result) else None
    _record_memory_ops(
        [memory_provider_op(tool_name, args)], provider=provider, outcome="failed" if failed else "success",
        failure_class=failed or "none",
    )


# ---- curator ---------------------------------------------------------------------------------

_CURATOR_COUNT_KEYS = ("archived", "created", "merged", "patched")


def curator_run_fields(*, trigger: Any, outcome: Any, counts: Any) -> dict[str, str]:
    from .shared_metrics_contract import CURATOR_OUTCOMES, count_bucket

    resolved = counts() if callable(counts) else counts
    resolved = resolved if isinstance(resolved, dict) else {}
    outcome_value = _norm(outcome)
    fields = {
        f"{key}_bucket": count_bucket(value if isinstance(value, int) and not isinstance(value, bool) else 0)
        for key in _CURATOR_COUNT_KEYS for value in (resolved.get(key),)
    }
    return {
        **fields,
        "outcome": outcome_value if outcome_value in CURATOR_OUTCOMES else "failed",
        "trigger": "scheduled" if _norm(trigger) == "scheduled" else "manual",
    }


def record_curator_run(
    *, trigger: str, outcome: str, counts: dict[str, int] | Callable[[], dict[str, int]] | None = None,
    hermes_home: Any = None,
) -> None:
    """One row per curator pass. ``counts`` may be a callable so the diff is only computed when on."""
    _emit(
        "CURATOR_RUN_MARK", curator_run_fields, trigger=trigger, outcome=outcome, counts=counts or {},
        hermes_home=hermes_home,
    )


# ---- delegation ------------------------------------------------------------------------------

_SUCCESS_STATUSES = frozenset({"completed"})
_CANCELLED_STATUSES = frozenset({"interrupted", "cancelled"})
# Runs whose units never report back (a crashed async runner) must not grow this forever.
_MAX_OPEN_DELEGATIONS = 256


@dataclass
class _DelegationRun:
    expected: int
    depth: int
    hermes_home: str | None
    statuses: list[Any] = field(default_factory=list)
    background: bool = False


_DELEGATIONS: dict[int, _DelegationRun] = {}
_DELEGATIONS_LOCK = threading.Lock()


def delegation_depth(depth: Any) -> str:
    value = depth if isinstance(depth, int) and not isinstance(depth, bool) else 1
    return str(max(1, value)) if value < 4 else "gte_4"


def delegation_outcome(statuses: Iterable[Any]) -> str:
    values = [_norm(s) for s in statuses]
    succeeded = sum(s in _SUCCESS_STATUSES for s in values)
    if values and succeeded == len(values):
        return "success"
    if succeeded:
        return "partial"
    return "cancelled" if any(s in _CANCELLED_STATUSES for s in values) else "failed"


def delegation_run_fields(*, statuses: Any, depth: Any, background: Any, subagents: Any) -> dict[str, str]:
    from .shared_metrics_contract import count_bucket

    return {
        "depth": delegation_depth(depth),
        "mode": "background" if background is True else "foreground",
        "outcome": delegation_outcome(statuses or ()),
        "subagent_count_bucket": count_bucket(subagents if isinstance(subagents, int) else 0),
    }


def begin_delegation_run(call_key: object, *, subagents: int, depth: int) -> None:
    """Open one delegate_task call; ``call_key`` is the object every unit of the call shares."""
    if subagents <= 0:
        return
    run = _DelegationRun(expected=subagents, depth=depth, hermes_home=_current_home())
    with _DELEGATIONS_LOCK:
        while len(_DELEGATIONS) >= _MAX_OPEN_DELEGATIONS:
            _DELEGATIONS.pop(next(iter(_DELEGATIONS)))
        _DELEGATIONS[id(call_key)] = run


def finish_delegation_unit(call_key: object, results: Iterable[Any], *, background: bool) -> None:
    """Fold one joined unit into its call; the call's single row is emitted by its last unit."""
    statuses = [r.get("status") if isinstance(r, dict) else None for r in results]
    with _DELEGATIONS_LOCK:
        run = _DELEGATIONS.get(id(call_key))
        if run is None:
            return
        run.statuses.extend(statuses)
        run.background = run.background or background
        if len(run.statuses) < run.expected:
            return
        del _DELEGATIONS[id(call_key)]
    _emit(
        "DELEGATION_RUN_MARK", delegation_run_fields, statuses=run.statuses, depth=run.depth,
        background=run.background, subagents=run.expected, hermes_home=run.hermes_home,
    )


# ---- execution backends ----------------------------------------------------------------------

_STATUS_ERROR_CLASSES = {
    "timeout": "timeout", "timed_out": "timeout", "interrupted": "interrupted", "cancelled": "interrupted",
    "blocked": "blocked",
}


def _terminal_failure(data: dict[str, Any]) -> str | None:
    # 124 is the tool's own foreground-deadline code; it carries partial output and no ``error``.
    if data.get("exit_code") == 124:
        return "timeout"
    return "tool_error" if data.get("error") else None


def _code_failure(data: dict[str, Any]) -> str | None:
    status = _norm(data.get("status"))
    if status in _STATUS_ERROR_CLASSES:
        return _STATUS_ERROR_CLASSES[status]
    # A traceback in the user's own code still ran on the backend; only a backend-level error
    # (no kernel reply, sandbox launch failure) carries ``error`` without any output.
    if status == "error" and data.get("error") and not data.get("output"):
        return "tool_error"
    return None


def _browser_failure(data: dict[str, Any]) -> str | None:
    return "tool_error" if data.get("success") is False or data.get("error") else None


_FAILURE_CLASSIFIERS = {"browser": _browser_failure, "code": _code_failure, "terminal": _terminal_failure}


def execution_backend_fields(*, kind: Any, backend: Any, result: Any, error_class: Any) -> dict[str, str] | None:
    from .shared_metrics_contract import EXECUTION_BACKENDS_BY_KIND, TOOL_ERROR_CLASSES

    kind_value = _norm(kind)
    if kind_value not in EXECUTION_BACKENDS_BY_KIND:
        return None
    backend_value = _norm(backend() if callable(backend) else backend)
    failure = error_class if error_class is not None else _FAILURE_CLASSIFIERS[kind_value](_parsed(result))
    return {
        "backend": backend_value if backend_value in EXECUTION_BACKENDS_BY_KIND[kind_value] else "other",
        "error_class": "none" if failure is None else failure if failure in TOOL_ERROR_CLASSES else "unknown",
        "kind": kind_value,
        "outcome": "success" if failure is None else "failed",
    }


# Set while Hermes itself drives a backend (TUI/Desktop path completion listings): that is not the
# user's workload, so it must not show up as backend usage.
_UNMETERED = contextvars.ContextVar("shared_metrics_unmetered_backend", default=False)


@contextlib.contextmanager
def unmetered_backend_calls() -> Iterator[None]:
    """Run Hermes-owned terminal/browser/code calls without counting them as execution-backend use."""
    token = _UNMETERED.set(True)
    try:
        yield
    finally:
        _UNMETERED.reset(token)


def record_execution_backend(kind: str, backend: Any, result: Any = None, *, error_class: str | None = None) -> Any:
    """Count one tool call that reached its backend; returns ``result`` unchanged. ``backend`` may be a
    callable so resolving it costs nothing while collection is off. Calls the background review /
    curator forks make are Hermes' own work, not the user's (terminal.outcome skips them too)."""
    from tools.skill_provenance import is_background_review

    if _UNMETERED.get() or is_background_review():
        return result
    _emit(
        "EXECUTION_BACKEND_MARK", execution_backend_fields, kind=kind, backend=backend, result=result,
        error_class=error_class,
    )
    return result


def record_terminal_backend(plan: Any, result: Any, *, error_class: str | None = None) -> Any:
    """Terminal calls that got past planning; ``plan`` is None when planning itself failed."""
    if plan is None:
        return result
    return record_execution_backend("terminal", getattr(plan, "env_type", None), result, error_class=error_class)


def record_browser_call(call: Callable[[Callable[[Callable[[], Any]], Any]], Any], resolve_backend: Callable[[], str]) -> Any:
    """Run one browser tool call and count it against the backend that served it. ``call`` receives a
    wrapper for the legacy-backend fallback: when the extension controller serves the call instead,
    the fallback never runs and the backend is ``extension``."""
    served_by_legacy = []

    def backend() -> str:
        return resolve_backend() if served_by_legacy else "extension"

    def legacy(fallback: Callable[[], Any]) -> Any:
        served_by_legacy.append(True)
        return fallback()

    try:
        result = call(legacy)
    except Exception:
        _safe_record_browser(backend, None, "exception")
        raise
    _safe_record_browser(backend, result, None)
    return result


def _safe_record_browser(backend: Callable[[], str], result: Any, error_class: str | None) -> None:
    def name() -> str:
        try:
            return backend()
        except Exception:
            return "other"

    record_execution_backend("browser", name, result, error_class=error_class)
