"""Generation-aware recovery for failed file mutations.

The turn-end verifier normally keeps failed ``write_file``/``patch`` calls
visible.  A later foreground ``terminal`` or ``execute_code`` call may be a
legitimate fallback, but it only clears a failure when content on the exact
backend target changes during that successful call's execution window.
"""

from __future__ import annotations

import json
import logging
import ntpath
import os
import threading
import time
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from agent.tool_dispatch_helpers import (
    _extract_error_preview,
    _extract_file_mutation_targets,
    _extract_landed_file_mutation_paths,
)
from agent.tool_result_classification import file_mutation_result_landed

logger = logging.getLogger(__name__)

_FILE_MUTATING_TOOLS = frozenset({"write_file", "patch"})
_EXTERNAL_FILE_CAPABLE_TOOLS = frozenset({"terminal", "execute_code"})


@dataclass(frozen=True)
class FileContentFingerprint:
    """A reliable existence/content token; metadata is intentionally absent."""

    kind: str
    sha256: str | None = None


@dataclass(frozen=True)
class MutationTargetIdentity:
    """The target identity captured by the mutation tool's own resolver."""

    raw_path: str
    resolved_path: str | None
    task_id: str
    defer_fingerprint: bool = False


@dataclass(frozen=True)
class _FallbackProbe:
    state_key: str
    resolved_path: str
    task_id: str
    generation: int
    before: FileContentFingerprint


@dataclass(frozen=True)
class _VerifierCallContext:
    """Exact per-turn state and revocation token captured before a tool call."""

    epoch: object
    state: dict[str, dict[str, Any]] | None
    lock: threading.Lock
    changed_paths: set[str] | None
    active_event: threading.Event
    deadline: float | None
    cancel_check: Callable[[], bool] | None


_mutation_targets: ContextVar[tuple[MutationTargetIdentity, ...]] = ContextVar(
    "file_mutation_verifier_targets",
    default=(),
)
_fallback_probes: ContextVar[tuple[_FallbackProbe, ...]] = ContextVar(
    "file_mutation_verifier_fallback_probes",
    default=(),
)
_call_context: ContextVar[_VerifierCallContext | None] = ContextVar(
    "file_mutation_verifier_call_context",
    default=None,
)
_pending_probe_activation: ContextVar[Callable[[], None] | None] = ContextVar(
    "file_mutation_verifier_pending_probe_activation",
    default=None,
)
_lock_creation_lock = threading.Lock()


def activate_pending_file_mutation_verifier_call() -> None:
    """Arm deferred fallback probes after the tool's execution gate approves."""

    activation = _pending_probe_activation.get()
    _pending_probe_activation.set(None)
    if activation is not None:
        activation()


def publish_mutation_target_identities(
    targets: Sequence[MutationTargetIdentity],
) -> None:
    """Publish identities from a file tool for same-thread result recording."""

    _mutation_targets.set(tuple(targets))


def _consume_mutation_target_identities() -> tuple[MutationTargetIdentity, ...]:
    targets = _mutation_targets.get()
    _mutation_targets.set(())
    return targets


def _consume_fallback_probes() -> tuple[_FallbackProbe, ...]:
    probes = _fallback_probes.get()
    _fallback_probes.set(())
    return probes


def _consume_call_context() -> _VerifierCallContext | None:
    context = _call_context.get()
    _call_context.set(None)
    return context


def _turn_epoch(agent: Any) -> object:
    epoch = getattr(agent, "_turn_file_mutation_epoch", None)
    if epoch is not None:
        return epoch
    with _lock_creation_lock:
        epoch = getattr(agent, "_turn_file_mutation_epoch", None)
        if epoch is None:
            epoch = object()
            agent._turn_file_mutation_epoch = epoch
    return epoch


def _call_context_is_current(
    agent: Any,
    context: _VerifierCallContext | None,
) -> bool:
    if context is None or not context.active_event.is_set():
        return False
    if context.deadline is not None and time.monotonic() >= context.deadline:
        return False
    if context.cancel_check is not None:
        try:
            if context.cancel_check():
                return False
        except Exception:
            # A broken cancellation predicate must never make stale evidence
            # eligible to clear an actionable mutation failure.
            return False
    return bool(
        getattr(agent, "_turn_file_mutation_epoch", None) is context.epoch
        and getattr(agent, "_turn_failed_file_mutations", None) is context.state
        and getattr(agent, "_turn_file_mutation_lock", None) is context.lock
        and getattr(agent, "_turn_file_mutation_paths", None) is context.changed_paths
    )


def _state_lock(agent: Any) -> threading.Lock:
    lock = getattr(agent, "_turn_file_mutation_lock", None)
    if lock is not None:
        return lock
    # Tool dispatch outside a normal conversation turn is unusual but valid in
    # tests/plugins.  Avoid racing two lazy lock creations in that path.
    with _lock_creation_lock:
        lock = getattr(agent, "_turn_file_mutation_lock", None)
        if lock is None:
            lock = threading.Lock()
            agent._turn_file_mutation_lock = lock
    return lock


def _fingerprint(
    resolved_path: str,
    task_id: str,
) -> FileContentFingerprint | None:
    """Read a content token through the same backend selected by file tools."""

    try:
        from tools.file_tools import _fingerprint_resolved_file_content

        return _fingerprint_resolved_file_content(resolved_path, task_id)
    except Exception as exc:
        logger.debug(
            "file-mutation verifier fingerprint failed for %r: %s",
            resolved_path,
            exc,
        )
        return None


def _is_external_fallback(tool_name: str, args: Mapping[str, Any]) -> bool:
    if tool_name not in _EXTERNAL_FILE_CAPABLE_TOOLS:
        return False
    # A successful background terminal result proves only that a process was
    # launched, not that the process completed its mutation. Treat any truthy
    # value conservatively because direct callers are not guaranteed to enforce
    # the JSON-schema boolean type before dispatch.
    return tool_name != "terminal" or not bool(args.get("background", False))


def prepare_file_mutation_verifier_call(
    agent: Any,
    tool_name: str,
    args: Mapping[str, Any],
    effective_task_id: str,
    *,
    active_event: threading.Event | None = None,
    deadline: float | None = None,
    cancel_check: Callable[[], bool] | None = None,
    defer_probe_activation: bool = False,
) -> threading.Event:
    """Reset call-local evidence and snapshot eligible fallback baselines."""

    _mutation_targets.set(())
    _fallback_probes.set(())
    _pending_probe_activation.set(None)
    if active_event is None:
        active_event = threading.Event()
        active_event.set()
    lock = _state_lock(agent)
    state = getattr(agent, "_turn_failed_file_mutations", None)
    context = _VerifierCallContext(
        epoch=_turn_epoch(agent),
        state=state,
        lock=lock,
        changed_paths=getattr(agent, "_turn_file_mutation_paths", None),
        active_event=active_event,
        deadline=deadline,
        cancel_check=cancel_check,
    )
    _call_context.set(context)

    if not _call_context_is_current(agent, context):
        return active_event
    if not _is_external_fallback(tool_name, args) or not state:
        return active_event

    with lock:
        if not _call_context_is_current(agent, context):
            return active_event
        candidates = []
        for key, info in state.items():
            resolved_path = info.get("resolved_path")
            baseline = info.get("fingerprint")
            generation = info.get("generation")
            task_id = info.get("task_id")
            deferred = info.get("fingerprint_deferred") is True
            if (
                isinstance(resolved_path, str)
                and resolved_path
                and isinstance(generation, int)
                and task_id == effective_task_id
                and (
                    isinstance(baseline, FileContentFingerprint)
                    or (baseline is None and deferred)
                )
            ):
                candidates.append(
                    (key, resolved_path, task_id, generation, baseline, deferred)
                )

    def activate_probes() -> None:
        probes: list[_FallbackProbe] = []
        for key, resolved_path, task_id, generation, baseline, deferred in candidates:
            if not _call_context_is_current(agent, context):
                return
            before = _fingerprint(resolved_path, task_id)
            if before is None:
                continue
            if baseline is not None and before != baseline:
                # Content changed before this fallback began.  It may have been
                # an unrelated watcher; do not attribute that change to this call.
                continue
            with lock:
                if not _call_context_is_current(agent, context):
                    return
                current = state.get(key)
                if not current or current.get("generation") != generation:
                    continue
                current_baseline = current.get("fingerprint")
                if deferred:
                    if current_baseline is not None or not current.get(
                        "fingerprint_deferred"
                    ):
                        continue
                    # The native policy gate captured exact identity without
                    # reading content. Establish the baseline only after the
                    # fallback's own execution gate has approved it.
                    current["fingerprint"] = before
                    current["fingerprint_deferred"] = False
                elif current_baseline != baseline:
                    continue
            probes.append(
                _FallbackProbe(
                    state_key=key,
                    resolved_path=resolved_path,
                    task_id=task_id,
                    generation=generation,
                    before=before,
                )
            )

        if _call_context_is_current(agent, context):
            _fallback_probes.set(tuple(probes))

    if defer_probe_activation:
        _pending_probe_activation.set(activate_probes)
    else:
        activate_probes()
    return active_event


def _canonical_reported_path(path: str) -> bool:
    """Return whether a tool-reported path is already backend-absolute."""

    return os.path.isabs(path) or ntpath.isabs(path)


def _state_key(identity: str, task_id: str) -> str:
    """Keep identical path strings isolated across backend task contexts."""

    if not task_id or task_id == "default":
        return identity
    return f"{task_id}\0{identity}"


def _record_failed_mutation(
    agent: Any,
    tool_name: str,
    args: Mapping[str, Any],
    result: Any,
    captured: Sequence[MutationTargetIdentity],
    context: _VerifierCallContext,
) -> None:
    state = context.state
    lock = context.lock
    if state is None or not _call_context_is_current(agent, context):
        return
    raw_targets = _extract_file_mutation_targets(tool_name, dict(args))
    captured_by_raw: dict[str, list[MutationTargetIdentity]] = {}
    for target in captured:
        captured_by_raw.setdefault(target.raw_path, []).append(target)

    targets: list[MutationTargetIdentity] = []
    for raw_path in raw_targets:
        matches = captured_by_raw.get(raw_path)
        if matches:
            targets.append(matches.pop(0))
        else:
            # Never re-resolve here: the mutation tool did not provide a
            # trustworthy identity, so this generation is intentionally
            # unverifiable and its warning cannot be externally suppressed.
            targets.append(MutationTargetIdentity(raw_path, None, ""))

    if not targets:
        return

    preview = _extract_error_preview(result)
    with lock:
        if not _call_context_is_current(agent, context):
            return
        generation = getattr(agent, "_turn_file_mutation_generation", 0) + 1
        agent._turn_file_mutation_generation = generation
        for target in targets:
            identity = target.resolved_path or target.raw_path
            key = _state_key(identity, target.task_id)
            previous = state.get(key)
            if previous is None and target.resolved_path:
                unresolved_key = _state_key(target.raw_path, target.task_id)
                previous = state.pop(unresolved_key, None)
            state[key] = {
                "tool": (previous or {}).get("tool") or tool_name,
                "error_preview": (previous or {}).get("error_preview") or preview,
                "display_path": target.resolved_path or target.raw_path,
                "resolved_path": target.resolved_path,
                "task_id": target.task_id,
                "generation": generation,
                "fingerprint": None,
                "fingerprint_deferred": target.defer_fingerprint,
            }

    # Fingerprints may require a backend round-trip.  Do it without holding
    # the state lock, then publish only if this generation is still current.
    for target in targets:
        if not target.resolved_path or target.defer_fingerprint:
            continue
        fingerprint = _fingerprint(target.resolved_path, target.task_id)
        key = _state_key(target.resolved_path, target.task_id)
        with lock:
            if not _call_context_is_current(agent, context):
                return
            current = state.get(key)
            if current and current.get("generation") == generation:
                current["fingerprint"] = fingerprint


def _clear_native_success(
    agent: Any,
    tool_name: str,
    args: Mapping[str, Any],
    result: Any,
    captured: Sequence[MutationTargetIdentity],
    landed: bool,
    context: _VerifierCallContext,
) -> None:
    state = context.state
    lock = context.lock
    if state is None or not _call_context_is_current(agent, context):
        return
    raw_targets = _extract_file_mutation_targets(tool_name, dict(args))
    landed_paths = (
        _extract_landed_file_mutation_paths(tool_name, dict(args), result)
        if landed
        else []
    )

    keys: set[str] = set()
    task_ids: set[str] = set()
    if captured:
        for target in captured:
            task_ids.add(target.task_id)
            keys.add(_state_key(target.raw_path, target.task_id))
            if target.resolved_path:
                keys.add(_state_key(target.resolved_path, target.task_id))
    else:
        # Backward-compatible direct/test calls have no resolver evidence and
        # therefore can only address the historical default-task keys.
        keys.update(raw_targets)

    canonical_paths: set[str] = set()
    for path in landed_paths:
        if task_ids:
            keys.update(_state_key(path, task_id) for task_id in task_ids)
        else:
            keys.add(path)
        if _canonical_reported_path(path):
            canonical_paths.add(path)

    with lock:
        if not _call_context_is_current(agent, context):
            return
        for key in keys:
            state.pop(key, None)
        if context.changed_paths is not None:
            context.changed_paths.update(canonical_paths)


def _external_fallback_result_succeeded(
    tool_name: str,
    result: Any,
    is_error: bool,
) -> bool:
    """Require each external tool's structured success contract."""

    if is_error or not isinstance(result, str):
        return False
    try:
        payload = json.loads(result)
    except (TypeError, ValueError):
        return False
    if not isinstance(payload, dict) or payload.get("error"):
        return False
    if tool_name == "terminal":
        exit_code = payload.get("exit_code")
        return type(exit_code) is int and exit_code == 0
    if tool_name == "execute_code":
        return payload.get("status") == "success"
    return False


def _finish_external_fallback(
    agent: Any,
    tool_name: str,
    result: Any,
    is_error: bool,
    context: _VerifierCallContext | None,
) -> None:
    probes = _consume_fallback_probes()
    if (
        not probes
        or not _external_fallback_result_succeeded(tool_name, result, is_error)
        or not _call_context_is_current(agent, context)
        or context is None
        or context.state is None
    ):
        return
    state = context.state
    lock = context.lock

    for probe in probes:
        if not _call_context_is_current(agent, context):
            return
        after = _fingerprint(probe.resolved_path, probe.task_id)
        if after is None or after == probe.before:
            continue
        with lock:
            if not _call_context_is_current(agent, context):
                return
            current = state.get(probe.state_key)
            if not current or current.get("generation") != probe.generation:
                continue
            if current.get("task_id") != probe.task_id:
                continue
            if current.get("fingerprint") != probe.before:
                continue
            state.pop(probe.state_key, None)
            if context.changed_paths is not None:
                context.changed_paths.add(probe.resolved_path)


def record_file_mutation_verifier_result(
    agent: Any,
    tool_name: str,
    args: Mapping[str, Any],
    result: Any,
    is_error: bool,
) -> None:
    """Record completion evidence for native mutations or external fallbacks."""

    context = _consume_call_context()
    if tool_name in _FILE_MUTATING_TOOLS:
        captured = _consume_mutation_target_identities()
        _fallback_probes.set(())
        if (
            context is None
            or context.state is None
            or not _call_context_is_current(agent, context)
        ):
            return
        landed = file_mutation_result_landed(tool_name, result)
        if landed:
            _clear_native_success(
                agent,
                tool_name,
                args,
                result,
                captured,
                landed=True,
                context=context,
            )
        else:
            # Dispatcher error classification is advisory; only the tool's
            # landed-result contract proves that a native mutation succeeded.
            # A malformed/ambiguous result must retain (or create) the warning
            # rather than silently clearing an earlier actionable failure.
            _record_failed_mutation(
                agent,
                tool_name,
                args,
                result,
                captured,
                context,
            )
        return

    _mutation_targets.set(())
    if tool_name in _EXTERNAL_FILE_CAPABLE_TOOLS:
        _finish_external_fallback(
            agent,
            tool_name,
            result,
            is_error,
            context,
        )
    else:
        _fallback_probes.set(())
