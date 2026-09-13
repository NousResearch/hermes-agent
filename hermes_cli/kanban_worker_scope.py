"""Kanban worker lifecycle; integrated from PR #101911 by danashburn.

Attempt ownership and persistence use the current main Kanban modules.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import re
import secrets
import shutil
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional

_log = logging.getLogger(__name__)


_KANBAN_WORKER_ISOLATION_MODES = ("auto", "systemd-scope", "none")


_SCOPE_UNIT_UNSAFE = re.compile(r"[^A-Za-z0-9_.\-]")


_scope_fallback_warned = False


_memory_bound_omitted_warned = False


WORKER_REGISTRATION_GRACE_SECONDS = 120


WORKER_SPAWN_PROBE_SECONDS = 1.5


def _worker_spawn_probe_seconds() -> float:
    """The launch-probe window, overridable via
    ``HERMES_KANBAN_SPAWN_PROBE_SECONDS`` (tests; also a support escape
    hatch for slow user buses)."""
    raw = os.environ.get("HERMES_KANBAN_SPAWN_PROBE_SECONDS", "").strip()
    if raw:
        try:
            value = float(raw)
            if value > 0:
                return value
        except ValueError:
            pass
    return float(WORKER_SPAWN_PROBE_SECONDS)


def _kanban_cfg_from_config() -> dict:
    """Load the ``kanban:`` config block, tolerating any load failure."""
    try:
        from hermes_cli.config import load_config

        return load_config().get("kanban") or {}
    except Exception:
        return {}


def _resolve_worker_isolation(kanban_cfg: Optional[dict] = None) -> str:
    """Normalise ``kanban.worker_isolation`` to one of the valid modes.

    Unknown values warn once and behave as ``auto`` — a typo must not
    silently disable isolation or break spawning.
    """
    cfg = _kanban_cfg_from_config() if kanban_cfg is None else (kanban_cfg or {})
    raw = str(cfg.get("worker_isolation") or "auto").strip().lower()
    if raw not in _KANBAN_WORKER_ISOLATION_MODES:
        _log.warning(
            "kanban: unknown worker_isolation %r (valid: %s) — using 'auto'",
            cfg.get("worker_isolation"),
            ", ".join(_KANBAN_WORKER_ISOLATION_MODES),
        )
        return "auto"
    return raw


def _kanban_worker_scope_enabled(kanban_cfg: Optional[dict] = None) -> bool:
    """True when this dispatcher should spawn workers in their own scope.

    One decision point shared by the spawn path (``_default_spawn`` wraps
    the argv) and the bookkeeping path (``dispatch_once`` records the unit
    name), so the two can never disagree. Resolution:

    * ``none``  — never; kanban adds no scope of its own, so workers spawn
      with the legacy argv and process-group kill, the required no-op on
      macOS/containers. On a systemd-MANAGED gateway the shared
      restart-safe wrap still applies underneath (a plain child would die
      with the service cgroup on the next gateway restart), so the worker
      runs in ``hermes-worker-kanban-<task>-run-<n>.scope``; that unit IS
      recorded in ``tasks.worker_scope`` and swept by the scope audit —
      ``none`` turns off kanban's own isolation, not the restart-safe
      guarantee (see :func:`_managed_gateway_dispatch`).
      Because such a run now starts with a recorded scope, it also starts
      UNREGISTERED, and the launch-grace rule that already covers scoped
      runs covers it too: a worker with no tool activity within
      ``WORKER_REGISTRATION_GRACE_SECONDS`` (120 s) is stopped and its run
      recorded ``spawn_failed``. That is deliberate — one rule for every
      run that owns a scope, whatever mode created it — and a slow boot is
      not lost to it, because the queued stop re-checks registration
      immediately before it signals and stands down if the worker has
      since registered.
    * ``systemd-scope`` — scope only when the real
      ``systemd-run --user --scope`` probe passes; on failure warn once
      and REFUSE the spawn (finding H): the spawn path raises so the
      dispatcher records spawn_failed — an operator who pinned strict
      chose "no worker" over "unisolated worker". Only auto may fall
      back to the unisolated spawn.
    * ``auto`` (default) — scope when the probe passes, silently fall
      back otherwise.

    The probe is the shared cached one from ``tools.process_registry``
    (real no-op scope, D-Bus connectivity included), so kanban and the
    terminal tool never maintain two notions of "systemd works here".
    """
    mode = _resolve_worker_isolation(kanban_cfg)
    if mode == "none":
        return False
    from tools.process_registry_scope import _systemd_run_user_scope_available

    available = _systemd_run_user_scope_available()
    if mode == "systemd-scope" and not available:
        global _scope_fallback_warned
        if not _scope_fallback_warned:
            _scope_fallback_warned = True
            _log.warning(
                "kanban: worker_isolation=systemd-scope requested but "
                "systemd-run --user --scope is unavailable on this host — "
                "worker spawns will be refused (spawn_failed); set "
                "worker_isolation: auto to allow unisolated fallback"
            )
    return available


def _managed_gateway_dispatch() -> bool:
    """True when this dispatcher's children get the restart-safe scope wrap.

    Mirrors the topology gate inside
    ``tools.process_registry.restart_safe_gateway_child_argv`` exactly: on a
    systemd-MANAGED gateway every worker is wrapped in
    ``hermes-worker-kanban-<task>-run-<n>.scope`` regardless of the kanban
    ``worker_isolation`` setting, because a plain child would be killed with
    the service cgroup on the next gateway restart.

    Two decisions read this: there is no unisolated fallback to degrade to on
    such a host (a refused isolation launch fails closed instead of "retrying
    without isolation" straight back into systemd-run), and the scope audit
    must sweep the restart-safe unit names even when isolation is ``none``.
    """
    from tools.process_registry_scope import _IS_LINUX, _is_supervised_gateway_process

    return bool(
        _IS_LINUX
        and _is_supervised_gateway_process()
        and os.environ.get("INVOCATION_ID")
    )


def _scope_unit_from_argv(argv: list[str]) -> str:
    """Bus unit name of a ``systemd-run --scope`` argv, ``""`` when absent.

    Read back out of the argv the launch actually uses rather than rebuilt
    from the naming rule, so a recorded unit can never drift from the one
    systemd creates. ``systemd-run`` appends the ``.scope`` suffix when
    ``--unit`` omits it (which the restart-safe wrap does).
    """
    try:
        unit = argv[argv.index("--unit") + 1]
    except (ValueError, IndexError):
        return ""
    if not unit:
        return ""
    return unit if unit.endswith(".scope") else f"{unit}.scope"


def _kanban_worker_scope_unit(
    task_id: str,
    run_id: Optional[int],
    *,
    board: Optional[str] = None,
    db_path: Optional[str] = None,
) -> str:
    """Transient unit name for one ATTEMPT of a task's worker.

    ``hermes-kanban-<task_id>-r<run_id>.scope`` — unique per attempt, so a
    retry spawn can never collide with a lingering scope from the attempt
    it replaced (``systemd-run`` refuses to create a unit that already
    exists, which would turn a slow-to-die old scope into a spawn
    failure). The attempt's unit is recorded in ``tasks.worker_scope`` /
    ``task_runs.worker_scope``, so a re-adopted worker's scope stays
    addressable by the same ``systemctl --user stop`` call across gateway
    restarts. Task ids are ``t_`` + hex, already unit-safe; the sanitiser
    is defence for any future id scheme.
    """
    safe = _scope_task_key(task_id, board=board, db_path=db_path)
    run_part = f"-r{int(run_id)}" if run_id is not None else ""
    return f"hermes-kanban-{safe}{run_part}.scope"


def _task_id_from_kanban_scope_unit(unit: str) -> Optional[str]:
    """Inverse of :func:`_kanban_worker_scope_unit` for audit sweeps.

    Returns the sanitised task id encoded in a kanban scope unit name
    (``hermes-kanban-<task>-r<run>.scope`` or the legacy attempt-free
    ``hermes-kanban-<task>.scope``), or ``None`` for anything else. The
    returned id is the SANITISED form — callers must treat a lookup miss
    as "no matching task", never fall back to fuzzy matching.

    Also parses the restart-safe wrap's own naming,
    ``hermes-worker-kanban-<task>-run-<run>.scope``: on a systemd-managed
    gateway that unit — not the isolation one — is what a worker spawned
    with ``worker_isolation: none`` actually runs in, and the audit sweep
    lists it alongside the isolation units.
    """
    key = _scope_identity_key_from_unit(unit)
    return re.sub(r"^b[0-9a-f]{24}--", "", key) if key else None


def _scope_identity_key_from_unit(unit: str) -> Optional[str]:
    m = re.match(
        r"^hermes-(?:worker-)?kanban-(?P<tid>.+?)"
        r"(?:-r\d+|-run-\d+)?\.scope$",
        unit,
    )
    return m.group("tid") if m else None


def _kanban_worker_memory_bytes(
    kanban_cfg: Optional[dict] = None,
) -> Optional[int]:
    """Per-worker cgroup memory bound used for BOTH MemoryMax and
    MemorySwapMax.

    Resolution: explicit ``kanban.worker_memory_max_mb`` (clamped to the
    shared 64 MiB floor / 4 GiB cap), else the shared default from
    ``tools.process_registry._worker_memory_max_bytes`` (the tighter of
    the gateway cgroup limit and half the host RAM). Returns ``None``
    only when no positive bound can be computed — the spawn path then
    OMITS both memory properties rather than emitting a broken
    ``MemoryMax=0`` (0 means "no limit" in systemd, the opposite of the
    intended bound).
    """
    cfg = _kanban_cfg_from_config() if kanban_cfg is None else (kanban_cfg or {})
    raw = cfg.get("worker_memory_max_mb")
    if raw is None or raw == "":
        from tools.process_registry_scope import _worker_memory_max_bytes

        resolved = _worker_memory_max_bytes()
        if not resolved or resolved <= 0:
            global _memory_bound_omitted_warned
            if not _memory_bound_omitted_warned:
                _memory_bound_omitted_warned = True
                _log.warning(
                    "kanban: no computable per-worker memory bound "
                    "(helper returned %r) — spawning workers WITHOUT "
                    "MemoryMax/MemorySwapMax",
                    resolved,
                )
            return None
        return resolved
    try:
        mb = int(raw)
    except (TypeError, ValueError):
        _log.warning(
            "kanban: worker_memory_max_mb=%r is not an integer — using the default",
            raw,
        )
        return _kanban_worker_memory_bytes({})
    from tools.process_registry_scope import (
        _MIN_WORKER_MEMORY_MAX_BYTES,
        _WORKER_MEMORY_MAX_CAP_BYTES,
    )

    bytes_ = mb * 1024 * 1024
    if bytes_ < _MIN_WORKER_MEMORY_MAX_BYTES:
        _log.warning(
            "kanban: worker_memory_max_mb=%s below the %d MiB floor — clamping",
            mb,
            _MIN_WORKER_MEMORY_MAX_BYTES // (1024 * 1024),
        )
        return _MIN_WORKER_MEMORY_MAX_BYTES
    return min(bytes_, _WORKER_MEMORY_MAX_CAP_BYTES)


def _kanban_scope_state(unit_name: Optional[str]) -> str:
    """``"active"`` / ``"dead"`` / ``"unknown"`` / ``"unsupported"``.

    Wraps ``tools.process_registry._scope_unit_active_state``; for scopes
    ``active`` means the unit's cgroup still holds at least one process —
    the authoritative "is anything of this worker alive" signal, immune
    to both PID reuse and the launcher-vs-worker split.
    ``unsupported`` is DEFINITE: the host exposes no readable cgroup
    hierarchy (v2 or v1) at all, so cgroup verification is impossible by
    construction — callers treat the run as NOT isolated and fall back
    to PID semantics rather than retrying ``unknown`` forever (Gate B
    pass 4, S).
    """
    if not unit_name:
        return "dead"
    try:
        from tools.process_registry_scope import _scope_unit_active_state

        return _scope_unit_active_state(unit_name)
    except Exception as exc:
        _log.debug("kanban: scope state probe %s failed: %s", unit_name, exc)
        return "unknown"


def _scope_unit_created(unit_name: Optional[str]) -> bool:
    """True if the transient unit exists on the user bus.

    Wraps ``tools.process_registry._scope_unit_was_created``: a probe
    failure counts as created — the cost of that wrong assumption is a
    duplicate spawn (the review's named failure mode), while the cost of
    the opposite assumption on a live unit is a plain-spawned worker
    beside an untracked scoped one."""
    if not unit_name:
        return False
    try:
        from tools.process_registry_scope import _scope_unit_was_created

        return _scope_unit_was_created(unit_name)
    except Exception as exc:
        _log.debug("kanban: unit-created probe %s failed: %s", unit_name, exc)
        return True


def _kanban_list_scope_units(pattern: str) -> dict:
    """``{unit_name: active_state}`` for kanban-style scopes on the bus.

    Wraps ``tools.process_registry._list_systemd_scope_units``; empty on
    any failure (the audit sweep treats "cannot list" as "nothing to
    reap" and retries next tick).
    """
    try:
        from tools.process_registry_scope import _list_systemd_scope_units

        return _list_systemd_scope_units(pattern)
    except Exception as exc:
        _log.debug("kanban: scope unit listing failed: %s", exc)
        return {}


def _kanban_scope_is_live(state: str) -> bool:
    """True when a scope state means "cgroup still holds a process"."""
    from tools.process_registry_scope import _SCOPE_ACTIVE_STATES

    return state in _SCOPE_ACTIVE_STATES


def _collect_kanban_scope(unit_name: Optional[str]) -> None:
    """Unload a verified-empty scope unit (explicit ``--collect``).

    Kanban scopes are spawned without ``--collect`` so a fast nonzero
    exit stays inspectable (see ``_default_spawn``); every path that
    CONFIRMS a unit dead collects it here instead, so dead scopes do
    not accumulate on the user bus.  Best-effort: a unit that already
    unloaded is a no-op, a failure retries via the next confirmed-dead
    observation."""
    if not unit_name:
        return
    try:
        from tools.process_registry_scope import _collect_dead_systemd_unit

        _collect_dead_systemd_unit(unit_name)
    except Exception as exc:
        _log.debug("kanban: collect of scope %s failed: %s", unit_name, exc)


def _stop_kanban_worker_scope(
    unit_name: Optional[str],
    *,
    cancel_event: Optional[threading.Event] = None,
    deadline: Optional[float] = None,
) -> bool:
    """Stop a worker's transient scope — the whole cgroup, not one pid —
    and VERIFY it is actually dead.

    ``systemctl --user stop`` SIGTERMs every process in the unit;
    ``_stop_systemd_unit_verified`` then escalates to
    ``systemctl kill --signal=SIGKILL`` and confirms the unit went
    inactive, because a scope stays active while ANY descendant (dev
    server, browser, database) lives and ``--collect`` alone only unloads
    an already-empty unit. Returns True when the unit is verified gone
    (or was never there); False means "still stopping" — callers that
    guard against duplicate spawns must NOT release their claim on False,
    they defer and retry next tick. Stopping an already-dead unit is a
    verified no-op (True), so this is safe to call speculatively from
    every terminal path.

    ``cancel_event``/``deadline`` (pass 8, Y) reach the escalation steps
    themselves: an in-flight stop checks them after the TERM wait,
    before the SIGKILL wait, and before each final-verify probe, so a
    shutdown whose budget expired cancels the stop it is inside of, not
    just the ones it has not started. Cancelled stops return False
    ("still stopping") like any unverified one.
    """
    if not unit_name:
        return False
    try:
        from tools.process_registry_scope import _stop_systemd_unit_verified

        return _stop_systemd_unit_verified(
            unit_name,
            cancel_event=cancel_event,
            deadline=deadline,
        )
    except Exception as exc:
        _log.warning(
            "kanban: failed to stop worker scope %s: %s",
            unit_name,
            exc,
        )
        return False


def _scope_task_key(
    task_id: str, *, board: Optional[str] = None, db_path: Optional[str] = None
) -> str:
    """Qualify host-global unit identity by canonical board storage."""
    from hermes_cli.kanban_db import kanban_db_path

    path = Path(db_path or kanban_db_path(board=board)).resolve()
    digest = hashlib.sha256(os.path.normcase(str(path)).encode()).hexdigest()[:24]
    safe = _SCOPE_UNIT_UNSAFE.sub("-", str(task_id)).strip("-.") or "task"
    return f"b{digest}--{safe}"
