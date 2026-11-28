"""Foreground default-plugins preflight (``ensure``), detached child handoff, and CLI entrypoint.

Split out of :mod:`hermes_cli.default_plugins` so the core installer/policy module and the
bounded preflight/CLI runner each stay focused and compact. Callers continue to import
``ensure_defaults``, ``main``, and the ``ENSURE_EXIT_*`` constants from ``hermes_cli.default_plugins``.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

ENSURE_EXIT_OK = 0
ENSURE_EXIT_OWED = 2
ENSURE_EXIT_TIMEOUT = 3
ENSURE_EXIT_SKIPPED = 4
DEFAULT_ENSURE_TIMEOUT = 300.0
PROFILE_CREATE_ENSURE_TIMEOUT = 120.0
_CHILD_LOCK_WAIT = 900.0  # an ensure child waits out another installer instead of exiting with nothing done
_RESULT_MAX_AGE = 3600.0  # prune orphaned per-run result files older than 1 hour


def _launch_ensure_child(run_id: str, home: Optional[Path]) -> subprocess.Popen:
    """The detached child an ``ensure`` waits on: it discovers/installs (for *home* only, when given,
    else every home), then writes its outcome under *run_id* to ``cache/default-plugins.result-<run_id>.json``."""
    from hermes_cli import default_plugins as _dp

    args: list[str] = ["_child", "--run-id", run_id]
    if home is not None:
        args += ["--home", str(home)]
    return _dp._popen_detached(_dp._child_argv(tuple(args)))


def _result_path(root: Path, run_id: str) -> Path:
    from hermes_cli import default_plugins as _dp

    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(run_id)) or "run"
    return _dp._state(root, f"result-{safe}.json")


def _cleanup_stale_results(root: Path, *, max_age: float = _RESULT_MAX_AGE) -> None:
    from hermes_cli import default_plugins as _dp

    cache_dir = _dp._state(root, "result").parent
    cutoff = time.time() - max_age
    try:
        for candidate in cache_dir.glob("default-plugins.result-*.json"):
            try:
                if candidate.stat().st_mtime < cutoff:
                    candidate.unlink(missing_ok=True)
            except OSError:
                pass
    except OSError:
        pass


def _write_child_result(root: Path, run_id: str, outcome: dict) -> None:
    from hermes_cli import default_plugins as _dp
    from utils import atomic_write_text

    path = _dp._result_path(root, run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    _cleanup_stale_results(root)
    payload = {
        "run_id": run_id,
        "installed": [str(x) for x in outcome.get("installed") or ()],
        "owed": [str(x) for x in outcome.get("owed") or ()],
    }
    atomic_write_text(path, json.dumps(payload), tmp_prefix=f"{path.name}.tmp-")


def _read_child_result(root: Path, run_id: str) -> Optional[dict]:
    """The result *run_id*'s child wrote, else None (none yet, unreadable, or another run's).
    Unlinks *run_id*'s result file once read so completed runs do not accumulate in cache."""
    from hermes_cli import default_plugins as _dp

    path = _dp._result_path(root, run_id)
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or data.get("run_id") != run_id:
        return None
    path.unlink(missing_ok=True)
    return {key: [str(x) for x in data.get(key) or ()] for key in ("installed", "owed")}


def _install_subset_and_record(
    target_homes: list[Path],
    entries: list,
    root: Path,
    *,
    say: Callable[[str], None],
    wait: float,
) -> dict:
    from hermes_cli import default_plugins as _dp
    from hermes_cli.memory_provider_migration import _note_failure

    outcome = _dp._install_defaults_for_homes(target_homes, say=say, wait=wait, entries=entries)
    if outcome["owed"]:
        failed = set(outcome["owed"]) - set(outcome.get("raced", ()))
        if failed:
            _note_failure(root, _dp._BACKOFF)
        _dp._state(root, "settled").unlink(missing_ok=True)
    return outcome


def _ensure_child_work(run_id: str, home: Optional[Path], *, say: Callable[[str], None]) -> int:
    """The ensure child's pass (every home, or *home* only), its outcome written under *run_id* whatever
    happens short of the process dying (that leaves no result: the waiting ensure reports it owed).

    Catalog discovery and offline fail-open run here so the parent's ``--timeout`` bounds slow
    catalog responses as well as git/uv installs.
    """
    from hermes_cli import default_plugins as _dp
    from hermes_cli.memory_provider_migration import _note_failure
    from pm.environments import dependency_home_root
    from pm.plugins_state import dependency_homes

    root = Path(dependency_home_root())
    outcome: dict = {"installed": [], "owed": []}
    try:
        target_homes = list(dependency_homes()) if home is None else [Path(home)]
        policies = {h: _dp.fetch_policy(h) for h in target_homes}
        allowed_homes = [h for h, policy in policies.items() if policy == "allowed"]
        unreadable_homes = [str(h) for h, policy in policies.items() if policy == "unreadable"]
        entries = _dp.default_entries() if allowed_homes else []
        pending = _dp._pending_names(allowed_homes, entries)
        if allowed_homes and _dp._live_catalog_unreachable():
            # Offline: the live catalog read just failed, so every git/uv install would fail too,
            # each after its own network timeout. Fail open at once; the next start retries.
            _dp._state(root, "settled").unlink(missing_ok=True)
            _note_failure(root, _dp._BACKOFF)
            if pending:
                say("  ⚠ Default plugins not installed: the plugin catalog is unreachable (offline?). "
                    "Hermes retries at a later start.")
            outcome = {"installed": [], "owed": ["live catalog", *unreadable_homes, *pending]}
        elif home is None:
            outcome = _dp.install_and_record(say=say, wait=_CHILD_LOCK_WAIT)
        else:
            outcome = _install_subset_and_record([home], entries, root, say=say, wait=_CHILD_LOCK_WAIT)
    except Exception as exc:  # health: allow BLE001 -- reported to the waiting ensure as owed, and backed off
        logger.exception("default plugin install failed")
        _note_failure(root, _dp._BACKOFF)
        outcome = {"installed": [], "owed": [f"error: {exc}"]}
    _dp._write_child_result(root, run_id, outcome)  # not on a BaseException: no result reads as owed
    return 1 if outcome["owed"] else 0


def _run_child(*, run_id: str = "", home: Optional[Path] = None) -> int:
    """The detached child; one that finds another installer running leaves its request to that one. Started
    by an ``ensure`` (with *run_id*), it waits for the lock and reports to that ensure."""
    from hermes_cli import default_plugins as _dp
    from pm.environments import dependency_home_root

    root = Path(dependency_home_root())
    (root / "logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s default-plugins %(message)s",
                        filename=str(root / "logs" / "default-plugins.log"))
    if run_id:
        return _dp._ensure_child_work(run_id, home, say=logger.info)
    try:
        return 1 if _dp.install_and_record(say=logger.info, wait=1.0)["owed"] else 0
    except TimeoutError:
        return 0
    except Exception:  # health: allow BLE001 -- a detached child has no stderr: log the cause and back off
        from hermes_cli.memory_provider_migration import _note_failure
        logger.exception("default plugin install failed")
        _note_failure(root, _dp._BACKOFF)
        return 1


def _ensure_payload(
    state: str,
    pending_before: list[str],
    installed: list[str],
    owed: list[str],
    started_monotonic: float,
) -> dict:
    return {
        "state": state,
        "pending_before": list(pending_before),
        "installed": list(installed),
        "owed": list(owed),
        "elapsed_ms": max(0, round((time.monotonic() - started_monotonic) * 1000)),
    }


def _outcome_exit(pending_before: list[str], outcome: dict, t0: float) -> tuple[int, dict]:
    installed, owed = list(outcome.get("installed") or []), list(outcome.get("owed") or [])
    if owed:
        return ENSURE_EXIT_OWED, _ensure_payload("owed", pending_before, installed, owed, t0)
    if installed:
        return ENSURE_EXIT_OK, _ensure_payload("installed", pending_before, installed, [], t0)
    return ENSURE_EXIT_OK, _ensure_payload("nothing-pending", pending_before, [], [], t0)


def _check_ensure_fast_path(
    homes: Optional[list[Path]],
    allowed_homes: list[Path],
    unreadable_homes: list[Path],
    root: Path,
    entries: list,
    pending_before: list[str],
    t0: float,
) -> Optional[tuple[int, dict]]:
    """Return a settled/owed response immediately when no install or record write is needed."""
    from hermes_cli import default_plugins as _dp

    if pending_before or unreadable_homes:
        return None
    if homes is None:
        if _dp._is_settled_for(root, entries) and _dp._all_recorded(allowed_homes, entries):
            return ENSURE_EXIT_OK, _ensure_payload("nothing-pending", [], [], [], t0)
        return None
    if allowed_homes and _dp._live_catalog_unreachable():
        _dp._state(root, "settled").unlink(missing_ok=True)
        return ENSURE_EXIT_OWED, _ensure_payload("owed", [], [], ["live catalog"], t0)
    if _dp._all_recorded(allowed_homes, entries):
        return ENSURE_EXIT_OK, _ensure_payload("nothing-pending", [], [], [], t0)
    return None


def _pending_names(homes: list[Path], entries: list) -> list[str]:
    from hermes_cli import default_plugins as _dp

    pending: list[str] = []
    seen: set[str] = set()
    for home in homes:
        done = _dp._recorded(home)
        for entry in entries:
            if entry.name not in done and not _dp._settled(home, entry) and entry.name not in seen:
                seen.add(entry.name)
                pending.append(entry.name)
    return pending


def _record_settled_home(home: Path, entries: list) -> Optional[str]:
    """Record already-settled defaults for one *home* (never installs); returns ``str(home)`` on error."""
    from hermes_cli import default_plugins as _dp

    try:
        done = _dp._recorded(home)
        to_record = {entry.name for entry in entries if entry.name not in done and _dp._settled(home, entry)}
        if not to_record:
            return None
        for entry in entries:
            if entry.name in to_record:
                _dp._installing(home, entry).unlink(missing_ok=True)
        _dp._record(home, to_record)
        return None
    except Exception:  # health: allow BLE001 -- one unwritable home stays owed; never blocks others
        logger.warning("default plugins record skipped for %s", home, exc_info=True)
        return str(home)


def _record_settled_only(
    subset: Optional[list[Path]],
    entries: list,
    root: Path,
    emit: Callable[[str], None],
    deadline: float,
    t0: float,
) -> tuple[str, Any]:
    """Strictly record-only in-process pass: writes per-home records for already-settled defaults
    and updates the settled marker under lock, NEVER invoking ``_install_into`` or starting installs.

    If a newly discovered or changed home has pending installs once the lock is held, releases the
    lock and returns ``("needs_install", pending)`` so the caller delegates to the detached child.
    """
    from hermes_cli import default_plugins as _dp
    from hermes_cli.memory_provider_migration import _note_failure, _note_success
    from pm.plugins_state import dependency_homes

    wait = max(0.0, deadline - time.monotonic())
    try:
        with _dp.installer_lock(wait):
            homes = list(dependency_homes()) if subset is None else list(subset)
            policies = {home: _dp.fetch_policy(home) for home in homes}
            owed = [str(home) for home, policy in policies.items() if policy == "unreadable"]
            allowed = [home for home, policy in policies.items() if policy == "allowed"]
            still_pending = _dp._pending_names(allowed, entries)
            if still_pending:
                return "needs_install", still_pending
            if allowed and _dp._live_catalog_unreachable():
                owed.append("live catalog")
            for home in allowed:
                failed_home = _record_settled_home(home, entries)
                if failed_home is not None:
                    owed.append(failed_home)
            if owed:
                _note_failure(root, _dp._BACKOFF)
                _dp._state(root, "settled").unlink(missing_ok=True)
                return "done", (ENSURE_EXIT_OWED, _ensure_payload("owed", [], [], owed, t0))
            if subset is None:
                _note_success(root, _dp._BACKOFF)
                if allowed:
                    _dp._write_settled(
                        _dp._state(root, "settled"),
                        _dp._identity(),
                        _dp._defaults_hash(entries),
                        default_names=[e.name for e in entries],
                        live_cache_sig=_dp._live_cache_signature(),
                    )
            return "done", (ENSURE_EXIT_OK, _ensure_payload("nothing-pending", [], [], [], t0))
    except TimeoutError:  # another installer holds the lock; its pass covers this one
        return "done", (ENSURE_EXIT_TIMEOUT, _ensure_payload("timeout", [], [], [], t0))
    except OSError as exc:
        emit(f"default plugins ensure failed: {exc}")
        return "done", (ENSURE_EXIT_OWED, _ensure_payload("owed", [], [], [str(exc)], t0))


def ensure_defaults(
    *,
    timeout: float = DEFAULT_ENSURE_TIMEOUT,
    homes: Optional[list[Path]] = None,
    say: Optional[Callable[[str], None]] = None,
) -> tuple[int, dict]:
    """Ensure every default catalog plugin is installed before a service starts (bounded foreground pass).

    Checks local state and fresh on-disk cache first (zero network). Whenever the catalog cache is
    missing/expired or defaults are pending, launches one detached child (:func:`_launch_ensure_child`)
    so ``timeout`` bounds the entire foreground wait (catalog HTTP fetch + installs) while the child
    survives a deadline and finishes cleanly in the background.

    Returns ``(exit_code, payload)`` where ``exit_code`` is:
    - ``0`` (``nothing-pending`` or ``installed``)
    - ``2`` (``owed`` — offline / failed / unreadable home; retried next start)
    - ``3`` (``timeout`` — hard deadline hit; the detached installer child keeps going and finishes)
    - ``4`` (``skipped`` — fetch policy denied for all target homes)
    """
    from hermes_cli import default_plugins as _dp
    from pm.environments import dependency_home_root
    from pm.plugins_state import dependency_homes

    t0 = time.monotonic()
    budget = max(0.0, float(timeout))
    deadline = t0 + budget
    emit: Callable[[str], None] = say if say is not None else (lambda msg: print(msg, file=sys.stderr))

    target_homes = list(dependency_homes()) if homes is None else [Path(h) for h in homes]
    policies = {home: _dp.fetch_policy(home) for home in target_homes}
    if target_homes and all(policy == "denied" for policy in policies.values()):
        return ENSURE_EXIT_SKIPPED, _ensure_payload("skipped", [], [], [], t0)

    allowed_homes = [home for home, policy in policies.items() if policy == "allowed"]
    unreadable_homes = [home for home, policy in policies.items() if policy == "unreadable"]
    root = Path(dependency_home_root())

    # Zero-parse fast path: install identity, fresh live catalog cache, and recorded defaults unchanged.
    if homes is None and not unreadable_homes and _dp._fast_settled_hit(root, allowed_homes):
        return ENSURE_EXIT_OK, _ensure_payload("nothing-pending", [], [], [], t0)

    # Local-only cache check (ZERO network): missing/expired cache delegates discovery to the child
    # so slow catalog HTTP fetches are strictly bounded by --timeout.
    cached_entries = _dp._cached_default_entries() if allowed_homes else []
    pending_before: Optional[list[str]] = None
    if cached_entries is not None:
        pending_before = _dp._pending_names(allowed_homes, cached_entries)
        fast = _check_ensure_fast_path(homes, allowed_homes, unreadable_homes, root, cached_entries, pending_before, t0)
        if fast is not None:
            return fast
        if not pending_before:
            # Nothing to fetch: strictly record-only in-process pass (never installs; falls through
            # to the detached child if a newly discovered home has pending installs under lock).
            status, detail = _record_settled_only(
                None if homes is None else target_homes, cached_entries, root, emit, deadline, t0,
            )
            if status == "done":
                return detail
            pending_before = detail

    if time.monotonic() >= deadline:
        owed = list(pending_before) if pending_before else ["live catalog"]
        return ENSURE_EXIT_TIMEOUT, _ensure_payload("timeout", list(pending_before or ()), [], owed, t0)

    run_id = uuid.uuid4().hex
    log_tail = _LogTail(root / "logs" / "default-plugins.log")
    subset_home = target_homes[0] if homes is not None and len(target_homes) == 1 else None
    try:
        child = _dp._launch_ensure_child(run_id, subset_home)
    except OSError as exc:
        emit(f"default plugins ensure failed: could not start the installer: {exc}")
        owed = list(pending_before) if pending_before else [str(exc)]
        return ENSURE_EXIT_OWED, _ensure_payload("owed", list(pending_before or ()), [], owed, t0)

    result = _wait_for_child(child, root, run_id, deadline, log_tail, emit)
    if result == "timeout":
        emit(f"default plugins: still installing after {budget:.0f}s; continuing in the background "
             f"(pid {child.pid}, log {root / 'logs' / 'default-plugins.log'})")
        owed = list(pending_before) if pending_before else ["live catalog"]
        return ENSURE_EXIT_TIMEOUT, _ensure_payload("timeout", list(pending_before or ()), [], owed, t0)
    if result is None:
        emit(f"default plugins ensure failed: the installer exited ({child.returncode}) without a result")
        owed = list(pending_before) if pending_before else [f"installer exit {child.returncode}"]
        return ENSURE_EXIT_OWED, _ensure_payload("owed", list(pending_before or ()), [], owed, t0)
    effective_pending = (
        list(pending_before)
        if pending_before is not None
        else sorted(set(result["installed"]) | {x for x in result["owed"] if x != "live catalog"})
    )
    return _outcome_exit(effective_pending, result, t0)


def _wait_for_child(child, root: Path, run_id: str, deadline: float, log_tail: _LogTail,
                    emit: Callable[[str], None]):
    """*run_id*'s result once the child wrote it, None when it exited without one, or ``"timeout"`` at
    the deadline (the child is left running). Relays the child's log lines meanwhile."""
    from hermes_cli import default_plugins as _dp

    while True:
        log_tail.pump(emit)
        result = _dp._read_child_result(root, run_id)
        if result is not None:
            break
        if child.poll() is not None:
            result = _dp._read_child_result(root, run_id)  # written just before it exited
            break
        if time.monotonic() >= deadline:
            return "timeout"
        time.sleep(0.05)
    log_tail.pump(emit)
    return result


class _LogTail:
    """Relays the lines the installer child appends to its log (per-plugin timing included) to the
    ensure's own output while it waits. Starts at the log's current end."""

    def __init__(self, path: Path) -> None:
        self.path = path
        try:
            self.offset = path.stat().st_size
        except OSError:
            self.offset = 0
        self.partial = b""

    def pump(self, emit: Callable[[str], None]) -> None:
        try:
            with self.path.open("rb") as fh:
                fh.seek(self.offset)
                chunk = fh.read()
        except OSError:
            return
        if not chunk:
            return
        self.offset += len(chunk)
        *lines, self.partial = (self.partial + chunk).split(b"\n")
        for line in lines:
            text = line.decode("utf-8", "replace").rstrip("\r")
            if text.strip():
                emit(text)


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point for ``python -m hermes_cli.default_plugins``.

    Bare invocation (no args) preserves the detached-child behavior (:func:`_run_child`).
    ``ensure [--timeout SECONDS] [--json]`` runs the bounded foreground preflight.
    ``_child --run-id ID [--home PATH]`` is the internal detached installer spawned by ``ensure``.
    """
    from hermes_cli import default_plugins as _dp

    args_list = list(sys.argv[1:] if argv is None else argv)
    if not args_list:
        return _dp._run_child()

    parser = argparse.ArgumentParser(prog="python -m hermes_cli.default_plugins")
    subparsers = parser.add_subparsers(dest="command", required=True)
    ensure_p = subparsers.add_parser("ensure", help="Install pending default catalog plugins in the foreground")
    ensure_p.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_ENSURE_TIMEOUT,
        help="Seconds to wait for the detached installer child before returning 3 (default: 300)",
    )
    ensure_p.add_argument(
        "--json",
        action="store_true",
        help="Emit a single JSON status line on stdout (human/log output goes to stderr)",
    )
    child_p = subparsers.add_parser("_child", help=argparse.SUPPRESS)
    child_p.add_argument("--run-id", required=True)
    child_p.add_argument("--home", default="")
    parsed = parser.parse_args(args_list)
    if parsed.command == "_child":
        target = parsed.home.strip()
        return _dp._run_child(run_id=parsed.run_id.strip(), home=Path(target) if target else None)
    if parsed.command == "ensure":
        real_stdout = sys.stdout
        if parsed.json:
            sys.stdout = sys.stderr
        try:
            code, payload = _dp.ensure_defaults(
                timeout=parsed.timeout,
                say=lambda msg: print(msg, file=sys.stderr),
            )
            if parsed.json:
                real_stdout.write(json.dumps(payload) + "\n")
                real_stdout.flush()
            else:
                print(
                    f"default-plugins ensure: state={payload['state']} "
                    f"installed={len(payload['installed'])} owed={len(payload['owed'])} "
                    f"elapsed_ms={payload['elapsed_ms']}",
                    file=sys.stderr,
                )
            return code
        finally:
            sys.stdout = real_stdout
    return 0
