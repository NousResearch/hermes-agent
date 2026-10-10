"""Detect a stale kanban dispatch plane: dispatch code on disk the serving gateway never loaded.

Dispatch-plane modules are imported *inside* the gateway process, so ``sys.modules`` freezes them at
boot. Editing one does not reload it — a change can be shipped, reviewed and verified while the live
board still runs the old code. ``gateway/code_skew.py`` answers a neighbouring but different question
(*has the git checkout moved since boot*); dispatch work lands as uncommitted working-tree edits, so
the revision never changes and that guard stays silent through exactly this failure.

Observed twice in one night (2026-09-24/25) — edits that were shipped, reviewed and verified while the
live board kept running the modules imported at boot:

| change | file mtimes | effect |
|---|---|---|
| Jev per-card model router hook (`scripts/jev_router.py`, the path `kanban_db_dispatch.py` shells out to) | 23:17:27 | inert until the 00:25:13 restart |
| goal-mode default policy (`kanban_goal_policy.py` 23:53:13, `kanban_db.py` 23:55:13, `kanban.py` 00:15:12, `kanban_db_dispatch.py` 00:15:51) | 00:15:51 | the policy the review approved was not the policy the board ran until the 00:25:13 restart |
| card `t_50ee8a30` | claimed 00:24:10 — 63 s before the new generation booted | ran on the old dispatcher, so it went out with `goal_mode=0` |

The check snapshots the watched modules' mtimes when a serving process starts and compares them with
the same files on disk afterwards (``st_mtime_ns``). Two surfaces consume it:

* ``hermes doctor`` judges a **live gateway's persisted snapshot** (``gateway_state.json``) — it runs
  in its own process and cannot see the gateway's memory.
* the embedded dispatcher calls :func:`probe_and_act` once per tick: one warning per process, then
  (unless ``kanban.dispatch_auto_reload`` is false) it asks for its own **deferred** restart through
  the existing drain-first ``request_restart`` path, and only once no kanban worker is running.

Never a false positive by construction: only ``mtime > boot snapshot`` counts, so a gateway started
after the last edit is clean. Every failure of this module (unreadable files, a missing stamp,
an unreadable process start time) degrades to "unknown", which warns about nothing.
"""

from __future__ import annotations

import importlib
import os
import time
from pathlib import Path
from typing import Any, Callable, NamedTuple, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Every module the gateway imports to claim, route, spawn and reap kanban workers. The check is only
# as good as this list: extend it when a new dispatch-plane module appears.
WATCHED_SOURCES: tuple[str, ...] = (
    "hermes_cli/kanban_db_dispatch.py",
    "hermes_cli/kanban_db.py",
    "hermes_cli/kanban_db_connect.py",
    "hermes_cli/kanban.py",
    "hermes_cli/kanban_goal_policy.py",
    "gateway/kanban_watchers.py",
    "gateway/kanban_watchers_common.py",
    "gateway/kanban_watchers_dispatcher.py",
)

# The dispatch-time model router is a workspace script outside the package; the dispatcher shells out
# to the path it holds in ``kanban_db_dispatch._JEV_ROUTER_SCRIPT``. Resolved lazily from that
# attribute (never duplicated here) so the check follows the dispatcher and stays import-light.
_ROUTER_SCRIPT_ATTR = ("hermes_cli.kanban_db_dispatch", "_JEV_ROUTER_SCRIPT")

_RESTART_COMMAND = "hermes gateway restart"
_AUTO_RELOAD_CONFIG_KEY = "dispatch_auto_reload"
_AUTO_RELOAD_ENV = "HERMES_KANBAN_DISPATCH_AUTO_RELOAD"

# ``gateway.status._get_process_start_time`` yields epoch *centiseconds* where psutil answers (macOS,
# Windows) and a /proc clock-tick counter on Linux. Only the former is an epoch, so the derived
# fallback is offered only when the value is plausibly a post-2001 epoch stamp — on Linux the value is
# far below this floor and the record simply stays "unknown" rather than misjudging.
_MIN_EPOCH_CENTISECONDS = 100_000_000_000  # 2001-09-09T01:46:40Z
_MAX_START_SKEW_SECONDS = 86_400  # a start time further in the future than this is junk


def watched_paths() -> list[tuple[str, Path]]:
    """``(label, absolute path)`` for every watched source present on disk, in a stable order.

    The label (repo-relative path, or the script's basename) is what the operator-facing messages
    name, so it must stay readable without a second lookup.
    """
    found: list[tuple[str, Path]] = []
    for relative in WATCHED_SOURCES:
        path = _PROJECT_ROOT / relative
        if path.is_file():
            found.append((relative, path))
    script = _router_script_path()
    if script is not None:
        found.append((script.name, script))
    return found


def _router_script_path() -> Optional[Path]:
    """The dispatch-time router script the dispatcher actually shells out to, when it exists."""
    module_name, attribute = _ROUTER_SCRIPT_ATTR
    try:
        raw = getattr(importlib.import_module(module_name), attribute, None)
    except Exception:
        return None
    if not isinstance(raw, str) or not raw.strip():
        return None
    path = Path(raw).expanduser()
    return path if path.is_file() else None


def snapshot() -> Optional[dict[str, Any]]:
    """``{max_mtime_ns, newest, files, sources}`` over the watched files, or None when none exist.

    ``sources`` maps each label to its ``st_mtime_ns`` so a later comparison can name exactly the
    files that changed; ``max_mtime_ns`` is the only field a persisted snapshot needs.
    """
    sources: dict[str, int] = {}
    for label, path in watched_paths():
        try:
            sources[label] = path.stat().st_mtime_ns
        except OSError:
            continue
    if not sources:
        return None
    newest = max(sources, key=lambda label: sources[label])
    return {"max_mtime_ns": sources[newest], "newest": newest, "files": len(sources), "sources": sources}


def newer_than(current: Optional[dict[str, Any]], boot_mtime_ns: int) -> tuple[str, ...]:
    """Labels in *current* whose mtime is newer than *boot_mtime_ns*, sorted.

    Exact rather than approximate: at snapshot time every file's mtime is at or below the snapshot's
    max, so a file edited afterwards always lands above it, and a file that was already loaded never
    does.
    """
    sources = (current or {}).get("sources") or {}
    return tuple(sorted(label for label, mtime in sources.items()
                        if isinstance(mtime, int) and mtime > boot_mtime_ns))


def stale_against(boot_mtime_ns: Optional[int]) -> Optional[dict[str, Any]]:
    """``{current, newer}`` when the watched code is newer than *boot_mtime_ns*, else None."""
    if boot_mtime_ns is None:
        return None
    current = snapshot()
    if current is None:
        return None
    newer = newer_than(current, boot_mtime_ns)
    return {"current": current, "newer": newer} if newer else None


# ---- This process's own boot snapshot ------------------------------------------------------------

_boot: Optional[dict[str, Any]] = None


def record_boot(force: bool = False) -> Optional[dict[str, Any]]:
    """Snapshot the watched mtimes for THIS process; idempotent (the first call wins).

    Called at gateway startup (``gateway.run.start_gateway``) so the snapshot exists before the first
    ``gateway_state.json`` write, and again at the top of the dispatcher loop so an embedder that
    never goes through ``start_gateway`` is still covered.
    """
    global _boot
    if _boot is None or force:
        _boot = snapshot()
    return _boot


def boot_snapshot() -> Optional[dict[str, Any]]:
    """This process's boot snapshot, or None when it was never taken."""
    return _boot


def boot_stamp_fields() -> dict[str, Any]:
    """Compact boot-snapshot fields for ``gateway_state.json`` (the read side is :func:`judge_record`).

    Empty until a snapshot exists: a status write from a process that never recorded one must not
    invent a snapshot taken *now*, which would read as "fresh" forever.
    """
    boot = _boot
    if not boot:
        return {}
    return {
        "dispatch_code_max_mtime_ns": boot["max_mtime_ns"],
        "dispatch_code_files": boot["files"],
        "dispatch_code_newest": boot["newest"],
    }


def detect_stale() -> Optional[tuple[int, dict[str, Any], tuple[str, ...]]]:
    """``(boot_mtime_ns, current_snapshot, newer_labels)`` when this process is serving stale code."""
    boot = _boot
    if boot is None:
        return None
    found = stale_against(boot["max_mtime_ns"])
    if found is None:
        return None
    return boot["max_mtime_ns"], found["current"], found["newer"]


# ---- Judging a persisted record (the ``hermes doctor`` side) --------------------------------------

class Verdict(NamedTuple):
    """Freshness verdict for one ``gateway_state.json`` record."""

    status: str                      # "stale" | "fresh" | "unknown"
    source: str                      # "stamp" (boot snapshot) | "start_time" | ""
    boot_mtime_ns: Optional[int]
    current_mtime_ns: Optional[int]
    newest: str
    files: int
    stale_files: tuple[str, ...]
    reason: str = ""


def _derived_boot_ns(record: dict[str, Any]) -> Optional[int]:
    """Boot instant in ns derived from a persisted process ``start_time``, or None when unusable."""
    start = record.get("start_time")
    if not isinstance(start, int) or isinstance(start, bool):
        return None
    now_cs = int(time.time() * 100)
    if not _MIN_EPOCH_CENTISECONDS < start <= now_cs + _MAX_START_SKEW_SECONDS * 100:
        return None
    return start * 10_000_000


def judge_record(record: Optional[dict[str, Any]]) -> Verdict:
    """Judge a gateway runtime record against the dispatch code on disk *now*.

    Prefers the record's own boot snapshot (``dispatch_code_max_mtime_ns``). A record written before
    this check existed carries none; for those the process ``start_time`` is used when it is
    readable as an epoch, which is the same comparison one step coarser (code edited during startup
    reads as stale). Anything else is "unknown" and warns about nothing.
    """
    current = snapshot()
    if current is None:
        return Verdict("unknown", "", None, None, "", 0, (), "no watchable dispatch module found on disk")
    record = record if isinstance(record, dict) else {}
    stamped = record.get("dispatch_code_max_mtime_ns")
    if isinstance(stamped, int) and not isinstance(stamped, bool):
        source, boot_ns = "stamp", stamped
    else:
        derived = _derived_boot_ns(record)
        if derived is None:
            return Verdict(
                "unknown", "", None, current["max_mtime_ns"], current["newest"], current["files"], (),
                "the runtime record carries no dispatch-code snapshot and no readable process start "
                "time (Linux keeps a /proc tick counter there, not an epoch); restart the gateway to "
                "stamp one")
        source, boot_ns = "start_time", derived
    newer = newer_than(current, boot_ns)
    status = "stale" if newer else "fresh"
    return Verdict(status, source, boot_ns, current["max_mtime_ns"], current["newest"], current["files"], newer)


def restart_command() -> str:
    """The command that loads dispatch code edited after the gateway booted."""
    return _RESTART_COMMAND


def restart_hint() -> str:
    """Operator-facing restart advice; the host gateway is the ONE process that serves every profile."""
    return (f"restart the gateway to load it: `{_RESTART_COMMAND}` "
            "(the host gateway serving every profile: `hermes --profile default gateway restart`)")


def auto_reload_enabled(kanban_cfg: Optional[dict[str, Any]] = None) -> bool:
    """``kanban.dispatch_auto_reload`` (default True); ``HERMES_KANBAN_DISPATCH_AUTO_RELOAD`` overrides.

    The env var is the escape hatch for an operator who wants the warning without the bounce.
    """
    raw_env = os.environ.get(_AUTO_RELOAD_ENV, "").strip().lower()
    if raw_env in {"0", "false", "no", "off"}:
        return False
    if raw_env in {"1", "true", "yes", "on"}:
        return True
    raw = (kanban_cfg or {}).get(_AUTO_RELOAD_CONFIG_KEY)
    if isinstance(raw, bool):
        return raw
    if raw is None:
        return True
    return str(raw).strip().lower() not in {"0", "false", "no", "off", ""}


def reload_opt_out_hint() -> str:
    """How to keep the warning but stop the self-reload (named in the boot log line and doctor)."""
    return f"kanban.{_AUTO_RELOAD_CONFIG_KEY}: false (or {_AUTO_RELOAD_ENV}=0)"


def sources_compile(labels: tuple[str, ...]) -> Optional[str]:
    """None when every named source compiles, else ``"<label>: <error>"`` for the first one that does not.

    A half-written module sits on disk with a *newer* mtime than anything loaded, so acting on that
    snapshot would restart the gateway onto unimportable code. Read once, at decision time only.
    """
    wanted = set(labels)
    for label, path in watched_paths():
        if label not in wanted:
            continue
        try:
            compile(path.read_text(encoding="utf-8-sig"), str(path), "exec")
        except (OSError, SyntaxError, ValueError) as exc:
            return f"{label}: {type(exc).__name__}: {exc}"
    return None


# ---- The serving process's per-tick probe ---------------------------------------------------------

# Process-local probe state; the embedded dispatcher is the only caller (one loop per process).
_seen_mtime_ns: Optional[int] = None
_settled = False
_warned = False
_broken_mtime_ns: Optional[int] = None
_defer_logged: Any = None


def settled() -> bool:
    """True once this process has nothing left to do (warned without auto-reload, or reload asked)."""
    return _settled


def reset_state() -> None:
    """Forget the process-local probe state (test seam)."""
    global _seen_mtime_ns, _settled, _warned, _broken_mtime_ns, _defer_logged
    _seen_mtime_ns = _broken_mtime_ns = _defer_logged = None
    _settled = _warned = False


def _worker_count(running_workers: Optional[Callable[[], Any]]) -> Optional[int]:
    """Live kanban worker count, or None when it cannot be established (never a reason to act)."""
    if running_workers is None:
        return None
    try:
        value = running_workers()
    except Exception:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def probe_and_act(
    log, *, running_workers: Optional[Callable[[], Any]] = None,
    auto_reload: bool = True, request_restart: Optional[Callable[[], Any]] = None,
) -> str:
    """One staleness probe for a serving process; returns the action taken.

    ``clean``     nothing newer on disk;
    ``unstable``  newer code, but not on two consecutive probes (a save may still be in flight);
    ``broken``    the newer file does not compile — never a reload target;
    ``warned``    the one-per-process warning was logged (auto-reload off or unavailable);
    ``deferred``  stale and auto-reload is on, but kanban worker(s) are still running;
    ``requested`` the deferred restart was requested through the gateway's drain-first path.
    """
    global _seen_mtime_ns, _settled, _warned, _broken_mtime_ns, _defer_logged
    if _settled:
        return "settled"
    detected = detect_stale()
    if detected is None:
        _seen_mtime_ns = None
        return "clean"
    _boot_mtime_ns, current, newer = detected
    if current["max_mtime_ns"] != _seen_mtime_ns:
        # First sighting of this snapshot: a writer may still be mid-save, and a restart onto a
        # truncated module would take the gateway down. Act only once two probes agree.
        _seen_mtime_ns = current["max_mtime_ns"]
        return "unstable"
    if not _warned:
        if _broken_mtime_ns == current["max_mtime_ns"]:
            return "broken"  # this unimportable snapshot was already reported; nothing new to say
        failure = sources_compile(newer)
        if failure is not None:
            _broken_mtime_ns = current["max_mtime_ns"]
            log.warning(
                "kanban dispatcher: dispatch-plane code newer on disk does not compile, so this "
                "process cannot reload onto it (%s). Fix the file and the check re-evaluates.", failure)
            return "broken"
        log.warning(
            "kanban dispatcher: STALE DISPATCH PLANE — %d dispatch module(s) changed on disk after this "
            "process started (%s), so the live board is still running the code loaded at startup. %s",
            len(newer), ", ".join(newer), restart_hint())
        _warned = True
        if not auto_reload or request_restart is None:
            _settled = True
            return "warned"
    if request_restart is None:
        _settled = True
        return "warned"
    active = _worker_count(running_workers)
    if active != 0:
        # Unknown (None) is deferred too: never bounce the gateway on a worker count that could not be
        # taken. The 2026-09-25 incident ran a card with the wrong policy — an unattended restart
        # under a live worker trades that for a lost worker.
        shown = active if isinstance(active, int) and active > 0 else "unknown"
        if _defer_logged != shown:
            _defer_logged = shown
            log.info(
                "kanban dispatcher: restart deferred: waiting on %s active kanban worker(s) before "
                "reloading the dispatch plane (%s)", shown, ", ".join(newer))
        return "deferred"
    if not request_restart():
        log.warning("kanban dispatcher: could not request the dispatch-plane restart; restart manually: %s",
                    restart_hint())
        _settled = True
        return "warned"
    log.info("kanban dispatcher: dispatch-plane reload requested (%s); the gateway drains in-flight work "
             "and exits for its supervisor to relaunch it.", ", ".join(newer))
    _settled = True
    return "requested"
