"""Process identity: spawn tags, the machine-wide spawn ledger, the Windows job-object self-attach,
and the POSIX owner-death supervisor.

Four layers make every long-lived Hermes process positively identifiable, so reapers (``hermes
update``, Desktop startup sweeps) never guess lineage from PPID archaeology or cmdline matching,
and make every long-lived CHILD die with the Hermes process that owns it:
1. spawn tags (``HERMES_SPAWN`` env stamped by the spawner); 2. a ``(pid, create_time)`` ledger;
3. Windows job-object self-attach with ``KILL_ON_JOB_CLOSE`` so the whole child tree dies with the
root — no launcher→worker chains left holding ``.pyd`` locks; 4. on POSIX, one supervisor per
Hermes process that reaps the owned children's process groups the moment Hermes dies.
"""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import os
import platform
import signal
import subprocess
import sys
import threading
import time
from contextlib import ExitStack, contextmanager, suppress
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Optional, Sequence

from utils import atomic_json_write

logger = logging.getLogger(__name__)

SPAWN_ENV_VAR = "HERMES_SPAWN"
_TAG_VERSION = "v1"
LEDGER_FILENAME = "spawn-ledger.json"

#: Purposes a reaper may treat as "safe to kill when the owner is gone".
#: Interactive processes (chat, REPLs) are deliberately NOT in this set.
REAPABLE_PURPOSES = frozenset({"serve", "dashboard", "gateway", "mcp-helper"})

#: Host-role owners a fresh backend may reap on a HELD_BY_OTHER conflict (#121964): the
#: serve/dashboard/gateway subset. The mcp-helper rung stays owned by reap_orphaned_mcp_helpers.
_BACKEND_OWNER_PURPOSES = REAPABLE_PURPOSES - {"mcp-helper"}

_IS_WINDOWS = platform.system() == "Windows"

# Module-global job handle: must live exactly as long as this process so the
# kernel closes it (and kills the job) when we die. Never close it manually.
_JOB_HANDLE = None
_LEDGER_LOCK = threading.Lock()


def install_id(project_root: Optional[Path] = None) -> str:
    """Stable 12-hex identifier for THIS install (derived from its path)."""
    if project_root is None:
        try:
            from hermes_constants import PROJECT_ROOT as _root

            project_root = Path(_root)
        except Exception:
            project_root = Path(__file__).resolve().parent.parent
    try:
        canonical = str(Path(project_root).resolve()).lower()
    except OSError:
        canonical = str(project_root).lower()
    return hashlib.sha256(canonical.encode("utf-8", "replace")).hexdigest()[:12]


def process_birth(proc) -> float:
    """A managed worker's identity birth for the ``psutil.Process`` *proc*, the same in every process.

    On macOS psutil shifts ``create_time()`` by ``abs(INIT_BOOT_TIME - boot_time())``
    (``psutil._psosx.adjust_proc_create_time``), with a boot-time reference frozen when psutil
    was imported in the CALLING process. After a clock step (NTP, sleep/wake) a long-lived
    gateway owner and a freshly spawned worker read different births for the same process, and
    the exact identity match refused every managed worker (#140580). The kernel's own start
    (``monotonic=True``: the unadjusted ``kinfo_proc`` epoch start) is what every process sees.
    Other platforms keep ``create_time()``: Linux and Windows apply no per-process reference.
    Raises ``psutil.Error`` like ``create_time()``.
    """
    import psutil
    if getattr(psutil, "MACOS", False):
        platform_proc = getattr(proc, "_proc", None)
        if platform_proc is not None:
            try:
                return float(platform_proc.create_time(monotonic=True))
            except TypeError:  # a psutil without the monotonic read: fall back to the adjusted one
                pass
    return float(proc.create_time())


def worker_birth(proc) -> float:
    """The birth a managed worker's identity fences compare: its hello, its scope and the Kanban
    owner receipt. :func:`process_birth`, except on Linux, where psutil's ``create_time()`` adds the
    ``/proc/stat`` btime it re-reads on every call. WSL2 re-anchors btime after host sleep, so the
    owner read a live worker's birth shifted from the one it reported and refused it (KB-12). The
    kernel's start ticks since boot (``monotonic=True``) never move. The spawn ledger keeps
    :func:`process_birth`: its persisted epoch births must keep matching across an update.
    Raises ``psutil.Error`` like ``create_time()``.
    """
    import psutil
    if getattr(psutil, "LINUX", False):
        return float(proc._proc.create_time(monotonic=True))
    return process_birth(proc)


def boot_start(proc) -> Optional[float]:
    """Seconds from boot to the start of the ``psutil.Process`` *proc* on Linux; ``None`` elsewhere.

    The ledger's epoch ``create_time`` is :func:`process_birth`, which on Linux is the kernel's
    start ticks plus the ``/proc/stat`` btime psutil re-reads on every call, so an NTP or VM-resume
    clock step moves it by whole seconds and a live process stops matching its entry. Ledger
    entries carry this alongside it (``boot_start`` / ``spawner_boot``) and compare it exactly.
    Raises ``psutil.Error`` like ``create_time()``.
    """
    import psutil
    if getattr(psutil, "LINUX", False):
        return float(proc._proc.create_time(monotonic=True))
    return None


def _process_create_time(pid: Optional[int] = None) -> Optional[float]:
    """:func:`process_birth` of ``pid`` (default: this process); ``None`` when psutil can't say."""
    try:
        import psutil

        return process_birth(psutil.Process(os.getpid() if pid is None else pid))
    except Exception:
        return None


@functools.lru_cache(maxsize=1)
def _live_boot_id() -> Optional[str]:
    """The kernel's id for the running boot (Linux); ``None`` elsewhere or when unreadable."""
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


# No live process has a negative boot_start, so an entry from an earlier boot never matches.
_EARLIER_BOOT = -1.0


def _entry_boot(entry: dict, key: str) -> Optional[float]:
    """The entry's boot-relative birth ``key`` (``boot_start``/``spawner_boot``) as exact proof.

    Boot ticks restart at every boot while the ledger survives reboots (and early-boot services
    get the same pid at the same tick), so they identify a process only within the boot that
    recorded them: an entry from another boot is :data:`_EARLIER_BOOT`, matching nothing alive.
    ``None`` (epoch match) when either boot id is unknown.
    """
    recorded, live = entry.get("boot_id"), _live_boot_id()
    if not isinstance(recorded, str) or live is None:
        return None
    return entry.get(key) if recorded == live else _EARLIER_BOOT


def _verified_boot_start(pid: Optional[int], create_time: Optional[float]) -> Optional[float]:
    """:func:`boot_start` of ``pid`` while it still matches ``create_time``; ``None`` when unproven
    or when the boot is unidentifiable (ticks alone repeat across reboots)."""
    if not isinstance(pid, int) or pid <= 0 or create_time is None or _live_boot_id() is None:
        return None
    try:
        import psutil

        proc = psutil.Process(pid)
        return boot_start(proc) if _same_incarnation(proc, create_time) else None
    except Exception:  # health: allow BLE001 -- best-effort probe like _process_create_time: any failure is "unproven"
        return None


# Layer 1 — spawn tags


@dataclass(frozen=True)
class SpawnTag:
    install: str
    purpose: str
    spawner_pid: int
    spawner_create: Optional[float]


def build_spawn_tag(purpose: str, *, project_root: Optional[Path] = None) -> str:
    """Value for the child's ``HERMES_SPAWN`` env var, stamped by the spawner."""
    create = _process_create_time()
    create_part = f"{create:.3f}" if create is not None else "-"
    return ":".join((_TAG_VERSION, install_id(project_root), purpose, str(os.getpid()), create_part))


def spawn_env(purpose: str, *, project_root: Optional[Path] = None) -> dict[str, str]:
    """Env fragment a spawner merges into a child's environment."""
    return {SPAWN_ENV_VAR: build_spawn_tag(purpose, project_root=project_root)}


def parse_spawn_tag(raw: object) -> Optional[SpawnTag]:
    """Parse a ``HERMES_SPAWN`` value; ``None`` for anything malformed."""
    parts = raw.split(":") if isinstance(raw, str) else []
    if len(parts) != 5 or parts[0] != _TAG_VERSION:
        return None
    _, install, purpose, pid_s, create_s = parts
    if not install or not purpose:
        return None
    try:
        pid = int(pid_s)
        create = None if create_s == "-" else float(create_s)
    except ValueError:
        return None
    return SpawnTag(install, purpose, pid, create) if pid > 0 else None


def spawned_by_this_process(pid: int) -> bool:
    """True when ``pid`` carries a spawn tag naming THIS process (pid and create time).

    A shared daemon (one session or pid file every Hermes of this user can reach) is owned only by
    the Hermes whose command actually started it: the CLI it ran was given ``spawn_env`` and the
    daemon inherited it. A pid file proves what a process is, not who started it. Fails closed:
    an unreadable environment (another user, a race) is not ours."""
    import psutil

    try:
        tag = parse_spawn_tag(psutil.Process(int(pid)).environ().get(SPAWN_ENV_VAR))
    except (psutil.Error, OSError):
        return False
    if tag is None or tag.spawner_pid != os.getpid():
        return False
    mine = _process_create_time()
    return tag.spawner_create is None or mine is None or abs(tag.spawner_create - mine) < 2.0


# Layer 2 — spawn ledger


@dataclass
class LedgerEntry:
    pid: int
    create_time: Optional[float]
    purpose: str
    install: str
    spawner_pid: Optional[int]
    spawner_create: Optional[float]
    registered_at: float
    argv: str
    # Structured launch identity a relauncher needs after an update, without parsing argv. Empty
    # for purposes that don't supply it; readers must use .get() — older ledger files predate these.
    host: str = ""
    port: Optional[int] = None
    profile: str = ""
    hermes_home: str = ""
    # `serve --isolated`: opted out of the host singleton (Desktop's SSH backend for another
    # machine). Attach-first readers must never adopt it; argv is truncated, so this is canonical.
    isolated: bool = False
    # Linux boot-relative births (:func:`boot_start`) of the process and its spawner: exact
    # identity across clock steps, within the boot named by ``boot_id`` (:func:`_entry_boot`).
    # ``None`` elsewhere and in older entries (epoch match only).
    boot_start: Optional[float] = None
    spawner_boot: Optional[float] = None
    boot_id: Optional[str] = None


def _ledger_path() -> Path:
    """Machine-root ledger path (shared by every profile of this install)."""
    try:
        from hermes_constants import get_default_hermes_root

        return Path(get_default_hermes_root()) / LEDGER_FILENAME
    except Exception:
        from hermes_cli.config import get_hermes_home

        return Path(get_hermes_home()) / LEDGER_FILENAME


def _read_ledger(path: Path) -> Optional[list[dict]]:
    """Entries list, ``[]`` for empty/missing, ``None`` for CORRUPT (never silently an empty roster).

    Mirrors the #89298 contract: corrupt is a distinct state that must never be silently treated as an empty
    roster.
    """
    try:
        text = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        return []
    except (OSError, UnicodeError):
        return None
    if not text.strip():
        return []
    try:
        parsed = json.loads(text)
    except (ValueError, TypeError):
        return None
    return [e for e in parsed if isinstance(e, dict)] if isinstance(parsed, list) else None


def _read_ledger_or_quarantine(path: Path) -> Optional[list[dict]]:
    """Ledger entries; ``None`` after parking a corrupt file. Caller holds ``_LEDGER_LOCK``."""
    entries = _read_ledger(path)
    if entries is None:
        parked = path.with_suffix(path.suffix + ".corrupt")
        try:
            os.replace(path, parked)
            logger.warning("spawn ledger was unreadable; moved to %s", parked)
        except OSError:
            pass
    return entries


def _same_incarnation(proc, create_time: Optional[float], boot: Optional[float] = None) -> bool:
    """Does the live ``proc`` match a recorded birth? A recorded ``boot`` (:func:`boot_start`)
    compares exactly where the live one is readable; otherwise ``create_time`` (2 s tolerance;
    ``None`` matches)."""
    if boot is not None and (live_boot := boot_start(proc)) is not None:
        return live_boot == float(boot)
    return create_time is None or abs(process_birth(proc) - float(create_time)) < 2.0


def _pid_alive_matches(
    pid: int, create_time: Optional[float], *, strict: bool = False, boot: Optional[float] = None,
) -> Optional[bool]:
    """True/False when provable; ``None`` when psutil can't say."""
    try:
        import psutil
    except Exception:
        return None
    try:
        proc = psutil.Process(int(pid))
        # ``strict`` is positive proof (an exact epoch birth); a clock step only withholds it.
        if strict:
            # Windows: skip the status() read entirely. psutil's Windows status path
            # (NtQuerySystemInformation) does not release the GIL, so under load it freezes every
            # other Python thread in the process for minutes (#142085) — and Windows has no POSIX
            # zombies, so the STATUS_ZOMBIE comparison had nothing to contribute there anyway. The
            # non-strict branch right below already makes this exact skip.
            if os.name == "nt":
                return (create_time is not None and process_birth(proc) == create_time
                        and proc.is_running())
            return (create_time is not None and process_birth(proc) == create_time
                    and proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE)
        # A zombie keeps its create_time until its parent reaps it, but it is already dead: the
        # dashboard stop check (``gateway.status._pid_exists``) books it stopped, so the ledger must
        # not report it as a live pre-update survivor. Windows has no zombies (and status() is slow there).
        return _same_incarnation(proc, create_time, boot) and (
            os.name == "nt" or proc.status() != getattr(psutil, "STATUS_ZOMBIE", "zombie"))
    except psutil.NoSuchProcess:
        return False
    except Exception:
        return None


def register_self(purpose: str, *, project_root: Optional[Path] = None, detail: Optional[dict] = None) -> bool:
    """Record this process in the machine spawn ledger. Best-effort.

    Called at the top of every long-lived entry point; dead ``(pid, create_time)`` entries are
    pruned on every write. ``detail`` may carry ``host``/``port``/``profile`` so the update
    pipeline can relaunch a manually-started serve with its real bind address, and ``isolated``
    so attach-first discovery skips a backend that opted out of the host singleton.
    """
    from hermes_constants import hermes_home_key

    tag = parse_spawn_tag(os.environ.get(SPAWN_ENV_VAR))
    spawner_pid, spawner_create = (tag.spawner_pid, tag.spawner_create) if tag else _desktop_spawner_identity()
    entry = _new_entry(os.getpid(), _process_create_time(), purpose, project_root, spawner_pid, spawner_create)
    entry.spawner_boot = _verified_boot_start(spawner_pid, spawner_create)
    entry.hermes_home = hermes_home_key()
    if detail:
        try:
            entry.host = str(detail.get("host") or "")
            entry.port = int(detail["port"]) if detail.get("port") is not None else None
            entry.profile = str(detail.get("profile") or "")
            entry.isolated = bool(detail.get("isolated"))
        except (TypeError, ValueError):
            pass
    try:
        import sys as _sys

        # 10 tokens: enough for `hermes serve --host X --port N --profile P` while bounding
        # pathological argv. Structured detail is canonical; argv is the human-readable fallback.
        entry.argv = " ".join(_sys.argv[:10])
    except Exception:
        pass
    return _append_entry(entry)


def is_desktop_owned_backend(argv: Optional[Sequence[str]] = None) -> bool:
    """Whether this process is the backend Desktop spawned and owns.

    ``HERMES_DESKTOP=1`` is inherited by every shell and agent child the app launches, so the
    flag alone is not ownership proof (same class as #116107). Desktop hands its backend a
    per-spawn credential the terminal pane never receives (and the terminal tool's env policy
    strips from agent children): the local pool spawn mints ``HERMES_DASHBOARD_SESSION_TOKEN``,
    the SSH spawn passes a 0600 token FILE on argv and deliberately sets no token env var.
    ``argv`` defaults to this process's own.
    """
    if os.environ.get("HERMES_DESKTOP") != "1":
        return False
    if os.environ.get("HERMES_DASHBOARD_SESSION_TOKEN"):
        return True
    from hermes_cli._startup_fast import is_desktop_ssh_backend_argv

    return is_desktop_ssh_backend_argv(list(sys.argv[1:] if argv is None else argv))


# Set while this process works for the Desktop app on its own machine: one request from, or one turn of
# a chat created by, a client ``local_desktop_client`` accepted. The gateway serves the Desktop, the CLI,
# the TUI and bots from one process, so this is a property of the activity, never of the process.
_SERVING_LOCAL_DESKTOP: ContextVar[bool] = ContextVar("hermes_serving_local_desktop", default=False)


@contextmanager
def serving_local_desktop(active: bool) -> Iterator[None]:
    """Mark the enclosed activity (and threads it hands its context to) as work for the local Desktop."""
    token = _SERVING_LOCAL_DESKTOP.set(bool(active) or _SERVING_LOCAL_DESKTOP.get())
    try:
        yield
    finally:
        _SERVING_LOCAL_DESKTOP.reset(token)


def serves_local_desktop_now() -> bool:
    return _SERVING_LOCAL_DESKTOP.get()


def local_desktop_client(identity: Optional[dict], declared: object) -> bool:
    """A connected client is the Desktop app on this machine when it says so (it knows its connection
    mode) AND it redeemed a native ticket, which only this machine's private control socket mints, for
    this OS user. Neither alone: the CLI and TUI hold native tickets too, and the claim is the client's
    word. A remote Desktop reaches a backend through SSH, URL or Cloud credentials, never a native ticket."""
    return (declared is True and isinstance(identity, dict) and identity.get("provider") == "local"
            and identity.get("native_bootstrap") is True)


def _desktop_spawner_identity() -> tuple[Optional[int], Optional[float]]:
    """Spawner ``(pid, create_time)`` from the Electron app's HERMES_PARENT_PID (+ optional
    ``winms:<ms>`` start marker) parent-death watchdog vars, so ledger lineage works with every
    Desktop version without a TS change. ``(None, None)`` when absent/malformed."""
    try:
        spawner_pid = int(os.environ.get("HERMES_PARENT_PID", ""))
    except (TypeError, ValueError):
        spawner_pid = 0
    if spawner_pid <= 0:
        return None, None
    marker = os.environ.get("HERMES_PARENT_START_MARKER", "")
    if not marker.startswith("winms:"):
        return spawner_pid, None
    try:
        return spawner_pid, float(marker.split(":", 1)[1]) / 1000.0
    except (ValueError, IndexError):
        return spawner_pid, None


def _new_entry(
    pid: int, create_time: Optional[float], purpose: str, project_root: Optional[Path],
    spawner_pid: Optional[int], spawner_create: Optional[float],
) -> LedgerEntry:
    return LedgerEntry(
        pid, create_time, purpose, install_id(project_root), spawner_pid, spawner_create, time.time(), argv="",
        boot_start=_verified_boot_start(pid, create_time), boot_id=_live_boot_id(),
    )


# A ledger writer stopped mid-write (SIGSTOP, a debugger) must not stall every spawn_owned on the
# machine: past this the write goes ahead under the thread lock alone, as when no lock can be taken.
_LEDGER_LOCK_TIMEOUT_S = 2.0
_ledger_lock_warned = False


def _append_entry(entry: LedgerEntry) -> bool:
    """Prune dead entries and append ``entry`` — the ONLY ledger write path.

    Serialized under ``_LEDGER_LOCK`` and, across processes, an exclusive lock on a sidecar file
    (every ``spawn_owned`` writes here, so concurrent Hermes processes otherwise dropped each
    other's rows), with an atomic tmp+replace; no writer touches the file outside this function.

    See #91660.
    """
    from hermes_cli.active_sessions import _FileLock

    path = _ledger_path()
    with _LEDGER_LOCK, ExitStack() as held:
        try:
            held.enter_context(_FileLock(path.with_name(path.name + ".lock"), timeout=_LEDGER_LOCK_TIMEOUT_S))
        except (RuntimeError, OSError):
            # Best-effort like the write itself: the thread lock is then the only guard, so a row another
            # Hermes process writes meanwhile can be lost. Say so once where it shows (agent.log).
            global _ledger_lock_warned
            if not _ledger_lock_warned:
                _ledger_lock_warned = True
                logger.warning("spawn ledger lock unavailable (a stalled holder, or no file locks here); writing "
                               "without it, so concurrent Hermes processes may drop each other's rows")
            logger.debug("spawn ledger lock unavailable", exc_info=True)
        entries = _read_ledger_or_quarantine(path) or []
        # Drop malformed entries, our own stale entry, and provably dead pids.
        pruned = [
            e for e in entries
            if isinstance(e.get("pid"), int)
            and e["pid"] != entry.pid
            and entry_still_running(e)
        ]
        pruned.append(asdict(entry))
        try:
            from hermes_constants import mkdir_under_hermes_home
            mkdir_under_hermes_home(path.parent)
            # argv may carry surrogate-escaped bytes (non-UTF-8 paths); ensure_ascii keeps the
            # utf-8 text handle from raising UnicodeEncodeError (a ValueError, not an OSError).
            atomic_json_write(path, pruned, mode=0o600, ensure_ascii=True)
            return True
        except OSError:
            logger.debug("spawn ledger write failed", exc_info=True)
            return False


def register_child(pid: int, purpose: str, *, project_root: Optional[Path] = None) -> bool:
    """Record a CHILD process this process just spawned. Best-effort.

    Mirror of :func:`register_self` for children that cannot register themselves (stdio MCP
    helpers: arbitrary ``npx``/binary servers never import Hermes code). Records the child's
    ``(pid, create_time)`` with THIS process as spawner, so a helper whose spawner is provably gone
    is a reapable orphan and one whose spawner is alive is never reaped.
    """
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    child_create = _process_create_time(pid) if pid > 0 else None
    if child_create is None:
        return False
    entry = _new_entry(pid, child_create, purpose, project_root, os.getpid(), _process_create_time())
    entry.spawner_boot = _verified_boot_start(os.getpid(), entry.spawner_create)
    try:
        import psutil

        entry.argv = " ".join(psutil.Process(pid).cmdline()[:10])
    except Exception:
        pass
    return _append_entry(entry)


def ledger_entries(
    *, project_root: Optional[Path] = None, all_installs: bool = False, verified_only: bool = False,
) -> list[dict]:
    """Ledger entries for this install, or all installs when explicitly requested.

    ``verified_only`` requires an exact live PID/create-time pair. The default preserves
    unknown processes for reapers, which must not mistake missing proof for a dead process.

    Entries whose ``(pid, create_time)`` no longer matches a live process are excluded (PID reuse reads as
    dead, thanks to the create-time pair). A corrupt ledger is quarantined and read as empty — identical
    philosophy to the backend-ownership fix (#89298): never let corruption erase or fake a roster; never let
    it block the caller either.
    """
    want_install = install_id(project_root)
    with _LEDGER_LOCK:
        entries = _read_ledger_or_quarantine(_ledger_path())
    if entries is None:
        return []
    return [
        e for e in entries
        if (all_installs or e.get("install") == want_install)
        and isinstance(e.get("pid"), int)
        and (_pid_alive_matches(e["pid"], e.get("create_time"), strict=True) is True
             if verified_only else _pid_alive_matches(e["pid"], e.get("create_time"), boot=_entry_boot(e, "boot_start")) is not False)
    ]


def entry_still_running(entry: dict) -> bool:
    """Is the ledger *entry*'s process still the recorded incarnation (``None`` from psutil
    counts as running)? Exact boot ticks within their boot survive a clock step; an entry
    without them keeps the 2 s epoch match, so a step reads it as a reused pid."""
    pid = entry.get("pid")
    if not isinstance(pid, int) or pid <= 0:
        return False
    return _pid_alive_matches(pid, entry.get("create_time"), boot=_entry_boot(entry, "boot_start")) is not False


def spawner_is_dead(entry: dict) -> Optional[bool]:
    """Is the recorded spawner of this entry provably gone? ``None`` when unrecorded/unprovable."""
    spawner_pid = entry.get("spawner_pid")
    if not isinstance(spawner_pid, int) or spawner_pid <= 0:
        return None
    # A spawner recorded in an earlier boot is dead (:func:`_entry_boot`), as the epoch match says.
    boot = _entry_boot(entry, "spawner_boot")
    alive = _pid_alive_matches(spawner_pid, entry.get("spawner_create"), boot=boot)
    return None if alive is None else not alive


def _terminate_then_kill(pid: int, create_time: Optional[float], boot: Optional[float] = None) -> bool:
    """SIGTERM, 2 s grace, then SIGKILL. False when the incarnation moved on or is already gone."""
    try:
        import psutil

        proc = psutil.Process(pid)
        if not _same_incarnation(proc, create_time, boot):
            return False  # PID reused since registration — never signal a stranger
        proc.terminate()
        try:
            proc.wait(timeout=2.0)
        except psutil.TimeoutExpired:
            proc.kill()
        return True
    except Exception:
        logger.debug("orphan terminate failed for pid %s", pid, exc_info=True)
        return False


def _reparented_orphan(pid: int) -> bool:
    """Null-spawner owner with no live supervisor: reparented to init, old, lock-unclaimed.

    Mirrors ``_reap_orphaned_desktop_local_serves``: operator-managed remote backends legitimately
    sit at ppid 1 (systemd, an exited sshd), so ppid alone is never proof — the SSH-lock claim
    and the 180 s lock-write grace exclude live-supervised backends. Windows: the ppid probe is
    unavailable there, so this rung never fires (desktop tree-kill reaps instead).
    """
    try:
        from hermes_cli.dashboard_procs import (
            _REAP_MIN_AGE_SECONDS, _lock_owned_serve_pids, _process_ppid)
        if _process_ppid(pid) not in (0, 1):
            return False
        if pid in (_lock_owned_serve_pids() or ()):
            return False
        import time as _time

        import psutil
        return max(0.0, _time.time() - psutil.Process(pid).create_time()) >= _REAP_MIN_AGE_SECONDS
    except Exception:
        return False  # unprovable → never touch


def reap_orphaned_backend_owner(
    pid: int, create_time: Optional[float], *, kill_fn=None) -> Optional[int]:
    """Kill the conflicting host owner when the ledger PROVES it is a dead session's orphan.

    HELD_BY_OTHER rung for #121964: the owner PID is alive (re-parented to init), so
    ``record_is_stale()`` never fires and a fresh backend would loop observe-only against the
    same orphan forever. Reaped only with a live ``serve``/``dashboard``/``gateway`` entry for
    THIS install whose spawner is provably dead — or whose spawner was never recorded and which
    is reparented to init, old, and claimed by no Desktop SSH lock. Live-spawned, lock-claimed,
    young, interactive, self, or unprovable owners are never touched. Returns the reaped pid,
    else ``None``; never raises.

    # ponytail: a dead session's surviving backend.lock.json still shields its orphan (the lock
    # has no session liveness); clearing stale locks is a Desktop-side lifecycle job, not this rung.
    """
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return None
    if pid <= 0 or pid == os.getpid():
        return None
    try:
        entry = next(
            (e for e in ledger_entries()
             if e.get("pid") == pid
             and e.get("purpose") in _BACKEND_OWNER_PURPOSES
             and (e.get("create_time") is None or create_time is None
                  or abs(float(e["create_time"]) - float(create_time)) < 2.0)),
            None)
    except Exception:
        return None
    if entry is None:
        return None
    if spawner_is_dead(entry) is not True and not (
            entry.get("spawner_pid") is None and _reparented_orphan(pid)):
        return None  # live or unprovable spawner → never touch
    try:
        if kill_fn is not None:
            kill_fn(pid)
        elif not _terminate_then_kill(pid, entry.get("create_time"), _entry_boot(entry, "boot_start")):
            return None
    except Exception:
        logger.debug("backend owner reap failed for %s", entry, exc_info=True)
        return None
    logger.info("reaped orphaned %s backend owner pid %s", entry.get("purpose"), pid)
    return pid


def reap_orphaned_mcp_helpers(*, project_root: Optional[Path] = None, kill_fn=None) -> list[int]:
    """Kill ledger-registered stdio MCP helpers whose spawner is provably dead.

    Ledger-driven startup-sweep rung (not cmdline-heuristic): a helper is reaped ONLY when it has a
    live ``mcp-helper`` entry for THIS install AND ``spawner_is_dead`` is ``True`` — never
    ``None``/unprovable, never a live spawner.
    """
    reaped: list[int] = []
    try:
        entries = ledger_entries(project_root=project_root)
    except Exception:
        return reaped
    own_pid = os.getpid()
    for entry in entries:
        try:
            pid = entry.get("pid")
            if entry.get("purpose") != "mcp-helper" or not isinstance(pid, int) or pid <= 0 or pid == own_pid:
                continue
            if spawner_is_dead(entry) is not True:
                continue  # live or unprovable spawner → never touch
            if kill_fn is not None:
                kill_fn(pid)
            else:
                import psutil

                proc = psutil.Process(pid)
                if not _same_incarnation(proc, entry.get("create_time"), _entry_boot(entry, "boot_start")):
                    continue  # PID reused since registration
                # Windows: descendants (npx.cmd → node.exe) have no pgid to group-kill and
                # reparent with ParentId=null when the direct child exits first (#61059), so
                # reap the whole tree. On POSIX the killpg-based sweep already reaches them.
                if _IS_WINDOWS:
                    _kill_process_tree_windows(proc)
                else:
                    proc.terminate()
                    try:
                        proc.wait(timeout=2.0)
                    except psutil.TimeoutExpired:
                        proc.kill()
            reaped.append(pid)
        except Exception:
            logger.debug("mcp-helper orphan reap failed for %s", entry, exc_info=True)
    if reaped:
        logger.info("reaped %d orphaned stdio MCP helper(s): %s", len(reaped), reaped)
    return reaped


# Layer 3 — Windows job-object self-attach


def _kill_process_tree_windows(proc) -> None:
    """Terminate *proc* and every still-alive descendant (npx.cmd → node.exe), Windows-only
    (#61059): without a pgid there is no group-kill, and grandchildren reparent to nothing
    (ParentId=null) once the direct child exits, so they must be reached through the tree."""
    import psutil

    try:
        descendants = proc.children(recursive=True)
    except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
        descendants = []
    for child in descendants:
        try:
            child.terminate()
        except Exception:
            pass
    try:
        proc.terminate()
    except Exception:
        pass
    try:
        _, alive = psutil.wait_procs(descendants + [proc], timeout=2.0)
    except Exception:
        return
    for survivor in alive:
        try:
            survivor.kill()
        except Exception:
            pass


def attach_self_to_kill_on_close_job() -> bool:
    """Place this process in a job that dies (whole tree) when we die. Windows-only, idempotent.

    ``BREAKAWAY_OK`` keeps children spawned with ``CREATE_BREAKAWAY_FROM_JOB`` (gateway relaunch
    during update, detached watchers) escaping exactly as before.
    """
    global _JOB_HANDLE
    if not _IS_WINDOWS or _JOB_HANDLE is not None:
        return _JOB_HANDLE is not None
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
        JOB_OBJECT_LIMIT_BREAKAWAY_OK = 0x0800
        JOB_OBJECT_LIMIT_SILENT_BREAKAWAY_OK = 0x1000
        JobObjectExtendedLimitInformation = 9

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [(n, ctypes.c_ulonglong) for n in (
                "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

        class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", wintypes.LARGE_INTEGER), ("PerJobUserTimeLimit", wintypes.LARGE_INTEGER),
                ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.POINTER(wintypes.ULONG)), ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
                ("IoInfo", IO_COUNTERS),
                *((n, ctypes.c_size_t) for n in (
                    "ProcessMemoryLimit", "JobMemoryLimit", "PeakProcessMemoryUsed", "PeakJobMemoryUsed")),
            ]

        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            return False
        info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = (
            JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE | JOB_OBJECT_LIMIT_BREAKAWAY_OK | JOB_OBJECT_LIMIT_SILENT_BREAKAWAY_OK
        )
        ok = kernel32.SetInformationJobObject(job, JobObjectExtendedLimitInformation, ctypes.byref(info), ctypes.sizeof(info))
        if not ok or not kernel32.AssignProcessToJobObject(job, kernel32.GetCurrentProcess()):
            kernel32.CloseHandle(job)
            return False
        _JOB_HANDLE = job  # keep alive for the life of the process — never close
        logger.debug("attached to kill-on-close job object")
        return True
    except Exception:
        logger.debug("job object self-attach failed", exc_info=True)
        return False


# Layer 4 — POSIX owner-death supervisor
#
# If this process dies without running its cleanup path (SIGKILL, OOM killer, crash, force-quit),
# its long-lived children reparent to init or the nearest subreaper (systemd --user) and run
# forever; macOS has no PR_SET_PDEATHSIG, and on Linux it fires when the spawning THREAD exits,
# which kills children spawned from short-lived worker threads while the owner is alive. So ONE
# supervisor per Hermes process (``process_supervisor.sh``) holds the read end of a pipe whose only
# write end we hold: EOF means we died, by any cause, and it killpg()s every group still registered.
# A registered group that was never unregistered IS the orphan set, so a clean exit that stopped its
# children first ends with nothing to kill, and one that didn't takes them along.
#
# ``spawn_owned`` is THE way to start a long-lived child that must die with this process: the child
# leads its own process group, that group is handed to the supervisor, and the child is recorded in
# the spawn ledger. Children something else spawns (the MCP SDK, the browser-harness daemon) hand their
# group to ``own_process_groups`` where we learn the pgid; a descendant that leaves the group on
# purpose is registered by its own pgid the same way. Stopping an owned child yourself ends with
# ``release_process_group_once_empty`` (survivors stay covered); ``release_process_groups`` hands a
# group back on purpose. ``scripts/check_detached_spawns.py`` flags new detached spawns
# (start_new_session / process_group / setsid / setpgid; its docstring lists what it misses). POSIX-only: Windows children die
# through the layer-3 job object instead.

_supervisor = None  # Optional[subprocess.Popen]; tests rebind it
_supervisor_lock = threading.Lock()
# Groups the supervisor is reaping on our behalf; replayed verbatim on respawn so a respawn never
# silently drops coverage for children that are still running.
_supervised_pgids: set = set()
_SUPERVISOR_SCRIPT = Path(__file__).with_name("process_supervisor.sh")
# Android (Termux) has no /bin/sh; its system shell is the same POSIX sh the script needs.
_POSIX_SH = "/bin/sh" if os.path.exists("/bin/sh") else "/system/bin/sh"


def _spawn_supervisor():
    """Start the per-process supervisor, or None if it cannot be started."""
    try:
        # start_new_session=True is load-bearing: shutdown paths killpg this process's own group,
        # which would kill the supervisor before it could reap anything. close_fds keeps the write
        # end of every OTHER pipe we hold out of it, so it never delays another child's EOF.
        proc = subprocess.Popen(
            [_POSIX_SH, str(_SUPERVISOR_SCRIPT), str(os.getpgid(0))],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True, close_fds=True, text=True, encoding="utf-8")
    except Exception:
        # Never let supervisor bookkeeping block a real child: graceful shutdown paths still reap
        # normally; only the ungraceful-exit safety net is lost.
        logger.debug("Could not start the owner-death supervisor", exc_info=True)
        return None
    _watch_supervisor(proc)
    return proc


# A supervisor that dies sooner than this after starting may not be able to stay up (a broken install:
# sh exiting 2 or 127; a sandbox or OOM killer that keeps killing it): the waiter leaves it to the next
# registration change instead of restarting it in a loop. One killed by a signal gets one young restart
# (a new supervisor starts after every idle period, so a `pkill -f hermes` often meets a young one); if
# its replacement is killed young as well, that one waits for the next registration change too.
_SUPERVISOR_RESTART_MIN_UPTIME_S = 5.0


# Set (under _supervisor_lock) while the waiter starts the young restart, so that replacement's own
# waiter knows it already had one.
_starting_young_restart = False


def _watch_supervisor(proc) -> None:
    """One waiter thread per supervisor, blocked in the untimed ``wait()`` (waitpid, no polling). It
    reaps the supervisor the moment it exits, so no zombie child is left for psutil walkers of this
    process's children to trip on. A supervisor that died on its own while groups are still registered
    (``pkill -f hermes``, the OOM killer) is restarted and replayed at once, so coverage does not wait
    for the next registration change; a released one (EOF with an empty set) is only reaped."""
    started = time.monotonic()
    young_restart = _starting_young_restart

    def wait_then_restart() -> None:
        global _supervisor, _starting_young_restart
        proc.wait()
        with _supervisor_lock:
            if _supervisor is not None and _supervisor is not proc:
                return  # already replaced: only reaped
            # Still current, or dropped by a caller whose write hit its broken pipe: restart below if
            # groups are still registered (a deliberate release leaves the set empty).
            _supervisor = None
            _prune_dead_supervised_pgids()
            if not _supervised_pgids:
                return
            old_enough = time.monotonic() - started >= _SUPERVISOR_RESTART_MIN_UPTIME_S
            young_signal = not old_enough and proc.returncode < 0 and not young_restart
            restart = old_enough or young_signal
            logger.warning(
                "owner-death supervisor exited (status %s) with %d process group(s) registered; %s",
                proc.returncode, len(_supervised_pgids),
                "restarting it" if restart else "it could not stay up, retrying at the next registration")
            if restart:
                _starting_young_restart = young_signal
                try:
                    _start_supervisor_locked()
                finally:
                    _starting_young_restart = False

    try:
        # health: allow HX012 -- the thread waitpid()s one child and touches only process-global supervisor state
        threading.Thread(target=wait_then_restart, name="owner-death-supervisor-waiter", daemon=True).start()
    except RuntimeError:  # "can't start new thread" (thread limit, interpreter shutdown)
        # The supervisor still works; it is left a zombie until we exit and is not restarted if killed.
        logger.debug("Could not start the owner-death supervisor's waiter", exc_info=True)


# An idle supervisor (nothing registered) exits this long after the set last emptied, unless a new
# registration arrives first.
_SUPERVISOR_IDLE_EXIT_S = 30.0
_idle_release: Optional[threading.Timer] = None


def _release_when_idle_locked(proc) -> None:
    """Close ``proc``'s pipe once the set has stayed empty for ``_SUPERVISOR_IDLE_EXIT_S``; the caller
    holds ``_supervisor_lock``. Closing our write end is the same EOF our death sends; with an empty
    set it exits, and its waiter thread reaps it. The next registration respawns and replays."""
    global _idle_release
    _cancel_idle_release_locked()
    if _SUPERVISOR_IDLE_EXIT_S <= 0:
        _close_idle_supervisor_locked(proc)
        return

    def release() -> None:
        with _supervisor_lock:
            _close_idle_supervisor_locked(proc)

    try:
        _idle_release = threading.Timer(_SUPERVISOR_IDLE_EXIT_S, release)
        _idle_release.daemon = True
        _idle_release.start()
    except RuntimeError:
        # No thread for the grace (thread limit, interpreter shutdown): release at once, as before the
        # grace existed. Losing the grace must never fail the live child this call is about.
        _idle_release = None
        logger.debug("Could not start the supervisor idle timer; releasing it now", exc_info=True)
        _close_idle_supervisor_locked(proc)


def _close_idle_supervisor_locked(proc) -> None:
    global _supervisor
    if _supervisor is not proc or _supervised_pgids:
        return  # replaced, or something registered meanwhile
    with suppress(BrokenPipeError, ValueError, OSError):
        proc.stdin.close()
    _supervisor = None


def _cancel_idle_release_locked() -> None:
    global _idle_release
    if _idle_release is not None:
        _idle_release.cancel()
        _idle_release = None


def _start_supervisor_locked():
    """Start a supervisor and replay every live registration to it; the caller holds
    ``_supervisor_lock``. Returns it, or None when it cannot start or its pipe broke at once."""
    global _supervisor
    proc = _supervisor = _spawn_supervisor()
    if proc is None:
        return None
    try:
        proc.stdin.write("".join(f"register {pgid}\n" for pgid in _supervised_pgids))
        proc.stdin.flush()
    except (BrokenPipeError, ValueError, OSError):
        _supervisor = None  # its waiter (if it has one) reaps it; the next registration change retries
        return None
    return proc


def is_owner_death_supervisor(argv: Sequence[str]) -> bool:
    """True when a direct child's ``argv`` is this install's supervisor (``sh <script> <pgid>``).
    Another family's child discovery (MCP's before/after child-pid delta) can catch it starting and
    must never adopt, and later kill, it. Matched on the script path, not ``_supervisor.pid``: that
    is assigned only after the spawn returns, so a scan racing the spawn would miss it."""
    return list(argv[1:2]) == [str(_SUPERVISOR_SCRIPT)]


def _prune_dead_supervised_pgids() -> set:
    """Forget supervised groups with no members left; return what went. Caller holds
    ``_supervisor_lock``. Signal 0 is a pure existence probe (cannot terminate anything). It
    narrows, but cannot close, the window where a dead group's pgid is recycled before we notice."""
    stale = set()
    for pgid in list(_supervised_pgids):
        try:
            os.killpg(pgid, 0)  # windows-footgun: ok — POSIX-only, callers return early on nt
        except ProcessLookupError:
            stale.add(pgid)
        except (PermissionError, OSError):
            # Exists but not ours to signal, or the probe failed: keep it — dropping coverage on
            # an ambiguous answer is the more expensive mistake.
            pass
    _supervised_pgids.difference_update(stale)
    return stale


def own_process_groups(pgids) -> None:
    """Make these process groups die with this process, however it dies (see layer 4 above)."""
    _update_supervised_pgids("register", pgids)


def release_process_groups(pgids) -> None:
    """Stop covering these groups: this process stopped them itself."""
    _update_supervised_pgids("unregister", pgids)


def release_process_group_once_empty(pgid: int, *, proc=None, timeout: float = 1.0) -> bool:
    """Release ``pgid`` once no member is left; True if released. Polls ``proc`` (its leader) so a
    dead-but-unreaped leader does not keep the group alive, and waits up to ``timeout`` for killed
    members to be reaped by init. A member still there after that keeps the group covered, so it
    still dies with this process. ``timeout=0`` probes once (async callers)."""
    if os.name != "posix":
        return False
    deadline = time.monotonic() + timeout
    while True:
        if proc is not None:
            try:
                proc.poll()
            except OSError:
                pass
        try:
            os.killpg(pgid, 0)  # windows-footgun: ok — POSIX-only, returned above; existence probe
        except ProcessLookupError:
            release_process_groups([pgid])
            return True
        except OSError:
            pass  # exists but not ours to probe: keep covering it
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)


def stop_owned_group_survivors(pgid: int, *, proc=None, grace: float = 3.0) -> bool:
    """Stop what is left of owned group ``pgid`` after its leader exited, then release it; True once
    released. Members that outlived a crashed leader (a router's model children) keep the group, and
    so its number, alive: TERM, wait ``grace``, KILL. Only groups this process registered are
    signalled, so a number never registered here (or already released) is never touched."""
    if os.name != "posix":
        return False
    with _supervisor_lock:
        owned = pgid in _supervised_pgids
    if owned:
        for sig, wait in ((signal.SIGTERM, grace), (signal.SIGKILL, 1.0)):  # windows-footgun: ok — POSIX-only, returned above
            if release_process_group_once_empty(pgid, proc=proc, timeout=0):
                return True
            try:
                os.killpg(pgid, sig)  # windows-footgun: ok — POSIX-only, returned above
            except ProcessLookupError:
                pass
            except OSError:
                return False  # not ours to signal: keep covering it
            if release_process_group_once_empty(pgid, proc=proc, timeout=wait):
                return True
        return False
    return release_process_group_once_empty(pgid, proc=proc, timeout=0)


def spawn_owned(argv, *, purpose: str, **popen_kwargs) -> subprocess.Popen:
    """Start a long-lived child that dies with this process: the one way to own a child.

    POSIX: the child leads a new session and process group, and that group goes to the owner-death
    supervisor, so the child and every descendant that stays in its group die the moment this
    process dies by any cause (SIGKILL, OOM, crash). The child is recorded in the spawn ledger under
    ``purpose``. After stopping it yourself, ``release_process_group_once_empty(proc.pid, proc=proc)``.
    Windows: a plain ``Popen`` with the caller's kwargs (the layer-3 job object covers ownership).
    """
    if os.name != "posix":
        return subprocess.Popen(argv, **popen_kwargs)
    proc = subprocess.Popen(argv, start_new_session=True, **popen_kwargs)
    own_process_groups([proc.pid])
    register_child(proc.pid, purpose)
    return proc


def _update_supervised_pgids(verb: str, pgids) -> None:
    """Send ``register``/``unregister`` for ``pgids`` to this process's supervisor. Failures are
    swallowed: losing the safety net must never fail a live child."""
    if os.name != "posix":
        return
    wanted = {int(pgid) for pgid in pgids}
    if not wanted:
        return

    global _supervisor
    with _supervisor_lock:
        if verb == "register":
            _supervised_pgids.update(wanted)
        else:
            _supervised_pgids.difference_update(wanted)

        # A registration outlives its child only while some member survives (e.g. an orphaned
        # grandchild teardown failed to kill, deliberately kept registered). Once that group is
        # empty its pgid can be recycled by a stranger, so prune here too.
        stale = _prune_dead_supervised_pgids()

        proc = _supervisor
        if proc is None or proc.poll() is not None:
            if not _supervised_pgids:
                # Nothing left to cover: nothing to tell and nothing to respawn for. Keyed on the
                # SET, not the verb: after a broken-pipe write dropped the supervisor with groups
                # still registered, an unregister must still rebuild coverage for the survivors.
                return
            # A fresh supervisor knows nothing: replay live coverage (already reflects this call's
            # mutation and the prune, so pruned groups never reach the replacement).
            _start_supervisor_locked()
            return
        if _supervised_pgids:
            _cancel_idle_release_locked()
        payload = "".join(f"{verb} {pgid}\n" for pgid in wanted)
        payload += "".join(f"unregister {pgid}\n" for pgid in stale)

        try:
            proc.stdin.write(payload)
            proc.stdin.flush()
        except (BrokenPipeError, ValueError, OSError):
            # It exited between poll() and write(). Drop it so the next call respawns and replays
            # from ``_supervised_pgids`` (the set, not the pipe, is the record of what to reap). Its
            # waiter thread reaps it.
            _supervisor = None
            return

        if not _supervised_pgids:
            # Nothing left to reap: release the supervisor after an idle grace rather than keep a
            # process and a pipe resident for the life of a gateway. Not at once: a foreground
            # command owns and releases its group every time, and a supervisor per command costs a
            # /bin/sh spawn (~11 ms) under this lock.
            _release_when_idle_locked(proc)
