"""
Disk retention & low-space guardian for HERMES_HOME (OOF-250 / OOF-269).

Hosted Hermes Cloud instances run on small persistent volumes (~1 GB).
Recurring fleet incidents show unbounded writers filling the disk (ENOSPC),
which cascades into SQLite "database or disk is full" errors, dashboard ASGI
exceptions, and an unhealthy gateway — while status endpoints still look
green.  This module provides the systemic fix:

1. ``truncate_log_tail()`` — in-place tail truncation for append-only
   diagnostic logs.  Truncating in place (rather than unlinking) frees space
   even when another process holds an open file descriptor — the
   "deleted-but-open file" failure mode from OOF-2.  Appenders using
   ``open(..., "a")`` (O_APPEND) continue working seamlessly.

2. ``prune_files()`` — age/count/total-size pruning for backup and cache
   file families (oldest first).

3. ``run_retention_sweep()`` — one cheap, exception-proof pass over all
   known-safe families under HERMES_HOME.  Never touches user data
   (sessions, state.db, memories, configured media); only diagnostic logs,
   stale DB-recovery backups, and regenerate-on-demand caches.

4. ``disk_status()`` — low-space signal for health/status endpoints so
   /api/status and /health/detailed report degraded BEFORE ENOSPC hits.

5. ``disk_usage_summary()`` — per-family byte usage for diagnostics.

The gateway cron ticker calls ``sweep_and_log()`` periodically (see
``gateway/run.py``).  All entry points are exception-proof: a retention
failure must never crash the host process.

Config lives under the ``disk`` section of config.yaml (see
``hermes_cli.config.DEFAULT_CONFIG``); every knob has a safe default sized
for a 1 GB volume.
"""

from __future__ import annotations

import logging
import os
import shutil
import stat as stat_mod
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

_TRUNCATION_MARKER = b"[...older lines removed by hermes disk retention...]\n"

# Files that must NEVER be touched by any retention operation, regardless of
# what a (mis)configured glob resolves to.  Belt-and-braces guard on top of
# the explicit family definitions below.
_PROTECTED_NAMES = frozenset({
    "state.db", "state.db-wal", "state.db-shm",
    "hermes_state.db",
    "config.yaml", ".env", "auth.json", "SOUL.md",
    "gateway.pid", "gateway_state.json",
})

# Top-level HERMES_HOME directories that hold user data — pruning never
# descends into these.
_PROTECTED_DIRS = frozenset({
    "sessions", "memories", "cron", "hooks", "skins", "plans",
    "workspace", "profiles",
})

# logs/ files managed by RotatingFileHandler (hermes_logging.py).  These
# already rotate at their configured size; tail-truncating them would fight
# the handler.  Their numbered backups (agent.log.1, ...) are also skipped.
_ROTATION_MANAGED_LOGS = frozenset({"agent.log", "errors.log", "gateway.log"})


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_DEFAULT_DISK_CONFIG: Dict[str, Any] = {
    "retention": {
        "enabled": True,
        # Gateway cron ticker runs at 60s; sweep every N minutes.
        "sweep_interval_minutes": 30,
        # Un-rotated *.log files under logs/ (and other diag logs) are
        # tail-truncated when they exceed max_bytes, keeping keep_bytes.
        "diag_log_max_bytes": 2 * 1024 * 1024,
        "diag_log_keep_bytes": 256 * 1024,
        # Backstop age prune for media caches (cache/images, cache/audio,
        # cache/documents, cache/screenshots).  The gateway already prunes
        # images/documents at 24h; this catches audio (previously unbounded)
        # and anything the hourly pass missed.
        "cache_max_age_hours": 72,
        # state.db malformed-recovery backups (state.db.malformed-*).
        "db_backup_keep_count": 5,
        "db_backup_max_age_days": 14,
        # Package-manager caches under HERMES_HOME/home (npm _cacache,
        # pip cache) — content-addressed, regenerate on demand.
        "pkg_cache_max_age_days": 14,
    },
    "low_space": {
        # Degraded when EITHER threshold is crossed.
        "min_free_bytes": 200 * 1024 * 1024,
        "min_free_percent": 10.0,
    },
}


def get_disk_config() -> Dict[str, Any]:
    """Return the merged ``disk`` config section (safe defaults on failure)."""
    import copy
    merged = copy.deepcopy(_DEFAULT_DISK_CONFIG)
    try:
        from hermes_cli.config import load_config
        user = load_config().get("disk") or {}
        for section in ("retention", "low_space"):
            sub = user.get(section)
            if isinstance(sub, dict):
                merged[section].update(sub)
    except Exception:  # noqa: BLE001 — config problems must not break retention
        pass
    return _sanitize_disk_config(merged)


def _coerce_bool(value: Any, default: bool) -> bool:
    """Tolerant bool coercion: YAML-ish strings never silently mean True."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "yes", "on", "1"):
            return True
        if lowered in ("false", "no", "off", "0"):
            return False
    return default


def _coerce_number(value: Any, default: float, *, minimum: float = 0.0) -> float:
    """Coerce a config knob to a finite non-negative number, else default.

    Malformed values (strings, None, dicts, negatives, NaN) fall back to the
    shipped default with a WARNING — one bad knob must never disable or
    weaponize retention (e.g. a negative size truncating everything).
    """
    try:
        if isinstance(value, bool):
            raise TypeError("bool is not a size/duration")
        num = float(value)
        if num != num or num in (float("inf"), float("-inf")):
            raise ValueError("non-finite")
        if num < minimum:
            raise ValueError("below minimum")
        return num
    except (TypeError, ValueError):
        logger.warning(
            "disk retention config value %r invalid; using default %r",
            value,
            default,
        )
        return default


def _sanitize_disk_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Clamp every knob to a sane value so user config can't break the sweep."""
    d = _DEFAULT_DISK_CONFIG
    ret = cfg.get("retention") or {}
    low = cfg.get("low_space") or {}
    dret = d["retention"]
    dlow = d["low_space"]
    return {
        "retention": {
            "enabled": _coerce_bool(ret.get("enabled"), dret["enabled"]),
            "sweep_interval_minutes": int(_coerce_number(
                ret.get("sweep_interval_minutes"),
                dret["sweep_interval_minutes"], minimum=1)),
            "diag_log_max_bytes": int(_coerce_number(
                ret.get("diag_log_max_bytes"),
                dret["diag_log_max_bytes"], minimum=4096)),
            "diag_log_keep_bytes": int(_coerce_number(
                ret.get("diag_log_keep_bytes"),
                dret["diag_log_keep_bytes"], minimum=0)),
            "cache_max_age_hours": _coerce_number(
                ret.get("cache_max_age_hours"),
                dret["cache_max_age_hours"], minimum=1.0),
            "db_backup_keep_count": int(_coerce_number(
                ret.get("db_backup_keep_count"),
                dret["db_backup_keep_count"], minimum=0)),
            "db_backup_max_age_days": _coerce_number(
                ret.get("db_backup_max_age_days"),
                dret["db_backup_max_age_days"], minimum=1.0),
            "pkg_cache_max_age_days": _coerce_number(
                ret.get("pkg_cache_max_age_days"),
                dret["pkg_cache_max_age_days"], minimum=1.0),
        },
        "low_space": {
            "min_free_bytes": int(_coerce_number(
                low.get("min_free_bytes"), dlow["min_free_bytes"], minimum=0)),
            "min_free_percent": _coerce_number(
                low.get("min_free_percent"), dlow["min_free_percent"],
                minimum=0.0),
        },
    }


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------

def _is_protected(path: Path, home: Path) -> bool:
    """Return True when *path* must never be modified by retention."""
    if path.name in _PROTECTED_NAMES:
        return True
    try:
        resolved = path.resolve()
    except (ValueError, OSError):
        return True
    # A symlink whose TARGET is a protected file (e.g. logs/evil.log ->
    # ../state.db) must be refused even though the link name looks safe.
    if resolved.name in _PROTECTED_NAMES:
        return True
    try:
        rel = resolved.relative_to(home.resolve())
    except (ValueError, OSError):
        # Outside HERMES_HOME — refuse to touch it.
        return True
    return bool(rel.parts) and rel.parts[0] in _PROTECTED_DIRS


# ---------------------------------------------------------------------------
# Primitive operations
# ---------------------------------------------------------------------------

def truncate_log_tail(
    path: Path,
    *,
    max_bytes: int,
    keep_bytes: int,
    home: Optional[Path] = None,
) -> int:
    """Tail-truncate *path* in place when it exceeds *max_bytes*.

    Keeps the last *keep_bytes* (aligned to the next newline) prefixed with a
    truncation marker.  Returns bytes reclaimed (0 when under the cap).

    Known-lossy race: bytes appended by an O_APPEND writer between the tail
    read and ``truncate()`` are discarded with the head.  Acceptable for the
    diagnostic log families this sweeps (loss window is microseconds, files
    are best-effort forensics); do NOT point this at data that must survive.

    In-place truncation (open "rb+", rewrite, ftruncate) frees disk space
    immediately even if another process holds the file open — unlike unlink,
    which leaves a deleted-but-open file consuming space (OOF-2).

    Hard safety contract: the fd we mutate is verified to be a regular file
    with a single hard link, opened with O_NOFOLLOW.  A symlink placed in a
    swept directory (``logs/evil.log -> ../state.db``) is refused by the
    kernel; a hardlink to a protected file (``st_nlink > 1``) is refused by
    the fstat check.  All truncation happens through the verified fd — there
    is no path-based reopen after the check (no TOCTOU window).
    """
    home = home or get_hermes_home()
    if _is_protected(path, home):
        return 0
    try:
        lst = os.lstat(path)
    except OSError:
        return 0
    if stat_mod.S_ISLNK(lst.st_mode) or not stat_mod.S_ISREG(lst.st_mode):
        logger.warning("disk retention: refusing non-regular file %s", path)
        return 0
    if lst.st_nlink != 1:
        # A hardlink shares its inode with another name — possibly a
        # protected file. Never mutate multi-linked inodes.
        logger.warning(
            "disk retention: refusing hardlinked file %s (nlink=%d)",
            path, lst.st_nlink,
        )
        return 0

    nofollow = getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(str(path), os.O_RDWR | nofollow)
    except OSError as exc:
        logger.debug("truncate_log_tail(%s) open failed: %s", path, exc)
        return 0
    try:
        fst = os.fstat(fd)
        if not stat_mod.S_ISREG(fst.st_mode) or fst.st_nlink != 1:
            logger.warning(
                "disk retention: refusing %s post-open (mode=%o nlink=%d)",
                path, fst.st_mode, fst.st_nlink,
            )
            return 0
        size = fst.st_size
        if size <= max_bytes:
            return 0

        keep_bytes = max(0, min(keep_bytes, max_bytes))
        with os.fdopen(fd, "rb+", closefd=False) as f:
            f.seek(size - keep_bytes)
            tail = f.read(keep_bytes)
            # Start the kept tail at a line boundary for readable logs.
            nl = tail.find(b"\n")
            if 0 <= nl < len(tail) - 1:
                tail = tail[nl + 1:]
            f.seek(0)
            f.write(_TRUNCATION_MARKER)
            f.write(tail)
            f.truncate()
        new_size = os.fstat(fd).st_size
        return max(0, size - new_size)
    except OSError as exc:
        logger.warning("truncate_log_tail(%s) failed: %s", path, exc)
        return 0
    finally:
        try:
            os.close(fd)
        except OSError:
            pass


def prune_files(
    files: Iterable[Path],
    *,
    keep_count: Optional[int] = None,
    max_age_days: Optional[float] = None,
    max_total_bytes: Optional[int] = None,
    home: Optional[Path] = None,
) -> Tuple[int, int]:
    """Prune a family of files, oldest first.

    Retention order: the newest *keep_count* files are always kept, files
    older than *max_age_days* are removed, then oldest files are removed
    until the family fits in *max_total_bytes*.

    Returns ``(files_removed, bytes_reclaimed)``.
    """
    home = home or get_hermes_home()
    entries: List[Tuple[Path, float, int]] = []
    for p in files:
        if _is_protected(p, home):
            continue
        try:
            st = p.lstat()
        except OSError:
            continue
        # Regular files only: symlinks are never followed (deleting one is
        # harmless but its stat lies about size), and multi-linked inodes
        # share their data with another name — pruning them reclaims nothing
        # and may surprise the other owner.
        if not stat_mod.S_ISREG(st.st_mode) or st.st_nlink != 1:
            continue
        entries.append((p, st.st_mtime, st.st_size))

    # Newest first.
    entries.sort(key=lambda e: e[1], reverse=True)

    removed = 0
    reclaimed = 0
    now = time.time()
    keep = keep_count if keep_count is not None else 0

    survivors: List[Tuple[Path, float, int]] = []
    for idx, (p, mtime, size) in enumerate(entries):
        expired = (
            max_age_days is not None
            and (now - mtime) > max_age_days * 86400
        )
        if idx >= keep and expired:
            if _unlink(p):
                removed += 1
                reclaimed += size
                continue
        survivors.append((p, mtime, size))

    if max_total_bytes is not None:
        total = sum(size for _, _, size in survivors)
        # Remove oldest first (end of the newest-first list), but never dip
        # into the protected keep_count head.  A failed unlink is treated as
        # unreclaimable — we do NOT delete additional newer files to
        # compensate (that would over-delete while the cap is never met).
        while total > max_total_bytes and len(survivors) > keep:
            p, _, size = survivors.pop()
            if _unlink(p):
                removed += 1
                reclaimed += size
            total -= size

    return removed, reclaimed


def _unlink(path: Path) -> bool:
    try:
        path.unlink()
        return True
    except OSError as exc:
        logger.warning("retention unlink(%s) failed: %s", path, exc)
        return False


_DB_BACKUP_PREFIX = "state.db.malformed-backup-"
_DB_BACKUP_SIDECAR_SUFFIXES = ("-wal", "-shm")


def prune_db_backup_family(
    home: Path,
    *,
    keep_count: int,
    max_age_days: float,
) -> Tuple[int, int]:
    """Prune forensic state.db backups honouring the writer's contract.

    ``hermes_state._backup_db_file`` writes ``state.db.malformed-backup-<ts>``
    plus optional ``-wal``/``-shm`` sidecars.  A backup and its sidecars are
    ONE retention unit: keep-count is counted in backup *sets*, and a pruned
    base always takes its sidecars with it (no orphaned WAL/SHM files).

    Exact-prefix enumeration only — never a broad ``state.db.*`` glob, so
    unrelated prefix-neighbours (``state.db.repair-attempts.json``,
    ``state.db-wal``) can never match.
    """
    bases: List[Path] = []
    try:
        for p in home.iterdir():
            name = p.name
            if not name.startswith(_DB_BACKUP_PREFIX):
                continue
            if name.endswith(_DB_BACKUP_SIDECAR_SUFFIXES):
                continue
            try:
                st = p.lstat()
            except OSError:
                continue
            if not stat_mod.S_ISREG(st.st_mode):
                continue
            bases.append(p)
    except OSError:
        return 0, 0

    # Timestamped names sort chronologically; newest first.
    bases.sort(key=lambda p: p.name, reverse=True)

    removed = 0
    reclaimed = 0
    now = time.time()
    for idx, base in enumerate(bases):
        try:
            mtime = base.lstat().st_mtime
        except OSError:
            continue
        expired = (now - mtime) > max_age_days * 86400
        if idx < keep_count or not expired:
            continue
        family = [base] + [
            base.with_name(base.name + suffix)
            for suffix in _DB_BACKUP_SIDECAR_SUFFIXES
        ]
        for victim in family:
            try:
                size = victim.lstat().st_size
            except OSError:
                continue
            if _unlink(victim):
                removed += 1
                reclaimed += size

    # Orphaned sidecars: a -wal/-shm whose base backup is already gone is
    # useless forensics — prune once expired.
    base_names = {p.name for p in bases}
    try:
        for p in home.iterdir():
            name = p.name
            if not name.startswith(_DB_BACKUP_PREFIX):
                continue
            if not name.endswith(_DB_BACKUP_SIDECAR_SUFFIXES):
                continue
            parent_base = name
            for suffix in _DB_BACKUP_SIDECAR_SUFFIXES:
                if parent_base.endswith(suffix):
                    parent_base = parent_base[: -len(suffix)]
                    break
            if parent_base in base_names:
                continue
            try:
                st = p.lstat()
            except OSError:
                continue
            if not stat_mod.S_ISREG(st.st_mode):
                continue
            if (now - st.st_mtime) > max_age_days * 86400:
                if _unlink(p):
                    removed += 1
                    reclaimed += st.st_size
    except OSError:
        pass
    return removed, reclaimed


def _dir_size(path: Path) -> int:
    """Best-effort recursive directory size in bytes."""
    total = 0
    try:
        with os.scandir(path) as it:
            for entry in it:
                try:
                    if entry.is_file(follow_symlinks=False):
                        total += entry.stat(follow_symlinks=False).st_size
                    elif entry.is_dir(follow_symlinks=False):
                        total += _dir_size(Path(entry.path))
                except OSError:
                    continue
    except OSError:
        pass
    return total


# ---------------------------------------------------------------------------
# Disk status (low-space probe)
# ---------------------------------------------------------------------------

def disk_status(
    home: Optional[Path] = None,
    *,
    min_free_bytes: Optional[int] = None,
    min_free_percent: Optional[float] = None,
) -> Dict[str, Any]:
    """Return free-space status for the volume backing HERMES_HOME.

    ``low_space`` is True when free space is below EITHER the byte or the
    percent threshold — the signal health endpoints use to report degraded
    before the instance actually hits ENOSPC.
    """
    home = home or get_hermes_home()
    if min_free_bytes is None or min_free_percent is None:
        low_cfg = get_disk_config()["low_space"]
        if min_free_bytes is None:
            min_free_bytes = int(low_cfg["min_free_bytes"])
        if min_free_percent is None:
            min_free_percent = float(low_cfg["min_free_percent"])

    usage = shutil.disk_usage(str(home))
    percent_free = (usage.free / usage.total * 100.0) if usage.total else 0.0
    low_space = usage.free < min_free_bytes or percent_free < min_free_percent
    return {
        "path": str(home),
        "total_bytes": usage.total,
        "free_bytes": usage.free,
        "percent_free": round(percent_free, 2),
        "min_free_bytes": min_free_bytes,
        "min_free_percent": min_free_percent,
        "low_space": low_space,
    }


def disk_usage_summary(home: Optional[Path] = None) -> Dict[str, int]:
    """Per-family byte usage under HERMES_HOME (diagnostics)."""
    home = home or get_hermes_home()
    summary: Dict[str, int] = {}

    def _add(name: str, path: Path) -> None:
        if path.is_dir():
            summary[name] = _dir_size(path)
        elif path.is_file():
            try:
                summary[name] = path.stat().st_size
            except OSError:
                pass

    _add("logs", home / "logs")
    _add("sessions", home / "sessions")
    _add("cache", home / "cache")
    _add("memories", home / "memories")
    _add("platforms", home / "platforms")
    _add("skills", home / "skills")
    _add("home", home / "home")
    _add("photon", home / "photon")
    _add("lazy_packages", home / "lazy-packages")
    _add("state_db", home / "state.db")
    _add("state_db_wal", home / "state.db-wal")

    db_backups = sum(
        f.stat().st_size
        for f in home.glob("state.db.malformed-backup-*")
        if f.is_file()
    )
    if db_backups:
        summary["state_db_backups"] = db_backups
    return summary


# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------

def run_retention_sweep(
    home: Optional[Path] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run one retention pass over all known-safe families under HERMES_HOME.

    Exception-proof: each family is wrapped independently; a failure in one
    family never prevents the others from running, and no exception escapes
    this function.  Returns a report dict::

        {"enabled": True, "bytes_reclaimed": int, "files_removed": int,
         "duration_ms": int, "families": {name: {...}}, "errors": [str]}
    """
    started = time.monotonic()
    report: Dict[str, Any] = {
        "enabled": True,
        "bytes_reclaimed": 0,
        "files_removed": 0,
        "families": {},
        "errors": [],
    }
    try:
        home = home or get_hermes_home()
        cfg = (config or get_disk_config())["retention"]
    except Exception as exc:  # noqa: BLE001
        report["errors"].append(f"config: {exc}")
        report["duration_ms"] = int((time.monotonic() - started) * 1000)
        return report

    if not cfg.get("enabled", True):
        report["enabled"] = False
        report["duration_ms"] = int((time.monotonic() - started) * 1000)
        return report

    max_bytes = int(cfg.get("diag_log_max_bytes", 2 * 1024 * 1024))
    keep_bytes = int(cfg.get("diag_log_keep_bytes", 256 * 1024))
    cache_age_h = float(cfg.get("cache_max_age_hours", 72))
    db_keep = int(cfg.get("db_backup_keep_count", 5))
    db_age_d = float(cfg.get("db_backup_max_age_days", 14))
    pkg_age_d = float(cfg.get("pkg_cache_max_age_days", 14))

    def _family(name, fn):
        try:
            removed, reclaimed = fn()
            report["families"][name] = {
                "files_removed": removed,
                "bytes_reclaimed": reclaimed,
            }
            report["files_removed"] += removed
            report["bytes_reclaimed"] += reclaimed
        except Exception as exc:  # noqa: BLE001 — sweep must never raise
            report["errors"].append(f"{name}: {exc}")

    # 1. Un-rotated diagnostic logs under logs/ — includes files written by
    #    external processes (container boot / restart / exit-diag logs on
    #    hosted instances) and nohup-redirected output.  Skips the
    #    RotatingFileHandler-managed trio and their numbered backups.
    def _diag_logs():
        reclaimed = 0
        truncated = 0
        log_dir = home / "logs"
        if log_dir.is_dir():
            for f in log_dir.iterdir():
                if not f.is_file():
                    continue
                base = f.name.split(".log")[0] + ".log" if ".log" in f.name else f.name
                if base in _ROTATION_MANAGED_LOGS:
                    continue
                if f.suffix not in (".log", ".txt", ".out", ".err"):
                    continue
                got = truncate_log_tail(
                    f, max_bytes=max_bytes, keep_bytes=keep_bytes, home=home
                )
                if got:
                    truncated += 1
                    reclaimed += got
        return truncated, reclaimed

    _family("diag_logs", _diag_logs)

    # 2. Root-level debug logs (interrupt_debug.log and friends).
    def _root_debug_logs():
        reclaimed = 0
        truncated = 0
        for f in list(home.glob("*_debug.log")) + list(home.glob("*-debug.log")):
            got = truncate_log_tail(
                f, max_bytes=max_bytes, keep_bytes=keep_bytes, home=home
            )
            if got:
                truncated += 1
                reclaimed += got
        return truncated, reclaimed

    _family("root_debug_logs", _root_debug_logs)

    # 3. WhatsApp bridge log (unbounded append from the Node bridge).
    def _bridge_logs():
        reclaimed = 0
        truncated = 0
        candidates = list((home / "platforms").glob("**/bridge.log")) if (home / "platforms").is_dir() else []
        legacy = home / "whatsapp" / "bridge.log"
        if legacy.is_file():
            candidates.append(legacy)
        for f in candidates:
            got = truncate_log_tail(
                f, max_bytes=max_bytes, keep_bytes=keep_bytes, home=home
            )
            if got:
                truncated += 1
                reclaimed += got
        return truncated, reclaimed

    _family("bridge_logs", _bridge_logs)

    # 4. Skills hub audit log (unbounded append).
    def _audit_log():
        f = home / "skills" / ".hub" / "audit.log"
        if not f.is_file():
            return 0, 0
        got = truncate_log_tail(
            f, max_bytes=max_bytes, keep_bytes=keep_bytes, home=home
        )
        return (1 if got else 0), got

    _family("skills_audit_log", _audit_log)

    # 5. state.db malformed-recovery backups: keep the newest N SETS
    #    (base + WAL/SHM sidecars pruned together — see
    #    prune_db_backup_family for the writer contract).
    def _db_backups():
        return prune_db_backup_family(
            home, keep_count=db_keep, max_age_days=db_age_d
        )

    _family("db_backups", _db_backups)

    # 6. Media cache backstop (cache/audio has no other cleanup; the others
    #    get a 24h hourly pass in the gateway — this is the safety net).
    def _media_caches():
        removed = 0
        reclaimed = 0
        for sub in ("images", "audio", "documents", "screenshots"):
            d = home / "cache" / sub
            if not d.is_dir():
                continue
            r, b = prune_files(
                d.iterdir(), max_age_days=cache_age_h / 24.0, home=home
            )
            removed += r
            reclaimed += b
        return removed, reclaimed

    _family("media_caches", _media_caches)

    # 7. Package-manager caches under the per-profile HOME (npm _cacache,
    #    pip cache) — content-addressed, safe to regenerate.
    def _pkg_caches():
        removed = 0
        reclaimed = 0
        for d in (
            home / "home" / ".npm" / "_cacache",
            home / "home" / ".cache" / "pip",
        ):
            if not d.is_dir():
                continue
            r, b = prune_files(
                d.glob("**/*"), max_age_days=pkg_age_d, home=home
            )
            removed += r
            reclaimed += b
        return removed, reclaimed

    _family("pkg_caches", _pkg_caches)

    report["duration_ms"] = int((time.monotonic() - started) * 1000)
    return report


def sweep_and_log(log: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """Run the sweep and emit exactly one structured log line.

    Never raises — safe to call from any host-process loop.
    """
    log = log or logger
    try:
        report = run_retention_sweep()
    except Exception as exc:  # noqa: BLE001 — absolute last-resort guard
        try:
            log.warning("Disk retention sweep crashed: %s", exc)
        except Exception:  # noqa: BLE001
            pass
        return {"enabled": False, "errors": [str(exc)]}

    try:
        status = disk_status()
        level = logging.WARNING if status["low_space"] else logging.INFO
        log.log(
            level,
            "Disk retention sweep: reclaimed %d bytes across %d file(s) "
            "in %dms (free: %d bytes / %.1f%%, low_space=%s%s)",
            report.get("bytes_reclaimed", 0),
            report.get("files_removed", 0),
            report.get("duration_ms", 0),
            status["free_bytes"],
            status["percent_free"],
            status["low_space"],
            f", errors={report['errors']}" if report.get("errors") else "",
        )
        report["disk"] = status
    except Exception as exc:  # noqa: BLE001
        try:
            log.debug("Disk retention status log failed: %s", exc)
        except Exception:  # noqa: BLE001
            pass
    return report
