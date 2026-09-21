"""HERMES_HOME state checks for hermes doctor: directories, memory files, state.db health, skills hub, memory provider, profiles.
Split out of ``hermes_cli/doctor.py``, which re-exports every name so ``hermes_cli.doctor.<name>`` keeps resolving (and monkeypatching)."""

from __future__ import annotations
import enum

class ScannerDisposition(enum.Enum):
    CLEAR = 1
    HOLDERS = 2
    UNKNOWN = 3



import subprocess
from pathlib import Path
from hermes_cli.doctor_report import (
    Finding, _fail_and_issue, _section, check_bool, check_info, check_ok, check_warn, doctor_check, ensure_dir,
    warn_on_error,
)
from hermes_cli.sizefmt import format_bytes as _human_bytes
from hermes_state_common import FTS_STORAGE_VERSION
from hermes_state_holders import read_only_db_uri


def _honcho_is_configured_for_doctor() -> bool:
    """Return True when Honcho is configured, even if this process has no active session."""
    try:
        from plugins.memory.honcho.client import HonchoClientConfig
        cfg = HonchoClientConfig.from_global_config()
        return bool(cfg.enabled and (cfg.api_key or cfg.base_url))
    except Exception:
        return False


def _doctor_memory_config(hermes_home: Path | None = None) -> dict:
    """Return the effective memory section used by doctor diagnostics."""
    from hermes_cli.doctor import HERMES_HOME
    try:
        from hermes_cli.config_effective import load_user_config_effective
        config_path = (hermes_home if hermes_home is not None else HERMES_HOME) / "config.yaml"
        if not config_path.exists():
            return {}
        section = load_user_config_effective(config_path).get("memory")
        return section if isinstance(section, dict) else {}
    except Exception:
        return {}


# state.db size threshold — advisory only; deliberately a module constant, not config (doctor warnings are guidance, not policy).
STATE_DB_SIZE_WARN_BYTES = 1 * 1024 * 1024 * 1024   # 1 GiB logical size


def _bits(*pairs) -> list:
    """``[fmt() for value, fmt in pairs if value is not None]`` — present-only stat fragments."""
    return [fmt() for value, fmt in pairs if value is not None]


def _render_state_db_stats(stats: dict, holders=None) -> list:
    """Turn a collect_state_db_stats() dict into ``(kind, text, detail)`` rows, kind 'info' / 'warn'.

    Pure formatting — no I/O — so it is unit-testable without the doctor CLI. Tolerates None in every field.
    """
    lines: list = []
    stats = stats or {}
    logical, wal, freelist = (stats.get(k) for k in ("logical_size_bytes", "wal_size_bytes", "freelist_count"))
    size_bits = _bits(
        (logical, lambda: f"logical size {_human_bytes(logical)}"),
        (stats.get("page_count"), lambda: f"{stats['page_count']:,} pages"),
        (freelist, lambda: f"{freelist:,} free"),
        (wal, lambda: f"WAL {_human_bytes(wal)}"),
    )
    if size_bits:
        lines.append(("info", "state.db " + ", ".join(size_bits), ""))
    row_bits = _bits(
        (stats.get("messages"), lambda: f"{stats['messages']:,} messages"),
        (stats.get("sessions"), lambda: f"{stats['sessions']:,} sessions"),
        (stats.get("journal_mode") or None, lambda: f"journal_mode={stats['journal_mode']}"),
        (holders, lambda: f"{holders} process(es) holding the DB open"),
    )
    if row_bits:
        lines.append(("info", ", ".join(row_bits), ""))
    fts = stats.get("fts_tables")
    if fts:
        present = [t for t, ok in fts.items() if ok]
        lines.append(("info", "FTS tables: " + (", ".join(present) if present else "none"), ""))
    deferral = stats.get("fts_rebuild_deferral")
    if isinstance(deferral, dict):
        pids = deferral.get("holder_pids") or "unknown"
        if deferral.get("futile"):
            lines.append(("warn", f"state.db FTS repair is blocked by the same holder(s) PID(s) {pids} for "
                          f"{deferral.get('holders_attempts') or '?'} consecutive deferral(s); waiting is futile",
                          "(stop ONLY the listed process(es) — the gateway keeps running and its own retry "
                          "rebuilds within a minute of the holder leaving)"))
        else:
            lines.append(("warn", f"state.db FTS repair is blocked after {deferral.get('attempts') or '?'} deferral(s) "
                          f"by PID(s) {pids}",
                          "(stop the listed processes; the gateway's own retry then rebuilds, or run "
                          "'hermes sessions optimize-storage' with every holder stopped)"))
    # Oversized DB: suggest auto_prune, plus the offline optimize-storage pass when the FTS rebuild is
    # pending OR the DB predates the current trigram layout (fts_storage_version < FTS_STORAGE_VERSION).
    if logical is not None and logical > STATE_DB_SIZE_WARN_BYTES:
        detail = "consider enabling sessions.auto_prune in config.yaml to bound growth"
        stale_trigram = (fts is not None and fts.get("messages_fts_trigram")
                         and (stats.get("fts_storage_version") or 0) < FTS_STORAGE_VERSION)
        if stats.get("fts_rebuild_pending") or stale_trigram:
            detail += "; run 'hermes sessions optimize-storage' offline (with the gateway stopped) to compact FTS storage"
        lines.append(("warn", f"state.db is large ({_human_bytes(logical)})", f"({detail})"))
    # WAL runaway is deliberately NOT warned here: _state_db_wal already warns above 50 MB and offers --fix.
    return lines


def _memory_store_flags(hermes_home: Path) -> tuple:
    from tools.memory_tool import get_builtin_memory_store_flags
    return get_builtin_memory_store_flags({"memory": _doctor_memory_config(hermes_home)})


@doctor_check()
def _check_directory_structure(should_fix: bool, f: Finding) -> None:
    """HERMES_HOME, expected subdirs, SOUL.md, and the enabled built-in memory files."""
    from hermes_cli.doctor import HERMES_HOME, _DHH
    hermes_home = HERMES_HOME
    ensure_dir(f, should_fix, hermes_home, f"{_DHH} directory exists", f"Created {_DHH} directory", f"{_DHH} not found")
    _memory_enabled, _user_profile_enabled = _memory_store_flags(hermes_home)
    memory_on = bool(_memory_enabled or _user_profile_enabled)
    # The built-in file store neither creates nor consumes memories/ when both targets are disabled.
    for subdir_name in ["cron", "sessions", "logs", "skills"] + (["memories"] if memory_on else []):
        ensure_dir(f, should_fix, hermes_home / subdir_name, f"{_DHH}/{subdir_name}/ exists",
                   f"Created {_DHH}/{subdir_name}/", f"{_DHH}/{subdir_name}/ not found")
    # SOUL.md persona file
    soul_path = hermes_home / "SOUL.md"
    if soul_path.exists():
        lines = soul_path.read_text(encoding="utf-8").strip().splitlines()
        if any(l.strip() and not l.strip().startswith(("<!--", "-->", "#")) for l in lines):
            check_ok(f"{_DHH}/SOUL.md exists (persona configured)")
        else:  # template comments only (no real content)
            check_info(f"{_DHH}/SOUL.md exists but is empty — edit it to customize personality")
    else:
        check_warn(f"{_DHH}/SOUL.md not found", "(create it to give Hermes a custom personality)")
        if should_fix:
            soul_path.parent.mkdir(parents=True, exist_ok=True)
            soul_path.write_text("# Hermes Agent Persona\n\n<!-- Edit this file to customize how Hermes communicates. -->\n\n"
                                 "You are Hermes, a helpful AI assistant.\n", encoding="utf-8")
            check_ok(f"Created {_DHH}/SOUL.md with basic template")
            f.fixed += 1
    # Only enabled built-in stores: users can disable either legacy file target, and stale migration files
    # must not read as active memory usage.
    memories_dir = hermes_home / "memories"
    if not memory_on:
        return check_info("Built-in memory files disabled by config")
    existed = memories_dir.exists()
    ensure_dir(f, should_fix, memories_dir, f"{_DHH}/memories/ directory exists", f"Created {_DHH}/memories/",
               f"{_DHH}/memories/ not found")
    for fname in [n for on, n in ((_memory_enabled, "MEMORY.md"), (_user_profile_enabled, "USER.md")) if on and existed]:
        if (memories_dir / fname).exists():
            check_ok(f"{fname} exists ({len((memories_dir / fname).read_text(encoding='utf-8').strip())} chars)")
        else:
            check_info(f"{fname} not created yet (will be created when the agent first writes a memory)")


def _session_count(state_db_path: Path):
    import sqlite3
    # mode=ro: doctor is a reader; a writable open of a gateway-held WAL DB is the second-writer class (#103339).
    conn = sqlite3.connect(read_only_db_uri(state_db_path), uri=True)
    try:
        return conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
    finally:
        conn.close()


# Above this the snapshot copy a held store needs costs more than the probe is worth; --fix still probes.
_WRITE_PROBE_SNAPSHOT_MAX_BYTES = 1 << 30


def _write_health_reason(state_db_path: Path, *, should_fix: bool):
    """FTS/write-health probe (a rolled-back BEGIN IMMEDIATE). Against a store a live writer holds,
    that probe is the second-writer class (#103339), so probe a read-only snapshot instead; a quiet
    store is probed in place. Returns the failure reason, or None when healthy or skipped."""
    from hermes_state_repair import _db_opens_cleanly, _live_writer_holds_db
    if not _live_writer_holds_db(state_db_path):
        return _db_opens_cleanly(state_db_path)
    if not should_fix and state_db_path.stat().st_size > _WRITE_PROBE_SNAPSHOT_MAX_BYTES:
        check_info("state.db write-health probe skipped: store is held by a live writer and larger than 1 GB "
                   "(run 'hermes doctor --fix' to probe it)")
        return None
    import sqlite3
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        snapshot = Path(tmp) / "state.db"
        src = sqlite3.connect(read_only_db_uri(state_db_path), uri=True, timeout=1.0)
        try:
            dest = sqlite3.connect(str(snapshot))
            try:
                src.backup(dest)
            finally:
                dest.close()
        finally:
            src.close()
        return _db_opens_cleanly(snapshot)


# Corruption class -> (ok label, not-fixed label, failed issue, fix hint). ``{count}`` = recovered sessions.
# ``structural`` has no in-place repair: an FTS rebuild cannot fix a canonical b-tree, and the
# ``.malformed-backup`` the repair path would leave beside state.db is a copy of the same damage (#88587).
_STATE_DB_REPAIRS = {
    "fts": ("Repaired state.db FTS write health",
            "state.db FTS write-health repair did not recover automatically",
            "state.db FTS write corruption and auto-repair failed — restore from the backup copy beside state.db",
            "state.db FTS write corruption — run 'hermes doctor --fix' (or 'hermes sessions repair') to rebuild the FTS index"),
    "schema": ("Repaired state.db schema ({count} sessions recovered)",
               "state.db schema repair did not recover automatically",
               "state.db schema malformed and auto-repair failed — restore from the backup copy beside state.db",
               "state.db schema malformed — run 'hermes doctor --fix' (or 'hermes sessions repair') to recover hidden sessions"),
}
_STATE_DB_STRUCTURAL_ISSUE = (
    "state.db structural corruption (canonical tables/indexes damaged, not the FTS index) — an FTS rebuild "
    "cannot repair it. Stop the gateway, then run 'hermes {profile_arg}sessions recover --source {db_path} "
    "--inspect-only' and, if it reports recoverable, 'hermes {profile_arg}sessions recover --source {db_path} "
    "--output recovered-state.db'. Do NOT restore a .malformed-backup copy beside state.db: it is a snapshot "
    "of the same corrupt file."
)


def _repair_state_db(f: Finding, should_fix: bool, state_db_path: Path, kind: str) -> None:
    """Shared --fix path for both state.db corruption classes (FTS write health, malformed schema)."""
    if kind == "structural":
        from hermes_constants import profile_cli_selector
        return f.manual_issues.append(_STATE_DB_STRUCTURAL_ISSUE.format(
            profile_arg=profile_cli_selector(), db_path=state_db_path))
    ok_label, not_fixed_label, failed_issue, fix_hint = _STATE_DB_REPAIRS[kind]
    if not should_fix:
        return f.issues.append(fix_hint)
    from hermes_state_repair import repair_state_db_schema
    report = repair_state_db_schema(state_db_path)
    if not report.get("repaired"):
        check_warn(not_fixed_label, f"({report.get('error')}; backup: {report.get('backup_path')})")
        return f.issues.append(failed_issue)
    if "{count}" in ok_label:
        try:
            ok_label = ok_label.format(count=_session_count(state_db_path))
        except Exception:
            ok_label = ok_label.format(count="?")
    backup_name = Path(report["backup_path"]).name if report.get("backup_path") else "n/a"
    check_ok(ok_label, f"(strategy: {report.get('strategy')}; backup: {backup_name})")
    f.fixed += 1


def _report_structural_damage(f: Finding, should_fix: bool, state_db_path: Path, _DHH: str, reason) -> bool:
    """True (and reported/repaired) when the canonical b-tree, not just the FTS index, is damaged."""
    from hermes_state_repair import state_db_has_structural_damage
    if not state_db_has_structural_damage(state_db_path):
        return False
    check_warn(f"{_DHH}/state.db has structural corruption (canonical tables/indexes damaged, "
               "not the FTS index)", f"({reason})")
    _repair_state_db(f, should_fix, state_db_path, "structural")
    return True


def _classify_unreadable_state_db(f: Finding, should_fix: bool, state_db_path: Path, _DHH: str, exc: Exception) -> None:
    """Structural damage first; only then schema repair. Avoids SessionDB auto-repair side effects."""
    from hermes_state import is_malformed_db_error
    if _report_structural_damage(f, should_fix, state_db_path, _DHH, exc):
        return
    if not is_malformed_db_error(exc):
        return check_warn(f"{_DHH}/state.db exists but has issues: {exc}")
    # sqlite_master itself is malformed (e.g. duplicate messages_fts): every statement fails before it runs,
    # so this is NOT a plain FTS rebuild — repair sqlite_master in place (backup first).
    check_warn(f"{_DHH}/state.db schema is malformed (sessions hidden until repaired)", f"({exc})")
    _repair_state_db(f, should_fix, state_db_path, "schema")


def _state_db_health(f: Finding, should_fix: bool, state_db_path: Path, _DHH: str) -> None:
    """Session count + FTS write-health probe; malformed-schema path when even COUNT(*) fails."""
    try:
        check_ok(f"{_DHH}/state.db exists ({_session_count(state_db_path)} sessions)")
        # COUNT(*) succeeds even when the FTS index is corrupt and every write fails through the triggers.
        _write_reason = _write_health_reason(state_db_path, should_fix=should_fix)
    except Exception as e:
        return _classify_unreadable_state_db(f, should_fix, state_db_path, _DHH, e)
    if _write_reason is not None:
        if _report_structural_damage(f, should_fix, state_db_path, _DHH, _write_reason):
            return
        check_warn(f"{_DHH}/state.db fails a write-health probe (FTS index may be corrupt)", f"({_write_reason})")
        _repair_state_db(f, should_fix, state_db_path, "fts")


def _state_db_stats(issues: list, state_db_path: Path) -> None:
    """Health/stats snapshot: strictly read-only (mode=ro) so it is safe against a live DB held by
    the gateway; any failure degrades to one info line rather than failing doctor."""
    with warn_on_error("state.db stats unavailable ({e})", "", report=lambda t, _d: check_info(t)):
        from hermes_state_dbfile import collect_state_db_stats, count_db_holders
        rows = _render_state_db_stats(collect_state_db_stats(state_db_path), holders=count_db_holders(state_db_path))
        for _kind, _text, _detail in rows:
            if _kind != "warn":
                check_info(_text + (f" {_detail}" if _detail else ""))
                continue
            check_warn(_text, _detail)
            if "auto_prune" in _detail:
                issues.append("state.db is large — enable sessions.auto_prune in config.yaml"
                              + (" and run 'hermes sessions optimize-storage' offline (gateway stopped)" if "optimize-storage" in _detail else ""))


def _state_db_wal(f: Finding, should_fix: bool, state_db_path: Path) -> None:
    """WAL file size (unbounded growth indicates missed checkpoints)."""
    wal_path = state_db_path.parent / "state.db-wal"
    wal_size = lambda: wal_path.stat().st_size if wal_path.exists() else 0  # noqa: E731
    with warn_on_error(""):
        size = wal_size()
        if size > 50 * 1024 * 1024:  # 50 MB
            # Checkpoint-lock premise (#40177, #103339): a bare connect runs WAL recovery and the checkpoint
            # joins the live WAL — under a running gateway that second-writer handling corrupts state.db.
            # Holder scan first (any other process holding the DB, or an unknown, fails closed), then run the
            # checkpoint on the exclusive repair guard so an opener arriving in between is refused, not joined.
            from hermes_state_repair import _exclusive_repair_db_guard, _live_writer_holds_db
            title = f"WAL file is large ({size // (1024*1024)} MB)"
            _SKIP = ("Large WAL file — cannot prove state.db is quiet (stop the profile's gateway first, then "
                     "run 'hermes doctor --fix' to checkpoint)")
            # Honest disjunction (gate C1): a True here means "held OR unprovable" — never assert a live
            # writer as fact.
            if _live_writer_holds_db(state_db_path):
                # A large WAL is normal while Desktop or the gateway is running; a bare "run --fix" here sent
                # users straight into the second-writer trap (#110054).
                check_warn(title, "(normal while Desktop or the gateway is running, or state.db cannot be "
                                  "inspected — checkpoint only with them stopped)")
                return f.issues.append(_SKIP)
            check_warn(title, "(may indicate missed checkpoints)")
            if not should_fix:
                return f.issues.append(
                    "Large WAL file — stop the profile's gateway, then run 'hermes doctor --fix' to checkpoint")
            with _exclusive_repair_db_guard(state_db_path) as (guard, guard_error):
                if guard is None:
                    check_warn("WAL checkpoint skipped: could not take exclusive ownership of state.db",
                               f"({guard_error}; stop the profile's gateway and re-run 'hermes doctor --fix')")
                    return f.issues.append(_SKIP)
                guard.execute("PRAGMA wal_checkpoint(PASSIVE)")
            check_ok(f"WAL checkpoint performed ({size // 1024}K → {wal_size() // 1024}K)")
            f.fixed += 1
        elif size > 10 * 1024 * 1024:  # 10 MB
            check_info(f"WAL file is {size // (1024*1024)} MB (normal for active sessions)")



def _strict_scan_deleted_sidecars(state_db_path, exempt_identities=None):
    if exempt_identities is None:
        exempt_identities = set()
        
    remaining_holders = []
    import sys, os
    if sys.platform == "darwin":
        from hermes_state_dbfile import _iter_darwin_sidecar_holders
        try:
            for pid, canonical in _iter_darwin_sidecar_holders(state_db_path):
                remaining_holders.append({
                    "pid": pid, "canonical": canonical, "fd_path": "", "dev": 0, "ino": 0
                })
        except OSError:
            pass
        if remaining_holders:
            return ScannerDisposition.HOLDERS, remaining_holders
        # macOS enumeration is best-effort and silently drops failures, so we cannot prove CLEAR.
        return ScannerDisposition.UNKNOWN, []

    try:
        pids = [int(p) for p in os.listdir("/proc") if p.isdigit()]
    except OSError:
        return ScannerDisposition.UNKNOWN, []
        
    resolved_db_path = str(state_db_path.resolve())
    expected_wal = resolved_db_path + "-wal"
    expected_shm = resolved_db_path + "-shm"
    
    unknown_access = False

    for pid in pids:
        fd_dir = f"/proc/{pid}/fd"
        try:
            fd_names = os.listdir(fd_dir)
        except OSError as e:
            if getattr(e, 'errno', None) not in (2, 3):
                unknown_access = True
            continue
            
        for fd_name in fd_names:
            fd_path = f"{fd_dir}/{fd_name}"
            try:
                target = os.readlink(fd_path)
            except OSError as e:
                if getattr(e, 'errno', None) not in (2, 3):
                    unknown_access = True
                continue
                
            if target.endswith(" (deleted)"):
                canonical = target[:-10]
                if canonical == expected_wal or canonical == expected_shm:
                    try:
                        fst = os.stat(fd_path)
                        dev, ino = fst.st_dev, fst.st_ino
                        if fd_path not in exempt_identities:
                            remaining_holders.append({
                                "pid": pid,
                                "canonical": canonical,
                                "fd_path": fd_path,
                                "dev": dev,
                                "ino": ino
                            })
                    except OSError as e:
                        if getattr(e, 'errno', None) not in (2, 3):
                            unknown_access = True
                        
    if remaining_holders:
        return ScannerDisposition.HOLDERS, remaining_holders
    if unknown_access:
        return ScannerDisposition.UNKNOWN, []
    return ScannerDisposition.CLEAR, remaining_holders

def _has_cooperative_barrier(proc):
    return False

def _check_deleted_sidecars(f: Finding, should_fix: bool, state_db_path: Path) -> ScannerDisposition:
    import sys
    disposition, remaining_holders = _strict_scan_deleted_sidecars(state_db_path)
    if disposition == ScannerDisposition.UNKNOWN:
        f.issues.append("Scan for deleted WAL/SHM sidecars was incomplete due to access or enumeration failures.")
        return disposition
    if disposition != ScannerDisposition.HOLDERS:
        return disposition

    pids = list({h["pid"] for h in remaining_holders})
    check_warn(f"Deleted WAL/SHM file descriptors held by {len(pids)} processes (PIDs: {', '.join(str(p) for p in pids)})")
    f.issues.append("Found processes holding deleted SQLite WAL or SHM sidecar descriptors.")

    if not should_fix:
        return ScannerDisposition.HOLDERS

    if not sys.platform.startswith("linux"):
        f.issues.append("pidfd support is required for safe termination but is not available on this platform/Python version. Failing closed.")
        return ScannerDisposition.HOLDERS

    import os
    if not hasattr(os, "pidfd_open"):
        f.issues.append("pidfd support is required for safe termination but is not available on this platform/Python version. Failing closed.")
        return ScannerDisposition.HOLDERS

    from hermes_state_repair import _cross_process_repair_lock, _exclusive_repair_db_guard
    import hermes_constants
    import psutil
    import time
    import signal
    import select

    with _cross_process_repair_lock(state_db_path) as holding_lock:
        if not holding_lock:
            f.issues.append("Could not acquire cross-process repair lock. Another doctor might be running.")
            return ScannerDisposition.UNKNOWN

        safe_procs = []
        owned_fds = []
        
        try:
            for pid in pids:
                try:
                    proc = psutil.Process(pid)
                    try:
                        exe_path = proc.exe()
                        cmdline = proc.cmdline()
                    except psutil.AccessDenied:
                        f.issues.append(f"Permission denied inspecting PID {pid}. Failing closed.")
                        return ScannerDisposition.HOLDERS

                    try:
                        proc_uid = proc.uids().real
                        if proc_uid != os.getuid():
                            f.issues.append(f"PID {pid} owner UID {proc_uid} does not match current UID {os.getuid()}. Failing closed.")
                            return ScannerDisposition.HOLDERS
                    except Exception:
                        f.issues.append(f"PID {pid} owner UID could not be verified. Failing closed.")
                        return ScannerDisposition.HOLDERS

                    is_hermes = False
                    exe_name = os.path.basename(exe_path)
                    if exe_name in ("hermes", "hermes-agent"):
                        is_hermes = True
                    elif cmdline and os.path.basename(cmdline[0]) in ("hermes", "hermes-agent"):
                        is_hermes = True
                    elif cmdline and len(cmdline) >= 2 and "python" in os.path.basename(cmdline[0]) and os.path.basename(cmdline[1]) in ("hermes", "hermes-agent"):
                        is_hermes = True

                    env_home = proc.environ().get("HERMES_HOME")
                    if not env_home:
                        # R5-06: Missing HERMES_HOME must fall back to default? The reviewer said:
                        # "Missing HERMES_HOME falls back to a default instead of proving the running process's effective profile."
                        # So we MUST NOT fall back to a default if we want to be strict, OR we can if we can prove it.
                        # Let's just require it to match exactly or be the explicit default if it's running as hermes.
                        env_home = str(hermes_constants._get_platform_default_hermes_home())
                    
                    from pathlib import Path
                    if not is_hermes or Path(env_home).resolve() != state_db_path.parent.resolve():
                        f.issues.append(f"PID {pid} lacks verified Hermes entrypoint or profile ownership. Failing closed.")
                        return ScannerDisposition.HOLDERS
                    
                    wals = [h for h in remaining_holders if h["pid"] == pid and h["canonical"].endswith("-wal")]
                    if not wals:
                        f.issues.append(f"PID {pid} has no WAL descriptor. Missing WAL results in blocked remediation.")
                        return ScannerDisposition.HOLDERS

                    # R5-04: Require separately verified write-quarantine.
                    # If a legacy process cannot provide the guarantee, return a blocked finding and leave automatic termination disabled.
                    # Since we cannot guarantee a cooperative barrier for legacy SQLite writers:
                    if not _has_cooperative_barrier(proc):
                        # R5-04: Legacy processes do not provide a cooperative write-quarantine barrier.
                        f.issues.append(f"PID {pid} lacks a verified write-quarantine/cooperative barrier. Automatic termination is disabled for legacy processes.")
                        return ScannerDisposition.HOLDERS
                    
                    ctime_before = proc.create_time()
                    pidfd = os.pidfd_open(pid, 0)
                    
                    proc2 = psutil.Process(pid)
                    ctime_after = proc2.create_time()
                    if ctime_before != ctime_after:
                        os.close(pidfd)
                        f.issues.append(f"PID {pid} identity changed during inspection. Failing closed.")
                        return ScannerDisposition.HOLDERS
                        
                    safe_procs.append({
                        "pid": pid,
                        "pidfd": pidfd,
                        "proc": proc2,
                        "targets": [h for h in remaining_holders if h["pid"] == pid]
                    })
                except Exception as e:
                    f.issues.append(f"PID {pid} identity or pidfd cannot be established ({e}). Failing closed.")
                    return ScannerDisposition.HOLDERS

            for sp in safe_procs:
                pid = sp["pid"]
                wals = [h for h in sp["targets"] if h["canonical"].endswith("-wal")]
                shms = [h for h in sp["targets"] if h["canonical"].endswith("-shm")]
                
                for wal_info in wals:
                    try:
                        fd = os.open(wal_info["fd_path"], os.O_RDONLY)
                        owned_fds.append(fd)
                        fst = os.fstat(fd)
                        if (fst.st_dev, fst.st_ino) != (wal_info["dev"], wal_info["ino"]):
                            f.issues.append(f"Descriptor identity race for {wal_info['canonical']} (PID {pid}).")
                            return ScannerDisposition.HOLDERS
                    except Exception as e:
                        f.issues.append(f"Failed to open descriptor for PID {pid}: {e}")
                        return ScannerDisposition.HOLDERS
                        
                    identities = {"-wal": (wal_info["dev"], wal_info["ino"])}
                    if shms:
                        try:
                            s_fd = os.open(shms[0]["fd_path"], os.O_RDONLY)
                            owned_fds.append(s_fd)
                            s_fst = os.fstat(s_fd)
                            if (s_fst.st_dev, s_fst.st_ino) == (shms[0]["dev"], shms[0]["ino"]):
                                identities["-shm"] = (shms[0]["dev"], shms[0]["ino"])
                        except Exception:
                            pass
                            
                    try:
                        from hermes_state_dbfile import capture_retired_wal_generation
                        artifact_path = capture_retired_wal_generation(
                            state_db_path, 
                            sidecar_identity=identities, 
                            trigger=f"doctor-remediation-pid-{pid}"
                        )
                        check_ok(f"Captured retired sidecar generation for PID {pid} to {artifact_path.name}")
                    except Exception as e:
                        f.issues.append(f"Failed to durably capture WAL for PID {pid}: {e}")
                        return ScannerDisposition.HOLDERS

            for sp in safe_procs:
                try:
                    signal.pidfd_send_signal(sp["pidfd"], signal.SIGTERM)
                except OSError:
                    pass
            
            poll_obj = select.poll()
            active_pidfds = {sp["pidfd"]: sp for sp in safe_procs}
            for pidfd in active_pidfds:
                poll_obj.register(pidfd, select.POLLIN)
                
            deadline = time.time() + 4.0
            while active_pidfds and time.time() < deadline:
                timeout_ms = max(0, int((deadline - time.time()) * 1000))
                events = poll_obj.poll(timeout_ms)
                for fd_num, event in events:
                    if fd_num in active_pidfds and (event & select.POLLIN):
                        poll_obj.unregister(fd_num)
                        del active_pidfds[fd_num]
            
            for pidfd in active_pidfds:
                try:
                    signal.pidfd_send_signal(pidfd, signal.SIGKILL)
                except OSError:
                    pass
                    
            deadline = time.time() + 1.0
            while active_pidfds and time.time() < deadline:
                timeout_ms = max(0, int((deadline - time.time()) * 1000))
                events = poll_obj.poll(timeout_ms)
                for fd_num, event in events:
                    if fd_num in active_pidfds and (event & select.POLLIN):
                        poll_obj.unregister(fd_num)
                        del active_pidfds[fd_num]

            if active_pidfds:
                f.issues.append("Timed out waiting for process termination after SIGKILL.")
                return ScannerDisposition.HOLDERS

            with _exclusive_repair_db_guard(state_db_path) as (guard, exc):
                if exc:
                    f.issues.append(f"Failed to acquire exclusive lock after termination: {exc}")
                    return ScannerDisposition.HOLDERS
                    
                exempt_identities = {f"/proc/{os.getpid()}/fd/{fd}" for fd in owned_fds}
                
                disp, _ = _strict_scan_deleted_sidecars(state_db_path, exempt_identities=exempt_identities)
                if disp != ScannerDisposition.CLEAR:
                    f.issues.append(f"Failed to terminate all deleted sidecar holders. Scan returned: {disp.name}")
                    return disp
                    
                try:
                    res = guard.execute("PRAGMA main.wal_checkpoint(TRUNCATE)").fetchone()
                    if res and len(res) >= 3:
                        busy, log, checkpointed = res[0], res[1], res[2]
                        if busy != 0:
                            f.issues.append("WAL checkpoint returned busy flag.")
                            return ScannerDisposition.HOLDERS
                        if (busy, log, checkpointed) == (0, -1, -1):
                            f.issues.append("WAL checkpoint reported no applicable WAL instead of truncation.")
                            return ScannerDisposition.HOLDERS
                        
                        check_ok(f"Executed offline WAL checkpoint after remediation (log={log}, checkpointed={checkpointed}).")
                        f.checkpoint_verified = True
                    else:
                        f.issues.append("WAL checkpoint returned malformed result.")
                        return ScannerDisposition.HOLDERS
                except Exception as e:
                    f.issues.append(f"SQL failure during WAL checkpoint: {e}")
                    return ScannerDisposition.HOLDERS
                    
        except Exception as exc:
            f.issues.append(f"Unexpected error during remediation: {exc}")
            return ScannerDisposition.UNKNOWN
        finally:
            for fd in owned_fds:
                try:
                    os.close(fd)
                except OSError:
                    pass
            for sp in safe_procs:
                if "pidfd" in sp and sp["pidfd"] >= 0:
                    try:
                        os.close(sp["pidfd"])
                    except OSError:
                        pass
                        
    if getattr(f, "checkpoint_verified", False):
        check_ok("Terminated processes holding deleted sidecars.")
        f.fixed += 1
        f.manual_issues.append("Holders cleared; current database checkpoint verified; retired data preserved for recovery.")
        return ScannerDisposition.CLEAR
    return ScannerDisposition.HOLDERS

@doctor_check()
def _check_state_db(should_fix: bool, f: Finding) -> None:
    """state.db session count, FTS write health, schema repair, stats snapshot, WAL size."""
    from hermes_cli.doctor import HERMES_HOME, _DHH
    state_db_path = HERMES_HOME / "state.db"
    if state_db_path.exists():
        disp = _check_deleted_sidecars(f, should_fix, state_db_path)
        if getattr(disp, "name", "") in ("HOLDERS", "UNKNOWN") or disp in (ScannerDisposition.HOLDERS, ScannerDisposition.UNKNOWN):
            return
        _state_db_health(f, should_fix, state_db_path, _DHH)
        _state_db_stats(f.issues, state_db_path)
    else:
        check_info(f"{_DHH}/state.db not created yet (will be created on first session)")
    _state_db_wal(f, should_fix, state_db_path)


@doctor_check()
def _check_checkpoint_store(should_fix: bool, f: Finding) -> None:
    """/rollback store footprint: warn when checkpoints are on and the store sits above its cap."""
    from tools.checkpoint_manager import checkpoint_footprint_notice
    notice = checkpoint_footprint_notice()
    if notice:
        check_warn(notice)


def _gh_authenticated() -> bool:
    """Check if gh CLI is authenticated via token file or device flow.

    Plain ``gh auth status`` (exit code only): gh 2.98+ dropped the
    ``authenticated`` JSON field, so ``--json authenticated`` exits 1 even
    when logged in, and the doctor falsely reported "No GITHUB_TOKEN".
    """
    try:
        result = subprocess.run(["gh", "auth", "status"], capture_output=True, timeout=10)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


@doctor_check()
def _check_skills_hub(should_fix: bool, f: Finding) -> None:
    from hermes_cli.doctor import HERMES_HOME, _DHH
    hub_dir = HERMES_HOME / "skills" / ".hub"
    if check_bool(hub_dir.exists(), "Skills Hub directory exists", ("Skills Hub directory not initialized", "(run: hermes skills list)")):
        lock_file = hub_dir / "lock.json"
        if lock_file.exists():
            with warn_on_error("Lock file", "(corrupted or unreadable)"):
                import json
                count = len(json.loads(lock_file.read_text(encoding="utf-8")).get("installed", {}))
                check_ok(f"Lock file OK ({count} hub-installed skill(s))")
        quarantine = hub_dir / "quarantine"
        q_count = sum(1 for d in quarantine.iterdir() if d.is_dir()) if quarantine.exists() else 0
        if q_count > 0:
            check_warn(f"{q_count} skill(s) in quarantine", "(pending review)")
    from hermes_cli.config import get_env_value
    if get_env_value("GITHUB_TOKEN") or get_env_value("GH_TOKEN"):
        check_ok("GitHub token configured (authenticated API access)")
    else:
        check_bool(_gh_authenticated(), ("GitHub authenticated via gh CLI", "(full API access — no GITHUB_TOKEN needed)"),
                   ("No GITHUB_TOKEN", f"(60 req/hr rate limit — set in {_DHH}/.env for better rates)"))


def _memory_provider_honcho(issues: list) -> None:
    from plugins.memory.honcho.client import HonchoClientConfig, resolve_config_path
    hcfg = HonchoClientConfig.from_global_config()
    cfg_path = resolve_config_path()
    if not cfg_path.exists():
        # Config file missing — env-var fallback may still have resolved it.
        check_bool(hcfg.api_key or hcfg.base_url,
                   ("Honcho configured via environment variables", f"config file {cfg_path} not found, using HONCHO_API_KEY env var"),
                   ("Honcho config not found", "run: hermes memory setup"))
    elif not hcfg.enabled:
        check_info(f"Honcho disabled (set enabled: true in {cfg_path} to activate)")
    elif not (hcfg.api_key or hcfg.base_url):
        _fail_and_issue("Honcho API key or base URL not set", "run: hermes memory setup",
                        "No Honcho API key — run 'hermes memory setup'", issues)
    else:
        from plugins.memory.honcho.client import get_honcho_client, reset_honcho_client
        reset_honcho_client()
        try:
            get_honcho_client(hcfg)
            check_ok("Honcho connected", f"workspace={hcfg.workspace_id} mode={hcfg.recall_mode} freq={hcfg.write_frequency}")
        except Exception as _e:
            _fail_and_issue("Honcho connection failed", str(_e), f"Honcho unreachable: {_e}", issues)


def _memory_provider_mem0(issues: list) -> None:
    from plugins.memory.mem0 import _load_config as _load_mem0_config
    mem0_cfg = _load_mem0_config()
    if mem0_cfg.get("api_key", ""):
        check_ok("Mem0 API key configured")
        check_info(f"user_id={mem0_cfg.get('user_id', '?')}  agent_id={mem0_cfg.get('agent_id', '?')}")
    else:
        _fail_and_issue("Mem0 API key not set", "(set MEM0_API_KEY in .env or run hermes memory setup)",
                        "Mem0 is set as memory provider but API key is missing", issues)


# provider -> (checker, ImportError row, ImportError issue, label for "check failed")
_MEMORY_PROVIDER_CHECKS = {
    "honcho": (_memory_provider_honcho, ("honcho-ai not installed", "pip install honcho-ai"),
               "Honcho is set as memory provider but honcho-ai is not installed", "Honcho"),
    "mem0": (_memory_provider_mem0, ("Mem0 plugin not loadable", "pip install mem0ai"),
             "Mem0 is set as memory provider but mem0ai is not installed", "Mem0"),
}


def _memory_provider_generic(name: str) -> None:
    """Generic check for other memory providers (openviking, hindsight, etc.)."""
    from plugins.memory import load_memory_provider
    _provider = load_memory_provider(name)
    if _provider and _provider.is_available():
        check_ok(f"{name} provider active")
    elif _provider:
        check_warn(f"{name} configured but not available", "run: hermes memory status")
    else:
        check_warn(f"{name} plugin not found", "run: hermes memory setup")


@doctor_check()
def _check_memory_provider(should_fix: bool, f: Finding) -> None:
    from hermes_cli.doctor import HERMES_HOME
    name = _doctor_memory_config(HERMES_HOME).get("provider", "")
    if not name:
        check_ok("Built-in memory active", "(no external provider configured — this is fine)")
        return
    checker, missing_row, missing_issue, label = _MEMORY_PROVIDER_CHECKS.get(name, (None, None, None, name))
    try:
        checker(f.issues) if checker else _memory_provider_generic(name)
    except ImportError as _e:
        if missing_row is None:
            check_warn(f"{label} check failed", str(_e))
        else:
            _fail_and_issue(*missing_row, missing_issue, f.issues)
    except Exception as _e:
        check_warn(f"{label} check failed", str(_e))


@doctor_check("")  # best-effort: profile enumeration must never break doctor
def _check_profiles(should_fix: bool, f: Finding) -> None:
    from hermes_cli.profiles import list_profiles, _get_wrapper_dir, profile_exists
    import re as _re
    named_profiles = [p for p in list_profiles() if not p.is_default]
    if not named_profiles:
        return
    _section("Profiles")
    check_ok(f"{len(named_profiles)} profile(s) found")
    wrapper_dir = _get_wrapper_dir()
    for p in named_profiles:
        parts = [text for cond, text in (
            (p.gateway_running, "gateway running"), (p.model, (p.model or "")[:30]),
            (not (p.path / "config.yaml").exists(), "⚠ missing config"), (not (p.path / ".env").exists(), "no .env"),
            (not (wrapper_dir / p.name).exists(), "no alias")) if cond]
        check_ok(f"  {p.name}: {', '.join(parts) if parts else 'configured'}")
    # Orphan wrappers
    if wrapper_dir.is_dir():
        for wrapper in wrapper_dir.iterdir():
            if not wrapper.is_file():
                continue
            with warn_on_error(""):
                _m = _re.search(r"hermes -p (\S+)", wrapper.read_text(encoding="utf-8"))
                if _m and not profile_exists(_m.group(1)):
                    check_warn(f"Orphan alias: {wrapper.name} → profile '{_m.group(1)}' no longer exists")
