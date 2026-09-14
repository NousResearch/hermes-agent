"""HERMES_HOME state checks for hermes doctor: directories, memory files, state.db health, skills hub, memory provider, profiles.
Split out of ``hermes_cli/doctor.py``, which re-exports every name so ``hermes_cli.doctor.<name>`` keeps resolving (and monkeypatching)."""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from hermes_cli.doctor_report import (
    Finding, _fail_and_issue, _section, check_bool, check_info, check_ok, check_warn, doctor_check, ensure_dir,
    warn_on_error,
)
from hermes_cli.sizefmt import format_bytes as _human_bytes
from hermes_state_common import FTS_STORAGE_VERSION
import logging

logger = logging.getLogger("hermes_cli.doctor")


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
    conn = sqlite3.connect(str(state_db_path))
    try:
        return conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
    finally:
        conn.close()


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


def _find_retired_wal_capture(db_path: Path, pid: int, wal_identity: Optional[tuple] = None) -> Optional[Path]:
    """Find an existing retired-wal capture directory matching pid and wal_identity."""
    retired_dirs = sorted(db_path.parent.glob(f"{db_path.name}.retired-wal-*"))
    for d in reversed(retired_dirs):
        manifest_file = d / "manifest.json"
        if not manifest_file.is_file():
            continue
        try:
            import json
            m = json.loads(manifest_file.read_text(encoding="utf-8"))
            if m.get("pid") == pid:
                if wal_identity is None:
                    return d
                captured_ident = tuple(m.get("wal", {}).get("identity") or ())
                if captured_ident == tuple(wal_identity):
                    return d
        except Exception:
            continue
    return None


def _report_retired_dirs_info(retired_dirs: List[Path], profile_arg: str) -> None:
    """Report latest retired WAL capture with mode-aware guidance (header_only vs copied)."""
    if not retired_dirs:
        return
    latest = retired_dirs[-1]
    manifest_file = latest / "manifest.json"
    mode = "unknown"
    if manifest_file.exists():
        import json
        try:
            m = json.loads(manifest_file.read_text(encoding="utf-8"))
            mode = m.get("main", {}).get("mode", "unknown")
        except Exception:
            pass
    if mode == "copied":
        check_info(
            f"Retired WAL capture preserved at {latest.name} (mode: copied; "
            f"inspect with 'hermes {profile_arg}sessions recover --source {latest / 'state.db'} --inspect-only')"
        )
    else:
        check_info(
            f"Retired WAL capture preserved at {latest.name} (mode: {mode}; "
            f"forensic artifact only — inspect {latest / 'manifest.json'})"
        )


def _recover_retired_wal(f: Finding, should_fix: bool, state_db_path: Path, _DHH: str) -> bool:
    """Detect and remediate processes holding unlinked or retired WAL generations."""
    from hermes_state_dbfile import iter_deleted_sqlite_sidecar_holders
    from hermes_constants import profile_cli_selector

    holders = iter_deleted_sqlite_sidecar_holders(state_db_path)
    retired_dirs = sorted(state_db_path.parent.glob(f"{state_db_path.name}.retired-wal-*"))
    profile_arg = profile_cli_selector()

    if not holders:
        _report_retired_dirs_info(retired_dirs, profile_arg)
        return False

    current_pid = os.getpid()
    pids = sorted({pid for pid, _ in holders if pid > 0 and pid != current_pid})
    pids_desc = ", ".join(f"PID {p}" for p in pids)
    check_warn(
        f"{_DHH}/state.db has {len(pids)} process(es) holding an unlinked or retired WAL generation",
        f"({pids_desc})",
    )

    if not should_fix:
        f.issues.append(
            f"state.db has {len(pids)} process(es) holding a retired WAL generation ({pids_desc}) — "
            f"run 'hermes {profile_arg}doctor --fix' to stop conflicting processes and reopen cleanly"
        )
        return True

    from gateway.status import terminate_pid, get_process_start_time, _start_times_agree
    from hermes_state_holders import _read_proc_argv
    from hermes_state_dbfile import (
        iter_deleted_sqlite_sidecar_holder_descriptors,
        capture_external_retired_wal_generation,
    )

    cmd_descs = {pid: (" ".join(_read_proc_argv(pid))[:60] if _read_proc_argv(pid) else f"PID {pid}") for pid in pids}

    # Precondition 1: Process Identity Witness
    # Capture start time while fd/holder is observed; refuse if unavailable.
    witness_start_times: Dict[int, int] = {}
    failed_pids: List[Tuple[int, str]] = []
    for pid in pids:
        st = get_process_start_time(pid)
        if st is None:
            failed_pids.append((pid, f"{cmd_descs.get(pid, f'PID {pid}')} (refusing to signal: start-time identity unavailable)"))
        else:
            witness_start_times[pid] = st

    # Precondition 2: Exact-Generation Preservation
    # Ensure this PID's exact orphaned WAL inode has a durable capture BEFORE sending any signal.
    holder_descriptors = {
        r["pid"]: r for r in iter_deleted_sqlite_sidecar_holder_descriptors(state_db_path)
        if r.get("suffix") == "-wal" and r.get("identity")
    }
    preserved_pids = set()
    for pid in list(witness_start_times.keys()):
        wal_ident = holder_descriptors.get(pid, {}).get("identity")
        existing_capture = _find_retired_wal_capture(state_db_path, pid, wal_ident)
        if existing_capture:
            preserved_pids.add(pid)
            continue
        fd_path = holder_descriptors.get(pid, {}).get("fd_path")
        if fd_path and wal_ident:
            try:
                capture_external_retired_wal_generation(state_db_path, pid=pid, fd_path=fd_path, wal_identity=wal_ident)
                preserved_pids.add(pid)
                continue
            except Exception as exc:
                logger.debug("Failed external capture of WAL for PID %s: %s", pid, exc)
        failed_pids.append((pid, f"{cmd_descs.get(pid, f'PID {pid}')} (refusing to signal: unlinked WAL generation {wal_ident} has no durable capture)"))

    target_pids = [p for p in witness_start_times.keys() if p in preserved_pids]

    # Bounded parallel termination with identity revalidation
    failed_initial = {}
    active_pids = set()
    for pid in target_pids:
        cur_start = get_process_start_time(pid)
        if cur_start is None or not _start_times_agree(cur_start, witness_start_times[pid]):
            # Holder exited or was recycled before TERM: no signal to replacement
            continue
        active_pids.add(pid)
        try:
            terminate_pid(pid, force=False, expected_start_time=witness_start_times[pid])
        except Exception as exc:
            failed_initial[pid] = exc

    def _is_running(p: int) -> bool:
        try:
            os.kill(p, 0)
            return True
        except OSError:
            return False

    for _ in range(20):
        if not active_pids:
            break
        active_pids = {p for p in active_pids if _is_running(p)}
        if active_pids:
            time.sleep(0.1)

    if active_pids:
        for pid in list(active_pids):
            cur_start = get_process_start_time(pid)
            if cur_start is None or not _start_times_agree(cur_start, witness_start_times[pid]):
                # PID was recycled between TERM and KILL: refuse KILL
                active_pids.discard(pid)
                continue
            try:
                terminate_pid(pid, force=True, expected_start_time=witness_start_times[pid])
            except Exception:
                pass
        for _ in range(10):
            if not active_pids:
                break
            active_pids = {p for p in active_pids if _is_running(p)}
            if active_pids:
                time.sleep(0.1)

    still_held = {h[0] for h in iter_deleted_sqlite_sidecar_holders(state_db_path) if h[0] != current_pid}
    stopped_pids = []
    for p in target_pids:
        cmd = cmd_descs.get(p, f"PID {p}")
        if p in active_pids or (p in failed_initial and p in still_held):
            err = f" ({failed_initial[p]})" if p in failed_initial else ""
            failed_pids.append((p, f"{cmd}{err}"))
        else:
            stopped_pids.append((p, cmd))

    if stopped_pids:
        check_ok(
            f"Stopped {len(stopped_pids)} process(es) holding retired WAL",
            f"({', '.join(str(p) for p, _ in stopped_pids)})",
        )
        f.fixed += 1

    if failed_pids:
        check_warn(
            f"Could not stop {len(failed_pids)} process(es) holding retired WAL",
            f"({', '.join(f'{p}: {c}' for p, c in failed_pids)})",
        )
        f.manual_issues.append(
            f"Could not automatically stop {len(failed_pids)} process(es) holding retired WAL: "
            f"{', '.join(f'PID {p} ({c})' for p, c in failed_pids)} — stop them manually and reopen"
        )

    remaining = [h for h in iter_deleted_sqlite_sidecar_holders(state_db_path) if h[0] > 0 and h[0] != current_pid]
    if not remaining:
        try:
            check_ok(f"{_DHH}/state.db reopened successfully ({_session_count(state_db_path)} sessions)")
            pending_spool = state_db_path.parent / "pending_messages"
            if pending_spool.exists():
                spooled_files = list(pending_spool.glob("pending-*.json"))
                if spooled_files:
                    check_info(
                        f"{len(spooled_files)} pending message spool file(s) preserved in {pending_spool.name}/ "
                        "for gateway ingestion"
                    )
        except Exception as e:
            check_warn(f"{_DHH}/state.db reopened but health check reported: {e}")

    refreshed_retired_dirs = sorted(state_db_path.parent.glob(f"{state_db_path.name}.retired-wal-*"))
    _report_retired_dirs_info(refreshed_retired_dirs, profile_arg)

    return True


def _state_db_health(f: Finding, should_fix: bool, state_db_path: Path, _DHH: str) -> None:
    """Session count + FTS write-health probe; malformed-schema path when even COUNT(*) fails."""
    from hermes_state_dbfile import iter_deleted_sqlite_sidecar_holders
    if iter_deleted_sqlite_sidecar_holders(state_db_path):
        _recover_retired_wal(f, should_fix, state_db_path, _DHH)
        return

    try:
        check_ok(f"{_DHH}/state.db exists ({_session_count(state_db_path)} sessions)")
        # COUNT(*) succeeds even when the FTS index is corrupt and every write fails through the triggers;
        # _db_opens_cleanly drives a rolled-back write to surface that.
        from hermes_state_repair import _db_opens_cleanly, state_db_has_structural_damage
        # `_db_opens_cleanly` now drives a rolled-back write so this otherwise-silent corruption class is
        # surfaced (and repaired in place with --fix). See #50502.
        _write_reason = _db_opens_cleanly(state_db_path)
        if _write_reason is not None:
            if state_db_has_structural_damage(state_db_path):
                check_warn(f"{_DHH}/state.db has structural corruption (canonical tables/indexes damaged, "
                           "not the FTS index)", f"({_write_reason})")
                return _repair_state_db(f, should_fix, state_db_path, "structural")
            check_warn(f"{_DHH}/state.db fails a write-health probe (FTS index may be corrupt)", f"({_write_reason})")
            _repair_state_db(f, should_fix, state_db_path, "fts")
    except Exception as e:
        from hermes_state import is_malformed_db_error
        from hermes_state_errors import DeletedWalGenerationError
        if isinstance(e, DeletedWalGenerationError) or "deleted state.db-wal" in str(e):
            _recover_retired_wal(f, should_fix, state_db_path, _DHH)
            return
        if not is_malformed_db_error(e):
            return check_warn(f"{_DHH}/state.db exists but has issues: {e}")
        # sqlite_master itself is malformed (e.g. duplicate messages_fts): every statement fails before it runs,
        # so this is NOT a plain FTS rebuild — repair sqlite_master in place (backup first).
        check_warn(f"{_DHH}/state.db schema is malformed (sessions hidden until repaired)", f"({e})")
        _repair_state_db(f, should_fix, state_db_path, "schema")


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
            from hermes_state_holders import live_writer_holds_db
            from hermes_state_repair import _connect_repair_durable
            is_held = live_writer_holds_db(state_db_path, connect_repair_durable=_connect_repair_durable)
            if not should_fix:
                if is_held:
                    check_warn(
                        f"WAL file is large ({size // (1024*1024)} MB)",
                        "(normal while Desktop/gateway are running; only checkpoint with them stopped)",
                    )
                    return f.issues.append(
                        f"Large WAL file ({size // (1024*1024)} MB) — normal while Desktop/gateway are "
                        "running; stop them first before checkpointing with 'hermes doctor --fix'"
                    )
                check_warn(
                    f"WAL file is large ({size // (1024*1024)} MB)",
                    "(may indicate missed checkpoints; only checkpoint with Desktop/gateway stopped)",
                )
                return f.issues.append(
                    "Large WAL file — run 'hermes doctor --fix' to checkpoint (ensure Desktop/gateway are stopped)"
                )
            # Checkpoint-lock premise (#40177): a bare connect runs WAL recovery and the checkpoint joins the
            # live WAL — under a running gateway that second-writer handling corrupts state.db. Skip instead.
            if is_held:
                # Honest disjunction (gate C1): a True here means "held OR
                # unprovable" — the DatabaseError lane fires when SQLite
                # cannot open the file at all, with nobody holding it. Never
                # assert a live writer as fact.
                check_warn("WAL checkpoint skipped: cannot prove state.db is quiet",
                           "(a live writer holds it, or it is unreadable — stop Desktop and the profile's gateway "
                           "first, then re-run 'hermes doctor --fix')")
                return f.issues.append("Large WAL file — cannot prove state.db is quiet (stop Desktop and the profile's "
                                       "gateway first, then re-run 'hermes doctor --fix' to checkpoint)")
            import contextlib
            import sqlite3
            with contextlib.closing(sqlite3.connect(str(state_db_path))) as conn:
                conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
            check_ok(f"WAL checkpoint performed ({size // 1024}K → {wal_size() // 1024}K)")
            f.fixed += 1
        elif size > 10 * 1024 * 1024:  # 10 MB
            check_info(f"WAL file is {size // (1024*1024)} MB (normal for active sessions)")


@doctor_check()
def _check_state_db(should_fix: bool, f: Finding) -> None:
    """state.db session count, FTS write health, schema repair, stats snapshot, WAL size."""
    from hermes_cli.doctor import HERMES_HOME, _DHH
    state_db_path = HERMES_HOME / "state.db"
    if state_db_path.exists():
        _state_db_health(f, should_fix, state_db_path, _DHH)
        _state_db_stats(f.issues, state_db_path)
    else:
        check_info(f"{_DHH}/state.db not created yet (will be created on first session)")
    _state_db_wal(f, should_fix, state_db_path)


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
