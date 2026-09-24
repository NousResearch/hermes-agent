"""HERMES_HOME state checks for hermes doctor: directories, memory files, state.db health, skills hub, memory provider, profiles.
Split out of ``hermes_cli/doctor.py``, which re-exports every name so ``hermes_cli.doctor.<name>`` keeps resolving (and monkeypatching)."""

from __future__ import annotations

import os
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
        from plugins.memory import import_provider_module
        cfg = import_provider_module("honcho", "client").HonchoClientConfig.from_global_config()
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


def host_gateway_note() -> str:
    """``" (the host gateway (PID 42) serving profiles default, coder)"`` when one gateway process
    owns this host, else ``""``. Multiplex-only: the state.db holder and WAL lines used to imply a
    gateway per profile; the truth is one shared process serving N profiles, and stopping it stops
    every one of them."""
    try:
        from gateway.host_topology import host_gateway_topology
        topology = host_gateway_topology()
    except Exception:
        return ""
    return f" ({topology.describe()})" if topology is not None else ""


def _render_state_db_stats(stats: dict, holders=None, host_note: str = "") -> list:
    """Turn a collect_state_db_stats() dict into ``(kind, text, detail)`` rows, kind 'info' / 'warn'.

    Pure formatting — no I/O — so it is unit-testable without the doctor CLI. Tolerates None in every field.
    ``host_note`` names the shared host gateway among the holders (see :func:`host_gateway_note`).
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
        (holders, lambda: f"{holders} process(es) holding the DB open{host_note}"),
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
                          "(stop ONLY the listed process(es) — the host gateway keeps running and its own "
                          "retry rebuilds within a minute of the holder leaving)"))
        else:
            lines.append(("warn", f"state.db FTS repair is blocked after {deferral.get('attempts') or '?'} deferral(s) "
                          f"by PID(s) {pids}",
                          "(stop the listed processes; the host gateway's own retry then rebuilds, or run "
                          "'hermes sessions optimize-storage' with every holder stopped)"))
    # Oversized DB: suggest auto_prune, plus the offline optimize-storage pass when the FTS rebuild is
    # pending OR the DB predates the current trigram layout (fts_storage_version < FTS_STORAGE_VERSION).
    if logical is not None and logical > STATE_DB_SIZE_WARN_BYTES:
        detail = "consider enabling sessions.auto_prune in config.yaml to bound growth"
        stale_trigram = (fts is not None and fts.get("messages_fts_trigram")
                         and (stats.get("fts_storage_version") or 0) < FTS_STORAGE_VERSION)
        if stats.get("fts_rebuild_pending") or stale_trigram:
            detail += "; run 'hermes sessions optimize-storage' offline (with the host gateway stopped) to compact FTS storage"
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
    _check_scratch_dir(hermes_home, _DHH)
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


# Cache-root entries at least this big that no pruner covers get a doctor warning.
_UNPRUNED_CACHE_WARN_BYTES = 1 << 30
_PRUNED_CACHE_DIRS = frozenset({"scratch", "terminal"})


def unpruned_cache_hogs(hermes_home: Path, min_bytes: int = _UNPRUNED_CACHE_WARN_BYTES) -> list[tuple[str, int]]:
    """``(name, bytes)`` for ``cache/`` entries outside the pruned dirs that exceed *min_bytes*.

    Finished campaign trees parked at the cache root sat for weeks (95 GB on one host)
    because only ``scratch/`` and ``terminal/`` are reaped; doctor is where that shows."""
    from hermes_constants import scratch_dir_usage_bytes

    cache = hermes_home / "cache"
    hogs: list[tuple[str, int]] = []
    try:
        entries = [e for e in cache.iterdir() if e.is_dir() and not e.is_symlink() and e.name not in _PRUNED_CACHE_DIRS]
    except OSError:
        return hogs
    for entry in entries:
        size = scratch_dir_usage_bytes(entry)
        if size >= min_bytes:
            hogs.append((entry.name, size))
    return sorted(hogs, key=lambda item: -item[1])


def _check_scratch_dir(hermes_home: Path, _DHH: str) -> None:
    """Report the scratch dir (TMPDIR target) and its size; a user-set TMPDIR elsewhere is shown, not judged."""
    from hermes_constants import (
        SCRATCH_DIR_MARKER_ENV, SCRATCH_MAX_IDLE_HOURS, get_scratch_dir, scratch_dir_usage_bytes)
    scratch = get_scratch_dir(hermes_home, prune=False)
    size = _human_bytes(scratch_dir_usage_bytes(scratch))
    check_ok(f"{_DHH}/cache/scratch/ is the scratch dir (TMPDIR; {size}, entries pruned after {SCRATCH_MAX_IDLE_HOURS}h idle)")
    for name, nbytes in unpruned_cache_hogs(hermes_home):
        check_warn(
            f"{_DHH}/cache/{name}/ is {_human_bytes(nbytes)} and outside every pruner "
            f"(only cache/scratch/ and cache/terminal/ are reaped) — move task files under "
            f"cache/scratch/<task>/ or delete it"
        )
    tmpdir = os.environ.get("TMPDIR", "")
    if tmpdir and tmpdir != os.environ.get(SCRATCH_DIR_MARKER_ENV, ""):
        check_info(f"TMPDIR={tmpdir} is set by you or the OS, so Hermes leaves it alone")


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
        rows = _render_state_db_stats(collect_state_db_stats(state_db_path), holders=count_db_holders(state_db_path),
                                      host_note=host_gateway_note())
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


def _retired_wal_holders(f: Finding, state_db_path: Path, _DHH: str) -> bool:
    """Name the processes holding a retired -wal/-shm generation (#110054). Every SessionDB open is
    refused while they live, and the current inode has no holders, so the plain holder count says
    "0 holding the DB open" beside a green state.db line — the opposite of the truth."""
    from hermes_constants import profile_cli_selector
    from hermes_state_dbfile import iter_deleted_sqlite_sidecar_holders
    from hermes_state_holders import describe_holder_pid
    pids = list(dict.fromkeys(pid for pid, _ in iter_deleted_sqlite_sidecar_holders(state_db_path)))
    if not pids:
        return False
    rendered = ", ".join(describe_holder_pid(pid) for pid in pids)
    check_warn(f"{_DHH}/state.db: {len(pids)} process(es) still hold a retired WAL generation ({rendered})",
               "(every new session refuses to open until they exit; health/stats probes skipped)")
    f.issues.append(f"state.db retired WAL generation held by {rendered}{host_gateway_note()} — stop the host "
                    f"gateway, dashboard and cron writers among them ('hermes {profile_cli_selector()}gateway "
                    "stop' stops the ONE host process serving every profile, quit the Desktop app), do not "
                    "delete the WAL yourself, then rerun 'hermes doctor'")
    return True


@doctor_check()
def _check_state_db(should_fix: bool, f: Finding) -> None:
    """Session database health, plus SQLite repair, storage stats, and WAL size."""
    from hermes_cli.doctor import HERMES_HOME, _DHH
    from hermes_state_backend import resolve_database_settings
    state_db_path = HERMES_HOME / "state.db"
    # Resolve the configured backend before running SQLite-only WAL checks.
    # Opening SQLite first would add another database holder.
    try:
        settings = resolve_database_settings(state_db_path)
    except Exception as exc:
        check_warn("Session database configuration is invalid", f"({type(exc).__name__})")
        f.manual_issues.append("Check database settings in this profile's config.yaml and .env")
        return
    if settings.backend == "postgres":
        try:
            from hermes_state import SessionDB
            with SessionDB(state_db_path, read_only=True, database_settings=settings) as db, db._read_ctx() as conn:
                count = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
            check_ok(f"PostgreSQL session database is reachable ({count} sessions)")
        except Exception as exc:
            check_warn("PostgreSQL session database is unavailable", f"({type(exc).__name__})")
            f.manual_issues.append("Check PostgreSQL connectivity, database.schema, and this profile's HERMES_DATABASE_URL")
        return
    if _retired_wal_holders(f, state_db_path, _DHH):
        return
    if state_db_path.exists():
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
        check_ok("GitHub token configured", "(validity checked under API Connectivity)")
    else:
        check_bool(_gh_authenticated(), ("GitHub authenticated via gh CLI", "(full API access — no GITHUB_TOKEN needed)"),
                   ("No GITHUB_TOKEN", f"(60 req/hr rate limit — set in {_DHH}/.env for better rates)"))


def _memory_provider_honcho(issues: list) -> None:
    from plugins.memory import import_provider_module
    client = import_provider_module("honcho", "client")
    hcfg = client.HonchoClientConfig.from_global_config()
    cfg_path = client.resolve_config_path()
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
        client.reset_honcho_client()
        try:
            client.get_honcho_client(hcfg)
            check_ok("Honcho connected", f"workspace={hcfg.workspace_id} mode={hcfg.recall_mode} freq={hcfg.write_frequency}")
        except Exception as _e:
            _fail_and_issue("Honcho connection failed", str(_e), f"Honcho unreachable: {_e}", issues)


def _memory_provider_mem0(issues: list) -> None:
    from plugins.memory import import_provider_module
    mem0_cfg = import_provider_module("mem0")._load_config()
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
    from agent.memory_provider import is_core_memory_provider
    name = _doctor_memory_config(HERMES_HOME).get("provider", "")
    if is_core_memory_provider(name):
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
    # Same helper as the multiplex migration preflight, so doctor names the duplicates that make
    # `hermes gateway migrate --multiplex` refuse (and made pre-multiplex standalone gateways race).
    from hermes_cli.gateway_migrate import duplicate_credential_findings
    for line in duplicate_credential_findings():
        check_warn("Duplicate platform credential across profiles", f"({line})")
        f.manual_issues.append(line)
