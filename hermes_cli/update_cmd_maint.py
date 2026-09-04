"""Post-update maintenance for ``hermes update``: pre-update backup snapshot, state-db verify/restore, curator/FTS notices, FHS path guard, completion summary, stale-module purge.

Split out of ``update_cmd.py``, which re-imports every name so ``hermes_cli.update_cmd.<name>``
still resolves/monkeypatches. Origin helpers are imported lazily per function (no cycle;
test patches on ``update_cmd`` stay effective).
"""

import importlib
import logging
from contextlib import suppress
import os
import shutil
import subprocess
import sys
import time as _time
from pathlib import Path
from typing import Optional
from hermes_constants import venv_python_path

from hermes_cli.update_cmd_common import _best_effort

# Log-record parity with the origin module.
logger = logging.getLogger("hermes_cli.update_cmd")


_UPDATE_RUNTIME_RELOAD_MODULES = "hermes_constants", "tools.environments.local", "tools.lazy_deps"

#: Package prefixes whose cached modules go stale when the checkout changes under this
#: process; purged (not reloaded) so any LATER import chain resolves against fresh source.
_STALE_PURGE_PREFIXES = "hermes_cli", "gateway", "tools", "tui_gateway", "agent"

#: Modules EXECUTING the update survive the purge: evicting them buys nothing (running frames
#: keep them alive) and reloading them mid-flight is the one genuinely unsafe move.
_STALE_PURGE_PROTECTED = frozenset(
    {
        "hermes_cli",
        "hermes_cli.main",
        "hermes_cli.update_cmd",
        "hermes_cli.hermes_logging",
    }
    # The updater's own split modules are executing too.
    | {f"hermes_cli.update_cmd_{c}" for c in (
        "config", "deps", "fleet", "git", "maint", "stash", "windows", "zip",
    )}
)

_PRE_UPDATE_SNAPSHOT_KEEP = 1

# Per-file cap for the quick snapshot (larger files skipped with a warning): it protects
# small hard-to-regenerate state, not a multi-GB state.db (24 GB cost ~60s + 24 GB/update).
_PRE_UPDATE_SNAPSHOT_MAX_FILE_SIZE = 1 << 30  # 1 GiB

_SQLITE_WAL_BUG_DETAIL = "SQLite {} still has the WAL-reset corruption bug"


def _load_updates_cfg() -> dict:
    """``updates`` section of config.yaml; ``{}`` on any failure."""
    from hermes_cli.config import load_config
    cfg = load_config() or {}
    updates = cfg.get("updates", {}) if isinstance(cfg, dict) else {}
    return updates if isinstance(updates, dict) else {}


def _reload_modules(names, *, modules, log) -> None:
    """``importlib.reload`` each module of *names* cached in *modules*; failures go to *log*."""
    importlib.invalidate_caches()
    for module_name in names:
        module = modules.get(module_name)
        if module is None:
            continue
        try:
            importlib.reload(module)
        except Exception as exc:
            log(module_name, exc)


def _purge_stale_hermes_modules() -> None:
    """Evict every cached Hermes module after the checkout changed in-place. Never raises.

    The update runs in the pre-pull process; later phases lazily import NEW source into an OLD
    ``sys.modules`` world and die when new code references a symbol missing from a cached
    module. Purging (unlike reload) only drops the ``sys.modules`` entry — running frames keep
    their module objects — so later imports rebuild a self-consistent graph from the new tree.
    """
    from hermes_cli.update_cmd import _m
    with _best_effort('Could not purge stale Hermes modules: %s'):
        importlib.invalidate_caches()
        modules = _m().sys.modules
        purged = [
            name for name in list(modules)
            if name not in _STALE_PURGE_PROTECTED
            # Root-package check: startswith() alone also matches unrelated ``gateway_foo``.
            and name.split(".", 1)[0] in _STALE_PURGE_PREFIXES
            and modules.pop(name, None) is not None
        ]
        if purged:
            logger.debug("Purged %d stale Hermes module(s) after checkout update", len(purged))


def _reload_updated_runtime_modules() -> None:
    """Reload the modules used by lazy-backend refresh: the pre-pull process's cached modules
    can expose old symbols despite new source on disk."""
    from hermes_cli.update_cmd import _m
    with _best_effort('Could not refresh update runtime modules: %s'):
        _reload_modules(
            _UPDATE_RUNTIME_RELOAD_MODULES,
            modules=_m().sys.modules,
            log=lambda name, exc: logger.debug("Could not reload updated module %s: %s", name, exc),
        )


def _print_curator_first_run_notice() -> None:
    """Curator heads-up after update. Fires only when enabled AND never run — the window where
    the first pass (deferred one ``interval_hours``) is pending, so the user can preview or
    disable it first. Silent on steady state."""
    try:
        from agent import curator
        if not curator.is_enabled():
            return
        state = curator.load_state()
    except Exception:
        return
    if state.get("last_run_at"):
        return
    try:
        hours = curator.get_interval_hours()
    except Exception:
        hours = 24 * 7
    days = max(1, hours // 24)
    print()
    print("ℹ Skill curator")
    print(
        f"  Background skill maintenance is enabled. First pass is deferred "
        f"~{days}d after installation; only agent-created skills are in "
        f"scope and nothing is ever auto-deleted (archive is recoverable)."
    )
    print("  Preview now:  hermes curator run --dry-run")
    print("  Pause it:     hermes curator pause")
    print("  Docs:         https://hermes-agent.nousresearch.com/docs/user-guide/features/curator")


def _print_fts_optimize_available_notice() -> None:
    """Advertise the opt-in FTS storage rebuild when state.db still needs one.

    ``sessions.fts_optimize_notice``: ``advise`` (default), ``require`` (firmer), ``off``.
    """
    try:
        from hermes_cli.config import load_config
        mode = str(((load_config() or {}).get("sessions") or {}).get("fts_optimize_notice", "advise")).strip().lower()
    except Exception:
        mode = "advise"
    if mode == "off":
        return

    try:
        from hermes_constants import get_hermes_home
        from hermes_state import SessionDB
    except Exception:
        return
    db_path = get_hermes_home() / "state.db"
    if not db_path.exists():
        return
    try:
        size_gb = db_path.stat().st_size / (1024 ** 3)
    except OSError:
        return
    # Small DBs: the win isn't worth the nag.
    if size_gb < 0.5:
        return
    db = None
    needs_upgrade = False
    try:
        db = SessionDB(db_path=db_path, read_only=True)
        # read_only opens skip schema init; probe the stored layout directly.
        row = db._conn.execute(
            "SELECT sql FROM sqlite_master "
            "WHERE type = 'table' AND name = 'messages_fts'"
        ).fetchone()
        needs_upgrade = bool(row) and getattr(db, "_db_needs_fts_storage_upgrade")(db._conn)
        # Interrupted optimize-storage: v23 table shape but backfill markers / trash
        # tables remain. Re-running resumes it, so offer the command again.
        interrupted = bool(
            db._conn.execute(
                "SELECT 1 FROM state_meta "
                "WHERE key = 'fts_rebuild_high_water' LIMIT 1"
            ).fetchone()
            or db._conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' "
                "AND name LIKE 'fts\\_v22\\_trash\\_%' ESCAPE '\\' LIMIT 1"
            ).fetchone()
            or db._conn.execute(
                "SELECT 1 FROM state_meta WHERE key IN "
                "('fts_cjk_rebuild_high_water', 'fts_cjk_stale') LIMIT 1"
            ).fetchone()
        )
    except Exception:
        return
    finally:
        if db is not None:
            with suppress(Exception):
                db.close()
    if not needs_upgrade and not interrupted:
        return  # current layout already present (fresh/optimized)

    if interrupted:
        print()
        print("◆ Session database optimization incomplete")
        print(
            "  A previous `hermes sessions optimize-storage` run was "
            "interrupted. Search still works; re-run the command to resume "
            "and finish reclaiming disk:"
        )
        print("    hermes sessions optimize-storage")
        return

    est_reclaim = size_gb * 0.6
    print()
    if mode == "require":
        print("◆ Session database upgrade required")
        print(
            f"  Your search index uses the OLD storage layout and should be "
            f"upgraded. The new layout typically frees ~60% of state.db "
            f"(≈{est_reclaim:.1f} GB of your current {size_gb:.1f} GB) and is "
            f"required for continued optimal operation."
        )
    else:
        print("◆ Reclaim ~60% of your session database disk")
        print(
            f"  Your search index uses the old storage layout. Upgrading it "
            f"typically frees ~60% of state.db — about {est_reclaim:.1f} GB "
            f"of your current {size_gb:.1f} GB."
        )
    print("  Run when convenient:  hermes sessions optimize-storage")
    print(
        "  It runs in the foreground with a progress bar, is safe to "
        "interrupt/re-run, and never changes your conversations."
    )


def _print_curator_recent_run_notice() -> None:
    """Print the latest background curator run summary (rename map) once, stamping
    ``last_run_summary_shown_at``. Silent when never run, already shown, or no rename info."""
    try:
        from agent import curator
        state = curator.load_state()
    except Exception:
        return

    last_run_at = state.get("last_run_at")
    if not last_run_at:
        return  # no curator run yet — first-run notice handles this case
    if state.get("last_run_summary_shown_at") == last_run_at:
        return  # already shown for this run
    summary = state.get("last_run_summary") or ""
    if not summary:
        return

    # Only a multi-line summary (rename map) is worth showing; still stamp it shown.
    if "\n" in summary:
        print()
        print(f"ℹ Skill curator — last run {_format_time_ago(last_run_at)}")
        for line in summary.splitlines():
            print(f"  {line}")
        print("  (This message shows once per curator run. View anytime: hermes curator status)")

    with suppress(Exception):
        state["last_run_summary_shown_at"] = last_run_at
        curator.save_state(state)


def _format_time_ago(iso_ts: str) -> str:
    """Render an ISO timestamp as `Xh ago` / `Xd ago` / `Xm ago`. Best effort."""
    try:
        from datetime import datetime, timezone
        ts = datetime.fromisoformat(iso_ts.replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        secs = int((datetime.now(timezone.utc) - ts).total_seconds())
        if secs < 60:
            return "just now"
        if secs < 3600:
            return f"{secs // 60}m ago"
        if secs < 86400:
            return f"{secs // 3600}h ago"
        return f"{secs // 86400}d ago"
    except Exception:
        return "recently"


def _reload_process_scan_modules() -> None:
    """Reload the process-scan modules, dependency-first, so ``dashboard_procs`` binds against a
    fresh ``_subprocess_compat``: cleanup runs in the PRE-update process and a symbol the update
    added would otherwise ImportError after the code update succeeded. Called from the cleanup
    entry point so every caller (git path, ZIP fallback) is covered.

    ``_finish_dashboard_update_cleanup`` runs in the PRE-update Python process, but
    ``_scan_dashboard_processes`` does a function-level ``from hermes_cli._subprocess_compat import
    bounded_probe_run``. If the update added a new symbol to ``_subprocess_compat`` (as #87134 did with
    ``bounded_probe_run``), the cached OLD module object doesn't have it and the cleanup step crashes with
    ImportError — after the code update itself already succeeded.
    """
    _reload_modules(
        ("hermes_cli._subprocess_compat", "hermes_cli.dashboard_procs"),
        modules=sys.modules,
        # warning, not debug: a failed reload surfaces as ImportError seconds later.
        log=lambda name, exc: logger.warning(
            "Could not reload %s for post-update cleanup: %s", name, exc
        ),
    )


def _finish_dashboard_update_cleanup(
    node_failures: list[str], already_restarted_units: "set[str] | None" = None
) -> None:
    """Refresh managed dashboards or stop stale manual ones after an update.

    *already_restarted_units*: systemd unit names (no ``.service``) the fleet-restart loop
    already restarted, so a Serve-only install isn't restarted a second time here.

    See #83595.
    """
    from hermes_cli.update_cmd import _m, _reload_process_scan_modules
    if node_failures:
        print()
        print("  ℹ Leaving running dashboard process(es) untouched because the")
        print("    Node.js dependency refresh did not complete.")
        return

    _reload_process_scan_modules()

    stop_result = _m()._kill_stale_dashboard_processes(
        restart_managed=True, already_restarted_units=already_restarted_units
    )
    if not stop_result.get("unrecovered"):
        return

    print()
    print("⚠ A web dashboard/serve process was stopped during update and could not be auto-restarted.")
    print("  Re-launch it when you want the web UI back:")
    print("    hermes dashboard --port <port>")


def _print_update_completion(message: str) -> None:
    """Print the outcome (with branch @ sha so drift is visible) plus, when launched by the
    dashboard with an action id, a receipt line the Desktop matches after restart.

    See #47359, #58764.
    """
    from hermes_cli.update_cmd import _branch_head_suffix
    print(f"{message}{_branch_head_suffix()}")
    action_id = os.environ.get("HERMES_ACTION_ID", "")
    if len(action_id) == 32 and all(char in "0123456789abcdef" for char in action_id):
        print(f"=== hermes-update completed {action_id} ===")


def _read_project_version() -> str | None:
    """``version`` from the checkout's pyproject.toml (not importlib.metadata, which still
    describes the OLD version after a pull). None on any failure — cosmetic, never breaks."""
    from hermes_cli.update_cmd import _m
    try:
        import tomllib
        with open(_m().PROJECT_ROOT / "pyproject.toml", "rb") as fh:  # windows-footgun: ok — binary mode, tomllib requires bytes
            version = tomllib.load(fh).get("project", {}).get("version")
        return str(version) if version else None
    except Exception:
        return None


def _update_complete_message(pre_version: str | None) -> str:
    """Completion line with ``vA → vB`` when known; plain when either side is unknown or
    the version did not change.

    Ported from PrimeIntellect-ai/prime-agent#630: after a successful self-update, show both versions
    (``v0.19.4 → v0.20.0``) so the user can see what they actually got. Falls back to the plain message when
    either side is unknown or the version did not change (e.g. several commits landed within one release).
    """
    post_version = _read_project_version()
    if pre_version and post_version and pre_version != post_version:
        return f"✓ Update complete! (v{pre_version} → v{post_version})"
    if post_version:
        return f"✓ Update complete! (v{post_version})"
    return "✓ Update complete!"


def _post_update_sqlite_runtime_status():
    """Return whether the interpreter used after update has safe SQLite."""
    from hermes_cli.update_cmd import _m
    from hermes_constants import project_venv_dir
    from hermes_cli.sqlite_runtime import probe_sqlite_runtime
    venv_dir = project_venv_dir(_m().PROJECT_ROOT)
    python = (venv_python_path(venv_dir, windows=_m()._is_windows()) if venv_dir is not None else Path(sys.executable))
    info = probe_sqlite_runtime(python)
    return info is not None and not info.wal_reset_vulnerable, info


def _print_verified_update_completion(message: str) -> bool:
    """Print a success completion only after probing the next Hermes runtime."""
    from hermes_cli.update_cmd import _post_update_sqlite_runtime_status
    if not message.startswith("✓"):
        _print_update_completion(message)
        return False
    sqlite_runtime_ok, sqlite_info = _post_update_sqlite_runtime_status()
    if sqlite_info is None:
        # Grace path: an unprobeable interpreter (dev checkout, no probe subprocess) must not
        # fail the update — only a POSITIVE vulnerable probe withholds success.
        logger.debug("Post-update SQLite runtime probe unavailable; not blocking")
    if sqlite_info is None or sqlite_runtime_ok:
        _print_update_completion(message)
        return True
    print()
    print(f"⚠ Update partially complete — {_SQLITE_WAL_BUG_DETAIL.format(sqlite_info.sqlite_version_string)}.")
    print("  Rebuild the Hermes venv with a uv-managed Python, restart Hermes, then verify with `hermes doctor`.")
    return False


def _clear_stale_sqlite_sidecars(db_path: Path) -> None:
    """Delete -wal/-shm/-journal next to *db_path*, immediately before overwriting it with a
    snapshot image.

    Snapshots are checkpointed ``sqlite3.backup()`` images with no WAL; copying replaces only
    the main file, so a leftover WAL from the OLD database would be replayed over the fresh
    image on next open (passes integrity_check while serving old contents). Safe because the
    caller has already declared that database corrupt.
    """
    for suffix in ("-wal", "-shm", "-journal"):
        db_path.with_name(db_path.name + suffix).unlink(missing_ok=True)


def _print_update_summary(*, node_failures: list, desktop_build_ok: bool, pre_update_version: str | None) -> bool:
    """Final banner. A failed Desktop rebuild is non-fatal but must not print ``✓ Update complete!``.

    See #88251.
    """
    from hermes_cli.update_cmd import _post_update_sqlite_runtime_status, _update_complete_message
    sqlite_runtime_ok, sqlite_info = _post_update_sqlite_runtime_status()
    if sqlite_info is None:
        # Grace path: only a POSITIVE vulnerable probe demotes success to partial.
        sqlite_runtime_ok = True
    print()
    if node_failures or not desktop_build_ok or not sqlite_runtime_ok:
        parts = []
        if node_failures:
            parts.append(f"Node.js dependencies for {', '.join(node_failures)} did not refresh")
        if not desktop_build_ok:
            parts.append("the desktop app was not rebuilt and is still on the previous build")
        if not sqlite_runtime_ok and sqlite_info is not None:
            parts.append(_SQLITE_WAL_BUG_DETAIL.format(sqlite_info.sqlite_version_string))
        print("⚠ Update partially complete — " + "; ".join(parts) + ".")
        if node_failures:
            print("  Code and Python deps are updated, but the dashboard/TUI may")
            print("  be in a mixed state until the Node deps are rebuilt.")
        if not desktop_build_ok:
            print("  Run `hermes desktop` to retry the desktop rebuild.")
        if not sqlite_runtime_ok:
            print(
                "  The Python runtime remediation did not complete. Run `hermes "
                "update` again; if SQLite is unchanged, rebuild the Hermes venv "
                "with a uv-managed Python, restart Hermes, then verify with "
                "`hermes doctor`."
            )
    else:
        _print_update_completion(_update_complete_message(pre_update_version))
    return desktop_build_ok and sqlite_runtime_ok


def _restore_state_db_from_snapshot(state_path: Path, snap_state: Path) -> bool:
    """Replace *state_path* with the snapshot image; True when the result passes integrity.

    Stale sidecars are cleared first so the corrupt DB's WAL can't replay over the image.
    Refuses (False) while another process — or a live connection in THIS process — holds the
    DB: copying over a live writer's inode desyncs its page cache and its next checkpoint
    clobbers pages. Holder scan ``None`` proceeds (gateways drained; refusing on unknown would
    disable auto-restore on non-Linux). Raises OSError if the copy fails.
    """
    from hermes_cli.backup import _foreign_db_holder_pids, verify_sqlite_integrity
    from hermes_cli.sqlite_safe_read import LiveConnectionError, offline_file_access
    holders = _foreign_db_holder_pids(state_path)
    if holders:
        print(
            f"  ✗ Auto-restore refused: process(es) {holders} still hold "
            "state.db or its WAL open. Stop them (hermes gateway stop), "
            "then restore manually with /snapshot restore."
        )
        return False
    # The foreign-pid scan excludes THIS process; an in-process SessionDB handle is just as
    # live (it would checkpoint through deleted-inode sidecars). offline_file_access fails
    # CLOSED on any tracked connection and holds the lock across clear + copy.
    try:
        with offline_file_access(state_path, what="restore a snapshot over"):
            _clear_stale_sqlite_sidecars(state_path)
            shutil.copy2(snap_state, state_path)
    except LiveConnectionError as exc:
        print(
            f"  ✗ Auto-restore refused: {exc} Close the in-process database "
            "handles (or restart Hermes) and retry."
        )
        return False
    restored = verify_sqlite_integrity(state_path, check_header=True, run_pragma=True)
    return bool(restored.get("valid"))


def _verify_and_restore_one_state_db(home: Path, *, label: str) -> None:
    """Integrity check + auto-restore for ONE home's state.db from its newest valid snapshot.
    Never raises: a guard that crashes the update tail is worse than what it detects."""
    try:
        from hermes_cli.backup import _quick_snapshot_root, verify_sqlite_integrity
        state_path = home / "state.db"
        if not state_path.exists():
            return
        ok = verify_sqlite_integrity(state_path, check_header=True, run_pragma=True)
        if ok.get("valid"):
            logger.debug("Post-update state.db integrity OK (%s): %s", label, ok.get("message"))
            return
        print()
        print(f"⚠ state.db is corrupted after update ({label}): " + ok.get("message", "unknown error"))
        snap_root = _quick_snapshot_root(home)
        if not snap_root.exists():
            print("  ⚠ No pre-update snapshot for this home")
            return
        for snap_dir in sorted((d for d in snap_root.iterdir() if d.is_dir()), reverse=True):
            snap_state = snap_dir / "state.db"
            if not snap_state.exists():
                continue
            if not verify_sqlite_integrity(snap_state, check_header=True, run_pragma=True).get("valid"):
                continue
            try:
                if _restore_state_db_from_snapshot(state_path, snap_state):
                    print(f"  ✓ Auto-restored from snapshot {snap_dir.name} ({label})")
                else:
                    print("  ✗ Auto-restore FAILED — restored copy also failed integrity")
            except OSError as exc:
                print(f"  ✗ Auto-restore file copy failed: {exc}")
            return
        print("  ⚠ No valid pre-update snapshot found for this home")
    except Exception as exc:
        logger.debug("Post-update state.db guard (%s) failed: %s", label, exc)


def _verify_and_restore_state_dbs_post_update() -> None:
    """Integrity guard for the ROOT state.db AND every sibling profile's (the snapshot covers
    siblings, so the guard must too or a corrupt profile DB goes undetected).

    See #97994.
    """
    from hermes_cli.update_cmd import get_hermes_home
    home = get_hermes_home()
    _verify_and_restore_one_state_db(home, label="default home")
    with _best_effort('Sibling-profile state.db guard sweep failed: %s'):
        from hermes_cli.backup import _sibling_profile_homes
        for name, profile_home in _sibling_profile_homes(home):
            _verify_and_restore_one_state_db(profile_home, label=f"profile {name}")


def _print_bundled_skills_sync_report() -> None:
    """Run ``sync_skills`` (copies new, updates changed, respects user deletions) and print its summary."""
    from tools.skills_sync import sync_skills
    result = sync_skills(quiet=True)
    if result["copied"]:
        print(f"  + {len(result['copied'])} new: {', '.join(result['copied'])}")
    if result.get("updated"):
        print(f"  ↑ {len(result['updated'])} updated: {', '.join(result['updated'])}")
    if result.get("user_modified"):
        print(f"  ~ {len(result['user_modified'])} user-modified (kept)")
        print("    → see them: hermes skills list-modified  (diff/reset to resume updates)")
    if result.get("cleaned"):
        print(f"  − {len(result['cleaned'])} removed from manifest")
    if result.get("relocated"):
        print(f"  → {len(result['relocated'])} moved to new upstream paths: {', '.join(result['relocated'])}")
    if not result["copied"] and not result.get("updated"):
        print("  ✓ Skills are up to date")


def _ensure_fhs_path_guard() -> None:
    """Ensure /usr/local/bin is on PATH for RHEL-family root non-login shells (su, sudo -s,
    tmux), where neither /etc/bashrc nor .bash_profile adds it. Mirrors install.sh. Idempotent;
    no-op on non-Linux/non-root/non-FHS or when ``bash -i -c 'command -v hermes'`` resolves."""
    from hermes_cli.update_cmd import _m
    if _m().sys.platform != "linux":
        return
    try:
        if os.geteuid() != 0:  # windows-footgun: ok — Linux FHS helper, guarded by sys.platform == "linux" above + AttributeError catch
            return
    except AttributeError:
        return
    # Only for FHS-layout installs (link at /usr/local/bin/hermes).
    fhs_link = Path("/usr/local/bin/hermes")
    if not fhs_link.is_symlink() and not fhs_link.exists():
        return

    # ``bash -i -c`` sources ~/.bashrc but NOT ~/.bash_profile or /etc/profile — the exact
    # scenario where RHEL root loses /usr/local/bin.
    home = os.environ.get("HOME") or "/root"
    try:
        probe = subprocess.run(
            [
                "env",
                "-i",
                f"HOME={home}",
                f"TERM={os.environ.get('TERM', 'dumb')}",
                "bash",
                "-i",
                "-c",
                "command -v hermes",
            ],
            # Fallback: blunt systemctl restart. This is what the old code always did; we get here only when
            # the graceful path failed (unit missing SIGUSR1 wiring, drain exceeded the budget,
            # restart-policy mismatch). Always `reset-failed` first. If systemd's own auto-restart attempts
            # already parked the unit in a failed state (transient CHDIR / OOM / filesystem race after our
            # drain + exit-75), a plain `systemctl restart` can wedge against the RestartSec backoff and
            # leave the unit dead. Clearing the failed state first makes the restart idempotent. Mirrors the
            # recovery path in `hermes gateway restart` (`systemd_restart()`) as of PR #20949.
            capture_output=True,
            text=True, encoding="utf-8", errors="replace",
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return  # no bash or probe hung — don't block update on this
    if probe.returncode == 0:
        return  # already on PATH, nothing to do

    path_line = 'export PATH="/usr/local/bin:$PATH"'
    path_comment = "# Hermes Agent — ensure /usr/local/bin is on PATH (RHEL non-login shells)"
    wrote_any = False
    for candidate in (".bashrc", ".bash_profile"):
        cfg = Path(home) / candidate
        if not cfg.is_file():
            continue
        try:
            existing = cfg.read_text(errors="replace", encoding="utf-8")
        except OSError:
            continue
        # Idempotency: any uncommented PATH line referencing /usr/local/bin (install.sh grep).
        if any(
            "/usr/local/bin" in line and "PATH" in line and not line.lstrip().startswith("#")
            for line in existing.splitlines()
        ):
            continue
        try:
            with cfg.open("a", encoding="utf-8") as f:
                f.write("\n" + path_comment + "\n" + path_line + "\n")
        except OSError as e:
            print(f"  ⚠ Could not update {cfg}: {e}")
            continue
        print(f"  ✓ Added /usr/local/bin to PATH in {cfg}")
        wrote_any = True
    if wrote_any:
        print("    (reload your shell or run 'source ~/.bashrc' to pick it up)")


def _ensure_acp_launcher() -> None:
    r"""Self-heal a ``hermes-acp`` launcher next to ``hermes`` (mirrors install.sh): ACP hosts
    resolve it on the login-shell PATH but the console script lives in the venv. The shim
    delegates to the sibling ``hermes acp``, correct for every layout.

    No-op on Windows (install.ps1 stages launchers into ``$HermesHome\bin``, never
    ``venv\Scripts`` which would shadow the user's python; launcher repair lives in
    _install_repair) and where it already exists. Unwritable dirs are skipped. Idempotent.

    ``/usr/local/bin`` as non-root) are skipped silently. See #83797.
    """
    from hermes_cli.update_cmd import _m
    if _m().sys.platform == "win32":
        return
    for bin_dir in (Path.home() / ".local" / "bin", Path("/usr/local/bin")):
        hermes_cmd = bin_dir / "hermes"
        acp_cmd = bin_dir / "hermes-acp"
        try:
            if not (hermes_cmd.is_file() or hermes_cmd.is_symlink()):
                continue
            # is_symlink() catches broken symlinks exists() misses; never follow-and-overwrite.
            # Already present — a console script (pip/pipx install), an earlier shim, or a symlink.
            # is_symlink() catches broken symlinks that exists() would miss; never follow-and-overwrite (the
            # #21454 failure mode).
            if acp_cmd.exists() or acp_cmd.is_symlink():
                continue
            shim = (
                "#!/usr/bin/env bash\n"
                "# Hermes Agent — ACP launcher (written by `hermes update`).\n"
                "# ACP hosts (Zed, JetBrains, Buzz) resolve the agent by this\n"
                "# command name on the login-shell PATH.\n"
                f'exec "{hermes_cmd}" acp "$@"\n'
            )
            acp_cmd.write_text(shim, encoding="utf-8")
            acp_cmd.chmod(acp_cmd.stat().st_mode | 0o755)
        except OSError:
            continue
        print(f"  ✓ Installed hermes-acp launcher → {acp_cmd}")


_BACKUP_MODE_ALIASES = {
    "off": "off", "false": "off", "none": "off", "disabled": "off",
    "full": "full", "zip": "full", "true": "full",
    "quick": "quick",
}


def _resolve_pre_update_backup_mode(args) -> str:
    """Backup mode ``off``/``quick``/``full``. CLI flags win (``--no-backup`` beats ``--backup``);
    config accepts mode strings plus legacy booleans (true→full, false→off, which also disables
    the quick snapshot). Default ``quick``."""
    if getattr(args, "no_backup", False):
        return "off"
    if getattr(args, "backup", False):
        return "full"

    try:
        raw = _load_updates_cfg().get("pre_update_backup", "quick")
    except Exception as exc:
        logger.debug("Could not load config for pre-update backup: %s", exc)
        raw = "quick"

    if raw is True:
        return "full"
    if raw is False:
        return "off"
    mode = _BACKUP_MODE_ALIASES.get(str(raw).strip().lower())
    if mode is None:
        logger.warning("Unknown updates.pre_update_backup value %r — using 'quick'", raw)
        return "quick"
    return mode


def _verify_state_db_after_snapshot(snapshot_id: str) -> None:
    """Verify live state.db after the snapshot: a concurrent process (antivirus, killed
    gateway, Windows filter driver) can corrupt it and we'd otherwise exit 0 silently."""
    from hermes_cli.backup import _quick_snapshot_root, verify_sqlite_integrity
    from hermes_cli.config import get_hermes_home
    _src_path = get_hermes_home() / "state.db"
    if not _src_path.exists():
        return
    _integrity = verify_sqlite_integrity(
        _src_path, check_header=True, run_pragma=True, max_bytes=_PRE_UPDATE_SNAPSHOT_MAX_FILE_SIZE,
    )
    if _integrity.get("valid"):
        return
    print(f"  ⚠ state.db integrity check FAILED after snapshot: {_integrity.get('message', 'unknown error')}")
    _snap_state = _quick_snapshot_root(get_hermes_home()) / snapshot_id / "state.db"
    if not _snap_state.exists():
        print("  ⚠ Snapshot does not contain state.db (was skipped or too large).")
    elif verify_sqlite_integrity(_snap_state, check_header=True, run_pragma=True).get("valid"):
        print("  ✓ Snapshot copy is valid — continuing update.")
        print("    If state.db is lost after update it will be auto-restored.")
    else:
        print("  ✗ Snapshot copy ALSO failed integrity — the source was already corrupted before the backup.")
    print()


def _run_quick_snapshots() -> Optional[str]:
    """Quick snapshot of the root home plus every sibling profile; returns the root snapshot id."""
    from hermes_cli.update_cmd import _record_update_step
    from hermes_cli.backup import create_quick_snapshot
    snapshot_id = create_quick_snapshot(
        label="pre-update", keep=_PRE_UPDATE_SNAPSHOT_KEEP, max_file_size=_PRE_UPDATE_SNAPSHOT_MAX_FILE_SIZE,
    )
    if snapshot_id:
        _verify_state_db_after_snapshot(snapshot_id)
        print(f"◆ Pre-update snapshot: {snapshot_id}")

    # The code swap + fleet restart touch EVERY profile, so each gets the same snapshot
    # under its own state-snapshots/. Best-effort per profile.
    with _best_effort('Sibling profile snapshots failed: %s'):
        from hermes_cli.backup import create_pre_update_snapshots_all_profiles
        _sibling_snaps = create_pre_update_snapshots_all_profiles(
            keep=_PRE_UPDATE_SNAPSHOT_KEEP, max_file_size=_PRE_UPDATE_SNAPSHOT_MAX_FILE_SIZE,
        )
        if _sibling_snaps:
            print(f"◆ Sibling profile snapshot(s): " + ", ".join(sorted(_sibling_snaps)))
            _record_update_step(
                "sibling_profile_snapshots",
                True,
                ", ".join(f"{k}={v}" for k, v in sorted(_sibling_snaps.items())),
            )
            import hermes_cli.update_cmd_config as _cfg
            # The reader lives in update_cmd_config; write ITS module global, not ours.
            _cfg._LAST_SIBLING_SNAPSHOTS = _sibling_snaps
    return snapshot_id


def _run_full_backup() -> None:
    """Zip HERMES_HOME under ``backups/`` (restorable via ``hermes import``). Never raises."""
    try:
        from hermes_cli.backup import create_pre_update_backup
    except Exception as exc:
        print(f"⚠ Pre-update backup: could not load backup module ({exc}); continuing update.")
        print()
        return

    try:
        _keep = _load_updates_cfg().get("backup_keep", 5)
    except Exception:
        _keep = 5

    print("◆ Creating pre-update backup...")
    t0 = _time.monotonic()
    try:
        out_path = create_pre_update_backup(keep=int(_keep))
    except Exception as exc:  # defensive — helper already swallows, but just in case
        print(f"  ⚠ Backup failed: {exc}")
        print("  Continuing with update.")
        print()
        return
    elapsed = _time.monotonic() - t0

    if out_path is None:
        print("  ⚠ Backup skipped (no files found or write failed); continuing update.")
        print()
        return

    try:
        size_bytes = out_path.stat().st_size
    except OSError:
        size_bytes = 0

    from hermes_cli.sizefmt import format_bytes
    # display_hermes_home so the user sees ~/.hermes/...
    try:
        from hermes_constants import get_hermes_home, display_hermes_home
        display_path = f"{display_hermes_home()}/{out_path.relative_to(get_hermes_home())}"
    except Exception:
        display_path = str(out_path)

    print(f"  Saved:    {display_path} ({format_bytes(size_bytes)}, {elapsed:.1f}s)")
    print(f"  Restore:  hermes import {out_path}")
    print("  Disable:  set updates.pre_update_backup: quick (or off) in config.yaml")
    print()


def _run_pre_update_backup(args) -> Optional[str]:
    """Run the pre-update backup; return the quick-snapshot id (None when off/failed). Never raises.

    ``off`` — nothing. ``quick`` (default) — snapshot of critical small files under
    ``state-snapshots/``, files over 1 GiB skipped so a bloated state.db can't stall the update.
    ``full`` — quick snapshot PLUS a zip of HERMES_HOME under ``backups/`` (``hermes import``).

    Explicit user opt-out is honored fully. See #34600.
    """
    mode = _resolve_pre_update_backup_mode(args)

    if mode == "off":
        if getattr(args, "no_backup", False):
            print("◆ Pre-update backup: skipped (--no-backup)")
            print()
        # Config-level off is silent: the user opted out.
        return None

    snapshot_id = None
    with _best_effort('Pre-update snapshot failed: %s'):
        snapshot_id = _run_quick_snapshots()

    if mode != "full":
        if snapshot_id:
            print()
        return snapshot_id

    _run_full_backup()
    return snapshot_id


def _sweep_bytecode_after_update(branch: str) -> None:
    """Clear stale ``__pycache__`` (else gateway restart ImportErrors on names absent from old
    bytecode), re-stamp the fingerprint, refresh the bootstrap cache scripts."""
    from hermes_cli.update_cmd import _m
    # The update process is still the old Python interpreter process. Run one final cache/module refresh
    # immediately before lazy backend refresh, which imports newly-pulled modules that may depend on fresh
    # symbols in hermes_constants or lazy_deps. The dependency install above may also have regenerated
    # bytecode from build-cache copies — this second sweep catches those stragglers (#60242, #65240).
    removed = _m()._clear_bytecode_cache(_m().PROJECT_ROOT)
    if removed:
        print(f"  ✓ Cleared {removed} stale __pycache__ director{'y' if removed == 1 else 'ies'}")
    _m()._record_bytecode_fingerprint()
    _m()._refresh_bootstrap_cache_scripts(branch)


def _profile_skill_sync_status(r) -> str:
    if r and r.get("skipped_opt_out"):
        return "opted out (--no-skills)"
    if not r:
        return "sync failed"
    parts = []
    for key, fmt in (("copied", "+{} new"), ("updated", "↑{} updated"), ("user_modified", "~{} user-modified")):
        count = len(r.get(key, []))
        if count:
            parts.append(fmt.format(count))
    return ", ".join(parts) if parts else "up to date"


def _sync_profiles_after_update() -> None:
    """Best-effort per-profile syncs: bundled skills, ``.env`` backfill, Honcho profiles."""
    # All profiles incl. the active one: seed_profile_skills() subprocesses with an explicit
    # HERMES_HOME, so sync_skills()'s module-level HERMES_HOME cache can't skew it.
    with suppress(Exception):
        from hermes_cli.profiles import list_profiles, seed_profile_skills
        all_profiles = list_profiles()
        if all_profiles:
            print()
            print("→ Syncing bundled skills to all profiles...")
            for p in all_profiles:
                try:
                    print(f"  {p.name}: {_profile_skill_sync_status(seed_profile_skills(p.path, quiet=True))}")
                except Exception as pe:
                    print(f"  {p.name}: error ({pe})")

    # Backfill .env for profiles created before .env seeding (copy the default's) so they
    # keep the credentials they were effectively using.
    with suppress(Exception):
        # See #44792.
        from hermes_cli.profiles import backfill_profile_envs
        backfilled = backfill_profile_envs(quiet=True)
        if backfilled:
            print()
            print(f"→ Seeded .env for {len(backfilled)} profile(s) (copied from default): {', '.join(backfilled)}")

    with suppress(Exception):
        from plugins.memory.honcho.cli import sync_honcho_profiles_quiet
        synced = sync_honcho_profiles_quiet()
        if synced:
            print(f"\n-> Honcho: synced {synced} profile(s)")


def _refresh_cua_driver_after_update() -> None:
    """cua-driver refresh, no-op unless on PATH; tied to update for a predictable cadence
    without a per-launch GitHub API call."""
    refresh_cua_driver = True
    with _best_effort('Could not read updates.refresh_cua_driver: %s'):
        refresh_cua_driver = bool(_load_updates_cfg().get("refresh_cua_driver", True))

    if (
        refresh_cua_driver and sys.platform in ("darwin", "win32", "linux") and shutil.which("cua-driver")
    ):
        from hermes_cli.tools_config import install_cua_driver
        print()
        print("→ Refreshing cua-driver (Computer Use)...")
        # require_confirmed_update: install only when check-update positively reports a
        # newer release (update must stay fast; `computer-use install --upgrade` forces).
        # Windows defers even confirmed updates (installer may need console/UAC consent).
        install_cua_driver(upgrade=True, require_confirmed_update=True, show_installer_progress=False)


def _print_plugin_compat_notice() -> None:
    """Installed plugins importing paths that the Sep 2026 decomposition scheduled for removal."""
    from hermes_cli.plugin_compat import compat_report, removal_in_effect, summary_lines
    lines = summary_lines(compat_report(force=True))
    if not lines:
        return
    colour = "\033[1;31m" if removal_in_effect() else "\033[1;33m"
    print(f"\n{colour}⚠  {lines[0]}\033[0m\n   {lines[1]}")


def _print_post_update_notices_and_self_heals() -> None:
    """Best-effort notices (FTS optimize, curator) and self-heals (FHS PATH, ACP launcher,
    Windows bin launchers, cua-driver refresh) that run after the summary."""
    from hermes_cli.update_cmd import _m, _print_curator_first_run_notice, _print_curator_recent_run_notice

    def _migrate_windows_bin_path() -> None:
        # Windows launchers into the managed bin dir: in-checkout launchers were swept by the
        # autostash (--include-untracked) and updates never run install.ps1. No-op on POSIX.
        from hermes_cli._install_repair import migrate_windows_bin_path
        migrate_windows_bin_path(_m().PROJECT_ROOT)

    for message, step in (
        # v23 FTS layout is opt-in (existing indexes untouched); surface the command here.
        ('FTS optimize notice failed: %s', _print_fts_optimize_available_notice),
        ('Curator first-run notice failed: %s', _print_curator_first_run_notice),
        ('Curator recent-run notice failed: %s', _print_curator_recent_run_notice),
        ('FHS PATH guard check failed: %s', _ensure_fhs_path_guard),
        ('hermes-acp launcher self-heal failed: %s', _ensure_acp_launcher),
        ('Windows bin launcher migration failed: %s', _migrate_windows_bin_path),
        ('cua-driver refresh failed: %s', _refresh_cua_driver_after_update),
        ('Plugin compat notice failed: %s', _print_plugin_compat_notice),
    ):
        with _best_effort(message):
            step()


def _run_post_update_maintenance(
    *, assume_yes, gateway_mode, pre_update_snapshot_id, had_desktop_app_before_update, node_failures, desktop_build_ok,
    pre_update_version,
) -> bool:
    """Post-pull housekeeping: state.db restore, catalog/skills/profile syncs, config migration,
    the update summary (verdict returned), and best-effort notices/self-heals. Every step is
    isolated so none can fail the update."""
    from hermes_cli.update_cmd import _check_and_apply_config_migration, _m
    # macOS TCC: Desktop bundles are re-signed each update, so old grants can go stale
    # (toggle ON, yet macOS re-prompts with no Allow button). Tell users how to re-grant.
    # With the post-#73681 identifier-pinned DR, new grants survive rebuilds — but a grant made to a pre-fix
    # binary stays stale: the System Settings toggle shows ON while macOS re-prompts on every capture, and
    # the modern prompt has no Allow button, so users loop. One line of guidance after update tells affected
    # users how to complete the one-time re-grant.
    if sys.platform == "darwin" and had_desktop_app_before_update:
        print()
        print(
            "  ℹ macOS: if Hermes re-prompts for permissions you already "
            "granted (toggle shows ON), the stored grant is stale — run "
            "`tccutil reset ScreenCapture com.nousresearch.hermes` (repeat "
            "per affected service), toggle it ON in System Settings, then "
            "fully quit & relaunch once."
        )

    # macOS TCC interpreter anchor; boot-gated — a failed probe leaves the venv untouched.
    try:
        # See #95596.
        from hermes_cli.macos_tcc_anchor import ensure_tcc_anchor
        ensure_tcc_anchor()
    except Exception:
        logger.debug("macOS TCC anchor refresh skipped", exc_info=True)

    # state.db integrity guard for root home AND every profile; restore from own snapshot.
    with _best_effort('Post-update state.db integrity check failed: %s'):
        _verify_and_restore_state_dbs_post_update()

    # Seed the model-catalog cache from the checkout instead of a bot-gated, flaky fetch.
    with _best_effort('Model catalog seed during update failed: %s'):
        from hermes_cli.model_catalog import seed_cache_from_checkout
        if seed_cache_from_checkout(_m().PROJECT_ROOT):
            print("  ✓ Model catalog cache refreshed from checkout")

    with _best_effort('Skills sync during update failed: %s'):
        print()
        print("→ Syncing bundled skills...")
        _print_bundled_skills_sync_report()

    _sync_profiles_after_update()

    _check_and_apply_config_migration(
        assume_yes=assume_yes, gateway_mode=gateway_mode, pre_update_snapshot_id=pre_update_snapshot_id,
    )

    update_complete = _print_update_summary(
        node_failures=node_failures, desktop_build_ok=desktop_build_ok, pre_update_version=pre_update_version,
    )

    _print_post_update_notices_and_self_heals()
    return update_complete
