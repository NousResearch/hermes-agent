"""Pre-update backup policy and snapshot orchestration.

Extracted mechanically from :mod:`hermes_cli.update_cmd`.  Runtime
references to the historical module surface resolve through the
compatibility facade so imports and monkeypatches remain effective.
"""

from typing import Optional
from hermes_cli._update_compat import facade as _u


_PRE_UPDATE_SNAPSHOT_KEEP = 1


_PRE_UPDATE_SNAPSHOT_MAX_FILE_SIZE = 1 << 30  # 1 GiB


def _resolve_pre_update_backup_mode(args) -> str:
    """Resolve the pre-update backup mode: ``"off"``, ``"quick"``, or ``"full"``.

    CLI flags win over config; ``--no-backup`` beats ``--backup`` when both
    are set. Config accepts the mode strings plus legacy booleans:
    ``true`` → ``full`` (the old zip behavior), ``false`` → ``off``
    (an explicit opt-out now disables the quick snapshot too — previously
    it ran unconditionally, ignoring the user's setting). A missing key
    defaults to ``quick``.
    """
    if getattr(args, "no_backup", False):
        return "off"
    if getattr(args, "backup", False):
        return "full"

    try:
        from hermes_cli.config import load_config

        cfg = load_config()
    except Exception as exc:
        _u().logger.debug(
            "Could not load config for pre-update backup: %s", exc
        )
        cfg = {}

    updates_cfg = cfg.get("updates", {}) if isinstance(cfg, dict) else {}
    raw = updates_cfg.get("pre_update_backup", "quick")

    if raw is True:
        return "full"
    if raw is False:
        return "off"
    mode = str(raw).strip().lower()
    if mode in ("off", "false", "none", "disabled"):
        return "off"
    if mode in ("full", "zip", "true"):
        return "full"
    if mode == "quick":
        return "quick"
    _u().logger.warning(
        "Unknown updates.pre_update_backup value %r — using 'quick'", raw
    )
    return "quick"


def _run_pre_update_backup(args) -> Optional[str]:
    """Run the pre-update safety backup and return the quick-snapshot id.

    Single consolidated mechanism gated on ``updates.pre_update_backup``:

    - ``off``   — nothing runs. Explicit user opt-out is honored fully.
    - ``quick`` (default) — a state snapshot of critical small files
      (pairing JSONs, cron jobs, config, auth; see ``_QUICK_STATE_FILES``)
      under ``state-snapshots/``. Files over 1 GiB are skipped with a
      warning so a bloated state.db can never stall the update
      (issues #15733, #34600 are the reason this safety net exists).
    - ``full``  — the quick snapshot PLUS a full zip of HERMES_HOME under
      ``backups/`` (restorable via ``hermes import``; the #48200 wrong-path
      wipe is the reason this level exists).

    ``--backup`` forces ``full`` for one run; ``--no-backup`` forces ``off``.
    Never raises — a backup failure should not block the update itself.

    Returns the quick-snapshot id (used by the post-update cron-jobs
    restore safety net), or ``None`` when mode is ``off`` or the snapshot
    failed.
    """
    mode = _resolve_pre_update_backup_mode(args)

    if mode == "off":
        if getattr(args, "no_backup", False):
            print("◆ Pre-update backup: skipped (--no-backup)")
            print()
        # Config-level off is silent — the user opted out; don't spam them
        # on every update.
        return None

    snapshot_id = None
    try:
        from hermes_cli.backup import (
            _quick_snapshot_root,
            create_quick_snapshot,
            verify_sqlite_integrity,
        )

        # NOTE: this function later does `from hermes_constants import
        # get_hermes_home`, which makes the name function-local — the
        # module-level import is shadowed and unbound here. Alias explicitly.
        from hermes_cli.config import get_hermes_home as _get_home

        snapshot_id = create_quick_snapshot(
            label="pre-update",
            keep=_PRE_UPDATE_SNAPSHOT_KEEP,
            max_file_size=_PRE_UPDATE_SNAPSHOT_MAX_FILE_SIZE,
        )

        # After the snapshot, verify the source state.db is still intact.
        # The snapshot was taken via _safe_copy_db (read-only SQLite backup
        # API), but a concurrent process (antivirus, force-killed gateway
        # releasing file handles, Windows filter driver) can corrupt the live
        # file at any point. A silent zeroing at this point would proceed with
        # the update and exit code 0 — exactly the #68474 symptom.
        if snapshot_id:
            _src_path = _get_home() / "state.db"
            if _src_path.exists():
                _integrity = verify_sqlite_integrity(
                    _src_path,
                    check_header=True,
                    run_pragma=True,
                    max_bytes=_PRE_UPDATE_SNAPSHOT_MAX_FILE_SIZE,
                )
                if not _integrity.get("valid"):
                    _msg = _integrity.get("message", "unknown error")
                    print(
                        f"  ⚠ state.db integrity check FAILED after snapshot: {_msg}"
                    )
                    # Check if the snapshot itself is valid.
                    _snap_root = _quick_snapshot_root(_get_home())
                    _snap_state = _snap_root / snapshot_id / "state.db"
                    if _snap_state.exists():
                        _snap_ok = verify_sqlite_integrity(
                            _snap_state, check_header=True, run_pragma=True
                        )
                        if _snap_ok.get("valid"):
                            print(
                                "  ✓ Snapshot copy is valid — continuing update."
                            )
                            print(
                                "    If state.db is lost after update it will be auto-restored."
                            )
                        else:
                            print(
                                "  ✗ Snapshot copy ALSO failed integrity — "
                                "the source was already corrupted before the backup."
                            )
                    else:
                        print(
                            "  ⚠ Snapshot does not contain state.db (was skipped or too large)."
                        )
                    print()
        if snapshot_id:
            print(f"◆ Pre-update snapshot: {snapshot_id}")

        # #66140: the code swap + fleet restart touch EVERY profile, so
        # every profile gets the same snapshot (same set, same 1GiB cap,
        # keep=1) under its own state-snapshots/. Best-effort per profile.
        try:
            from hermes_cli.backup import create_pre_update_snapshots_all_profiles

            _sibling_snaps = create_pre_update_snapshots_all_profiles(
                keep=_PRE_UPDATE_SNAPSHOT_KEEP,
                max_file_size=_PRE_UPDATE_SNAPSHOT_MAX_FILE_SIZE,
            )
            if _sibling_snaps:
                print(
                    f"◆ Sibling profile snapshot(s): "
                    + ", ".join(sorted(_sibling_snaps))
                )
                try:
                    from hermes_cli.update_receipt import record_step

                    record_step(
                        "sibling_profile_snapshots",
                        True,
                        ", ".join(
                            f"{k}={v}" for k, v in sorted(_sibling_snaps.items())
                        ),
                    )
                except Exception:
                    pass
                _u()._LAST_SIBLING_SNAPSHOTS = _sibling_snaps
        except Exception as _sib_exc:
            _u().logger.debug(
                "Sibling profile snapshots failed: %s", _sib_exc
            )
    except Exception as exc:
        # Never let a snapshot failure block an update.
        _u().logger.debug("Pre-update snapshot failed: %s", exc)

    if mode != "full":
        if snapshot_id:
            print()
        return snapshot_id

    try:
        from hermes_cli.backup import create_pre_update_backup
    except Exception as exc:
        print(
            f"⚠ Pre-update backup: could not load backup module ({exc}); continuing update."
        )
        print()
        return snapshot_id

    try:
        from hermes_cli.config import load_config

        _keep = (load_config() or {}).get("updates", {}).get("backup_keep", 5)
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
        return snapshot_id

    elapsed = _time.monotonic() - t0

    if out_path is None:
        print("  ⚠ Backup skipped (no files found or write failed); continuing update.")
        print()
        return snapshot_id

    try:
        size_bytes = out_path.stat().st_size
    except OSError:
        size_bytes = 0

    # Human-readable size
    from hermes_cli.sizefmt import format_bytes

    size_str = format_bytes(size_bytes)

    # Render path using display_hermes_home so the user sees ~/.hermes/...
    try:
        from hermes_constants import get_hermes_home, display_hermes_home

        home = get_hermes_home()
        try:
            display_path = f"{display_hermes_home()}/{out_path.relative_to(home)}"
        except ValueError:
            display_path = str(out_path)
    except Exception:
        display_path = str(out_path)

    print(f"  Saved:    {display_path} ({size_str}, {elapsed:.1f}s)")
    print(f"  Restore:  hermes import {out_path}")
    print("  Disable:  set updates.pre_update_backup: quick (or off) in config.yaml")
    print()
    return snapshot_id
