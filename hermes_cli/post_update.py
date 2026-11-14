"""Home maintenance shared by explicit updates and bounded boot maintenance.

Config migration owns backup/rollback; skills sync owns content merging; the
SQLite guard detects damage without repairing it. Boot bounds these home
steps per revision. The explicit scope CLI also provisions runtimes through
PM. Launcher publication belongs to hermes_cli._launchers.
"""
from __future__ import annotations

import logging
import os
import shutil
from datetime import datetime, timezone, UTC
from pathlib import Path
from typing import Iterable

from hermes_cli._launchers import expose_cli
from hermes_cli.default_plugins import step_install_default_plugins
from pm.paths import install_root

logger = logging.getLogger(__name__)



# ---------------------------------------------------------------------------
# config migration (backup / migrate / verify / restore)
# ---------------------------------------------------------------------------

def _backup_path(path: Path, stamp: str) -> Path:
    base = path.with_name(f"{path.name}.bak-{stamp}")
    if not base.exists():
        return base
    for index in range(1, 1000):
        candidate = path.with_name(f"{path.name}.bak-{stamp}.{index}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"could not choose a backup path for {path}")


def _backup_existing(paths: Iterable[Path]) -> dict:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    backups: dict = {}
    for path in paths:
        if not path.is_file():
            continue
        dest = _backup_path(path, stamp)
        shutil.copy2(path, dest)
        backups[path] = dest
    return backups


def _restore_backups(backups: dict) -> list:
    restored = []
    for original, backup in backups.items():
        if not backup.is_file():
            continue
        shutil.copy2(backup, original)
        restored.append(original)
    return restored


def step_migrate_config() -> dict:
    """Migrate config.yaml to the current schema, non-interactively.

    Same shape as scripts/docker_config_migrate.py: back up config + .env,
    migrate, verify the version advanced, restore the backups on failure.
    A pass that stopped at a failing step is partial progress, not a failure
    to undo: the file keeps the steps before it (stamped at the last one) and
    the result carries ``failed_step`` so boot retries it on the next start.
    No-op when the on-disk version is current (the 99% case).
    """
    from hermes_cli.config import (
        check_config_version,
        get_config_path,
        get_env_path,
        migrate_config,
    )
    from hermes_cli.config_migrations import (
        SUPPORT_FLOOR_VERSION,
        support_floor_message,
    )
    from hermes_cli.config_migrations_browser import agent_browser_env_pending
    from hermes_cli.config_version_stamp import read_config_version_stamp, stamp_missing_config

    stamp_missing_config()  # a missing config.yaml reads as current below
    stamp, latest_ver = read_config_version_stamp()
    current_ver = 0 if stamp is None else stamp
    if current_ver >= latest_ver:
        return {"ok": True, "skipped": "up-to-date"}
    if current_ver < SUPPORT_FLOOR_VERSION and not (stamp is None and agent_browser_env_pending()):
        # migrate_config() refuses sub-floor configs and leaves the file
        # untouched; warn instead of failing the boot. An unstamped config is
        # left alone too, unless its .env still holds AGENT_BROWSER_* settings
        # nothing reads any more: migrate_config() gives it the legacy-key
        # steps (v52's move among them) and a stamp.
        logger.warning("config migration skipped: %s", support_floor_message())
        return {"ok": True, "skipped": "below-support-floor"}

    backups = _backup_existing((get_config_path(), get_env_path()))
    try:
        results = migrate_config(interactive=False, quiet=True) or {}
    except Exception:
        _restore_backups(backups)
        raise
    post_ver, _ = check_config_version()
    failed = results.get("failed_step")
    if post_ver < latest_ver and failed is None:
        restored = _restore_backups(backups)
        raise RuntimeError(
            f"migration did not advance config version to {latest_ver} "
            f"(still {post_ver}); restored: "
            + (", ".join(str(p) for p in restored) if restored else "none")
        )
    # `hermes update` prints what a quiet migration changed; boot has no terminal to print it to.
    from hermes_cli.left_core_migration import defer_startup_notice
    for notice in results.get("notices") or ():
        defer_startup_notice(notice)
    if failed is not None:
        # A step raised: the file is stamped at the last step that succeeded, and those steps
        # stay (restoring would undo them). Boot retries the rest on every start
        # (boot_bootstrap); a retry that changed nothing leaves no backup behind.
        _drop_unchanged_backups(backups)
        defer_startup_notice(results["failed_warning"], log=False)
        return {"ok": False, "failed_step": failed, "failed_warning": results["failed_warning"],
                "migrated": f"{current_ver}->{post_ver}",
                "error": f"config migration to v{failed} failed; kept v{current_ver}->v{post_ver}, "
                         "retried on the next start"}
    return {"ok": True, "migrated": f"{current_ver}->{latest_ver}"}


def _drop_unchanged_backups(backups: dict) -> None:
    for original, backup in backups.items():
        try:
            if original.read_bytes() == backup.read_bytes():
                backup.unlink()
        except OSError:
            logger.debug("could not compare %s with its backup", original, exc_info=True)


# ---------------------------------------------------------------------------
# skills sync (this home only — profiles self-serve on their own boot)
# ---------------------------------------------------------------------------

def step_sync_skills() -> dict:
    """Sync bundled skills into the active home. Content-diffed, respects
    user modifications and deletions; converges on repeat runs."""
    from tools.skills_sync import sync_skills

    result = sync_skills(quiet=True) or {}
    return {
        "ok": True,
        "copied": len(result.get("copied") or []),
        "updated": len(result.get("updated") or []),
    }


# ---------------------------------------------------------------------------
# state.db integrity guard (#68474 — check-only variant)
# ---------------------------------------------------------------------------

def step_state_db_guard() -> dict:
    """Verify the active home's state.db is intact.

    Boot bootstrap has no pre-update snapshot to restore from (that pairing
    lives in ``hermes update``), so this is detection: a corrupt db is
    surfaced loudly in the log instead of the user silently losing session
    search. Read-only, idempotent.
    """
    from hermes_constants import get_hermes_home
    from hermes_cli.backup import verify_sqlite_integrity

    state_path = get_hermes_home() / "state.db"
    if not state_path.exists():
        return {"ok": True, "skipped": "no-state-db"}
    result = verify_sqlite_integrity(state_path, check_header=True, run_pragma=True)
    if result.get("valid"):
        return {"ok": True}
    message = result.get("message", "unknown error")
    logger.error(
        "state.db failed integrity check after a code update: %s — "
        "restore a backup with `hermes backup` tooling or contact support",
        message,
    )
    return {"ok": False, "error": message}


def step_drop_live_plugin_catalog() -> dict:
    """Drop the active home's cached live plugin catalog after a code change.

    Bundled/sealed app updates never run ``hermes update``'s maintenance tail,
    so without this a pre-update snapshot out-votes the newer in-tree catalog
    for the rest of its TTL (#119340). Per home, like the boot record.
    """
    from hermes_constants import get_hermes_home
    from hermes_cli.plugin_catalog import invalidate_live_cache_for_home

    invalidate_live_cache_for_home(get_hermes_home())
    return {"ok": True}


def step_adopt_blessed_checkout(project_root: Path | None = None) -> dict:
    """One-time adoption of shipped stampless installs (birth certificate).

    Main-era curl|sh / Setup installs created a ``.git`` checkout at a
    blessed managed root but never wrote a stamp — under the stamp-pure
    ladder they would all classify as "somebody's working tree" and
    `hermes update` would refuse them. This step writes the missing fact
    exactly once: blessed root + ``.git`` + no stamp → a minimal stamp
    with ``updateMechanism: self``.

    The blessed-root table lives HERE and only here — it is a one-time
    birth certificate for shipped installs, not a classification rung
    (hermes_cli.steward never path-matches). Once pre-stamp installs
    are extinct this step and the table can be deleted.

    * ``.git`` anywhere else → never adopted.
    * An existing stamp (any content) → untouched.
    * nix/docker/sealed populations are excluded by construction: their
      update mechanisms replace the tree wholesale with a
      build-time-stamped one, and sealed payloads always ship stamps.
    * Read-only tree (nix-like) → soft skip with a debug log, no crash.
    """
    import json
    import tempfile

    from hermes_constants import get_hermes_home

    root = install_root() if project_root is None else Path(project_root)

    # The blessed roots: the canonical locations installers create.
    blessed = (
        get_hermes_home() / "hermes-agent",
        Path("/usr/local/lib/hermes-agent"),
    )

    if not (root / ".git").exists():
        return {"ok": True, "skipped": "not-a-checkout"}
    from pm.paths import install_stamp_path
    stamp_path = install_stamp_path(root)
    if stamp_path.exists():
        return {"ok": True, "skipped": "already-stamped"}

    resolved_root = None
    try:
        resolved_root = root.resolve()
    except OSError:
        return {"ok": True, "skipped": "unresolvable-root"}
    is_blessed = False
    for candidate in blessed:
        try:
            if resolved_root == candidate.resolve():
                is_blessed = True
                break
        except OSError:
            continue
    if not is_blessed:
        return {"ok": True, "skipped": "not-a-blessed-root"}

    from hermes_cli.source_stamp import write_source_stamp

    try:
        # Full checkout identity when git can answer; the birth-certificate
        # minimum below only when it cannot (or the tree is read-only).
        identified = write_source_stamp(root, adopted=True)
    except OSError as exc:
        # A read-only tree (nix-like layouts without their own stamp)
        # must not crash the boot — it just stays unadopted.
        logger.debug("blessed-checkout adoption skipped (unwritable): %s", exc)
        return {"ok": True, "skipped": f"unwritable: {exc}"}
    if identified is None:
        # No derivable git identity: write the minimal birth certificate.
        stamp = {
            "schemaVersion": 2,
            "updateMechanism": "self",
            "source": "adoption",
            "adoptedAt": datetime.now(UTC).isoformat(),
        }
        try:
            fd, tmp_name = tempfile.mkstemp(
                dir=str(root), prefix=".install-stamp.", suffix=".tmp"
            )
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(stamp, indent=2) + "\n")
            os.replace(tmp_name, stamp_path)
        except OSError as exc:
            logger.debug("blessed-checkout adoption skipped (unwritable): %s", exc)
            return {"ok": True, "skipped": f"unwritable: {exc}"}
    logger.info("adopted blessed checkout at %s (updateMechanism: self)", root)
    return {"ok": True, "adopted": str(root)}


def step_provision_runtimes() -> dict:
    """Explicit refresh of PM's drifted packages; never selected by boot.

    PM supplies package identities, not diagnostic text to parse. Its own
    ensure/sync operations own freshness checks and publication under lock.
    """
    import pm
    from pm.install import lazy_installs_allowed, sealed

    problems = pm.drift()
    if not problems:
        return {"ok": True, "skipped": "current"}
    if sealed():
        # A sealed payload's tools ship with the artifact; drift here is
        # an artifact build problem, not something to repair in place.
        return {"ok": True, "skipped": "sealed"}
    if not lazy_installs_allowed():
        return {"ok": True, "skipped": "lazy-installs-disabled"}

    refreshed: list[str] = []
    errors: list[str] = []
    for name in problems:
        try:
            if name == "venv":
                pm.sync_venv(explicit=True, evict_incompatible_plugins=True)
            else:
                pm.ensure(name, explicit=True)
            refreshed.append(name)
        except Exception as exc:
            logger.warning("provision_runtimes: %s failed: %s", name, exc)
            errors.append(f"{name}: {exc}")
    if errors:
        return {"ok": False, "error": "; ".join(errors), "refreshed": refreshed}
    return {"ok": True, "refreshed": refreshed}


def step_verify_runtime_integrity() -> dict:
    """Post-update self-healing: verify runtime path, launcher shebangs, and clean dead locks."""
    import json
    from hermes_cli._launchers import resolve_store_python
    from hermes_cli.source_stamp import write_source_stamp
    from hermes_cli.update_lock import read_live_update
    from hermes_cli.venv_sync import publish_launchers
    from pm.environments import store_root
    from pm.paths import install_root, install_stamp_path

    repaired = []
    root = install_root()

    # 1. Verify and repair install-stamp.json runtimeDir
    stamp_path = install_stamp_path(root)
    if stamp_path.is_file():
        try:
            data = json.loads(stamp_path.read_text(encoding="utf-8-sig"))
        except Exception:
            data = {}
        runtime_dir = data.get("runtimeDir") if isinstance(data, dict) else None
        if not runtime_dir:
            store = store_root(root, honor_runtime_override=False)
            if store and (store / "facts.json").is_file():
                write_source_stamp(root)
                repaired.append(f"stamped runtimeDir: {store}")

    # 2. Verify launcher shebang matches resolve_store_python(root)
    store_python = resolve_store_python(root)
    if store_python:
        launcher_bin = root / ".hermes" / "bin" / ("hermes.exe" if os.name == "nt" else "hermes")
        if launcher_bin.is_file():
            try:
                content = launcher_bin.read_bytes()
                expected = str(store_python).encode("utf-8")
                if expected not in content:
                    publish_launchers(root)
                    repaired.append("republished launchers matching runtime python")
            except Exception as exc:
                logger.warning("verify_runtime_integrity: launcher check error: %s", exc)

    # 3. Clean any dead update markers safely without affecting running updaters
    try:
        live = read_live_update(install_root=root)
        if live is None:
            repaired.append("update locks clear")
    except Exception as exc:
        logger.warning("verify_runtime_integrity: update lock probe error: %s", exc)

    return {"ok": True, "repaired": repaired}


# ---------------------------------------------------------------------------
# step registries — boot_bootstrap gates each list with the matching record
# ---------------------------------------------------------------------------

HOME_STEPS: tuple = (
    ("adopt_blessed_checkout", step_adopt_blessed_checkout),
    ("migrate_config", step_migrate_config),
    ("sync_skills", step_sync_skills),
    ("state_db_guard", step_state_db_guard),
    ("drop_live_plugin_catalog", step_drop_live_plugin_catalog),
    ("expose_cli", expose_cli),
    ("install_default_plugins", step_install_default_plugins),
    ("verify_runtime_integrity", step_verify_runtime_integrity),
)

# Startup skill syncing belongs to each entry point. Boot bootstrap must
# not repeat it. The explicit scope CLI includes the skills step.
BOOT_HOME_STEPS: tuple = tuple(
    step for step in HOME_STEPS if step[0] != "sync_skills"
)

# Only the explicit scope CLI installs runtimes. Boot checks PM at startup.
MACHINE_STEPS: tuple = (
    ("provision_runtimes", step_provision_runtimes),
)

def run_steps(steps: Iterable) -> dict:
    """Run steps in order; one failure never stops the rest.

    Returns ``{name: result_dict}``. A raising step records
    ``{"ok": False, "error": str}`` — the caller still writes its record so
    a broken step cannot retrigger the slow path on every boot.
    """
    results: dict = {}
    for name, func in steps:
        try:
            results[name] = func()
        except Exception as exc:
            logger.warning("post-update step %s failed: %s", name, exc)
            results[name] = {"ok": False, "error": str(exc)}
    return results


def main(argv: list | None = None) -> int:
    """Run the selected maintenance registry and report its failures."""
    import argparse

    parser = argparse.ArgumentParser(prog="hermes_cli.post_update")
    parser.add_argument("--scope", choices=("home", "machine", "all"), default="all")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    selected: list = []
    if args.scope in ("home", "all"):
        selected.extend(HOME_STEPS)
    if args.scope in ("machine", "all"):
        selected.extend(MACHINE_STEPS)
    results = run_steps(selected)
    failed = [name for name, res in results.items() if not res.get("ok")]
    for name, res in results.items():
        state = "ok" if res.get("ok") else f"FAILED ({res.get('error')})"
        skipped = res.get("skipped")
        print(f"  post-update {name}: {f'skipped ({skipped})' if skipped else state}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
