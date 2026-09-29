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
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from hermes_cli._launchers import expose_cli
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
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
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

    current_ver, latest_ver = check_config_version()
    if current_ver >= latest_ver:
        return {"ok": True, "skipped": "up-to-date"}
    if current_ver < SUPPORT_FLOOR_VERSION:
        # migrate_config() refuses sub-floor configs and leaves the file
        # untouched; warn instead of failing the boot.
        logger.warning("config migration skipped: %s", support_floor_message())
        return {"ok": True, "skipped": "below-support-floor"}

    backups = _backup_existing((get_config_path(), get_env_path()))
    try:
        migrate_config(interactive=False, quiet=True)
    except Exception:
        _restore_backups(backups)
        raise
    post_ver, _ = check_config_version()
    if post_ver < latest_ver:
        restored = _restore_backups(backups)
        raise RuntimeError(
            f"migration did not advance config version to {latest_ver} "
            f"(still {post_ver}); restored: "
            + (", ".join(str(p) for p in restored) if restored else "none")
        )
    return {"ok": True, "migrated": f"{current_ver}->{latest_ver}"}


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
        identified = write_source_stamp(root)
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
            "adoptedAt": datetime.now(timezone.utc).isoformat(),
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
        except Exception as exc:  # noqa: BLE001 — one tool must not stop the rest
            logger.warning("provision_runtimes: %s failed: %s", name, exc)
            errors.append(f"{name}: {exc}")
    if errors:
        return {"ok": False, "error": "; ".join(errors), "refreshed": refreshed}
    return {"ok": True, "refreshed": refreshed}


# ---------------------------------------------------------------------------
# spawn guard (t_7b0df4cf): the 3 spawn paths rise with their dependencies
# ---------------------------------------------------------------------------

SPAWN_GUARD_LEG_TIMEOUT_S = 120
SPAWN_GUARD_MUTATION_ENV = "GUARDA_3_SPAWNS_MUTATION"


def _spawn_guard_generation_fingerprint(root: Path) -> str:
    """sha256 of the committed dependency selection (``facts.json``).

    Empty when no selection is committed (dev checkout before first sync):
    there is no generation to verify, so the step skips instead of failing.
    """
    import hashlib

    from pm.environments import runtime_facts_path

    try:
        raw = runtime_facts_path(Path(root)).read_bytes()
    except OSError:
        return ""
    return hashlib.sha256(raw).hexdigest()


def _spawn_guard_state_path(root: Path) -> Path:
    """Where this step remembers the last verified generation (per profile)."""
    from hermes_cli.profiles import get_active_profile_name
    from pm.environments import install_state_dir

    name = get_active_profile_name() or "default"
    return install_state_dir(Path(root)) / "spawn-guard" / f"{name}.json"


def _spawn_guard_pin(env: dict, root: Path) -> dict:
    """The production pin, plus the ``root-only`` mutation (must fail L3)."""
    import os as _os

    from cron.scheduler_worker_env import pin_hermes_tree_on_pythonpath

    if _os.environ.get(SPAWN_GUARD_MUTATION_ENV) == "root-only":
        # Rebuild the pre-fix pin (root only, no generation site): the
        # mutation the guard must catch on the cron leg.
        cur = [e for e in env.get("PYTHONPATH", "").split(_os.pathsep) if e]
        env["PYTHONPATH"] = _os.pathsep.join(
            [str(root), *[c for c in cur if c != str(root)]]
        )
        return env
    return pin_hermes_tree_on_pythonpath(env, Path(root))


def _spawn_guard_run(argv: list, env: dict, root: Path):
    import subprocess as _subprocess

    return _subprocess.run(
        argv, cwd=str(root), env=env, stdin=_subprocess.DEVNULL,
        stdout=_subprocess.PIPE, stderr=_subprocess.PIPE, text=True,
        timeout=SPAWN_GUARD_LEG_TIMEOUT_S, start_new_session=True,
    )


def _spawn_guard_leg_kanban(root: Path) -> tuple:
    import sys as _sys

    from tools.environments.local import build_subprocess_env

    env = build_subprocess_env(scrub_secrets=False, inherit_profile_home=True)
    env = _spawn_guard_pin(env, root)
    argv = [_sys.executable, "-m", "hermes_cli.main", "--help"]
    proc = _spawn_guard_run(argv, env, root)
    err = proc.stderr or ""
    if proc.returncode == 0 and "ModuleNotFoundError" not in err:
        return True, "rc=0 --help ok"
    tail = "\n".join(err.strip().splitlines()[-5:])
    return False, f"rc={proc.returncode} stderr: {tail}"


def _spawn_guard_leg_dm(root: Path) -> tuple:
    import os as _os
    import sys as _sys

    from tools.environments.local import served_profile_child_env

    env = served_profile_child_env(
        base=_os.environ, target_home=None, inherit_credentials=True
    )
    env.pop("HERMES_TURN_AUTHOR", None)
    argv = [_sys.executable, "-m", "hermes_cli.main", "--help"]
    proc = _spawn_guard_run(argv, env, root)
    err = proc.stderr or ""
    if proc.returncode == 0 and "ModuleNotFoundError" not in err:
        return True, "rc=0 --help ok"
    tail = "\n".join(err.strip().splitlines()[-5:])
    return False, f"rc={proc.returncode} stderr: {tail}"


def _spawn_guard_leg_cron(root: Path) -> tuple:
    import sys as _sys
    import tempfile as _tempfile

    from tools.environments.local import (
        build_subprocess_env,
        strip_launch_profile_env,
    )

    env = strip_launch_profile_env(
        build_subprocess_env(
            scrub_secrets=False, inherit_profile_home=True,
            extra={"HERMES_HOME": str(Path.home() / ".hermes")},
        )
    )
    env = _spawn_guard_pin(env, root)
    missing = Path(_tempfile.gettempdir()) / "spawn-guard-sem-payload.json"
    try:
        missing.unlink()
    except OSError:
        pass
    ack = Path(_tempfile.gettempdir()) / "spawn-guard-ack.ready"
    argv = [_sys.executable, "-m", "cron.scheduler", "--external-worker-file",
            str(missing), "--ack-file", str(ack)]
    proc = _spawn_guard_run(argv, env, root)
    err = proc.stderr or ""
    if "ModuleNotFoundError" in err or "No module named" in err:
        tail = "\n".join(err.strip().splitlines()[-5:])
        return False, f"IMPORT-FAIL stderr: {tail}"
    # Every pre-import failure prints a traceback on stderr; the post-import
    # path (missing payload -> ``_run_external_worker_payload`` returns False
    # -> SystemExit(1)) logs to a file and exits 1 silently. So rc==1 with no
    # traceback PROVES the import graph loaded whole.
    if proc.returncode == 1 and "Traceback" not in err:
        return True, ("imports ok (missing payload rejected AFTER imports, "
                      "expected exit 1; no stderr traceback)")
    tail = "\n".join(err.strip().splitlines()[-5:])
    return False, f"unexpected signature rc={proc.returncode} stderr: {tail}"


def step_spawn_guard(project_root: Path | None = None) -> dict:
    """Verify the 3 spawn paths rise with their dependencies (t_7b0df4cf).

    Each leg builds its env with the SAME builders production uses and spawns
    the real argv in a subprocess — never an in-process import of
    ``cron``/``hermes_cli`` (importing in-process does not exercise the real
    path). Legs: kanban worker (``--help``), DM channel (``--help``), cron
    external worker (missing payload: PASS = exit 1 with NO stderr
    traceback; any ``ModuleNotFoundError``/``No module named``/traceback
    fails naming the leg).

    Generation-gated, NOT revision-gated: boot keys its record by installed
    revision, but the dependency generation (``facts.json``) can change
    without a revision change (PM sync), and re-running every boot step on a
    generation change would redo the expensive ones (SQLite guard). So this
    step owns its own per-profile ``spawn-guard/`` record and skips when the
    generation fingerprint is unchanged. Recorded skips (never silent):
    wheel installs (pin is a no-op there), trees with no committed site to
    verify, and unchanged generations.
    """
    import json as _json
    import os as _os
    import time as _time

    from pm.paths import install_root

    root = install_root() if project_root is None else Path(project_root)

    # The skip decision uses the REAL pin (never the mutation): it answers
    # "is there a committed site to verify", not "does the pin work".
    from cron.scheduler_worker_env import pin_hermes_tree_on_pythonpath

    probe = pin_hermes_tree_on_pythonpath({}, root)
    if "PYTHONPATH" not in probe:
        return {"ok": True, "skipped": "wheel-install"}
    if probe["PYTHONPATH"] == str(root):
        return {"ok": True, "skipped": "no-committed-site"}

    fingerprint = _spawn_guard_generation_fingerprint(root)
    state_path = _spawn_guard_state_path(root)
    if fingerprint:
        try:
            known = _json.loads(state_path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            known = {}
        if isinstance(known, dict) and known.get("generation") == fingerprint:
            return {"ok": True, "skipped": "generation-unchanged"}

    legs = (("L1-kanban-worker", _spawn_guard_leg_kanban),
            ("L2-canal-dm", _spawn_guard_leg_dm),
            ("L3-cron-externo", _spawn_guard_leg_cron))
    failed: dict = {}
    for name, func in legs:
        try:
            ok, sig = func(root)
        except Exception as exc:  # noqa: BLE001 — the guard reports, never raises
            ok, sig = False, f"HARNESS-ERROR {type(exc).__name__}: {exc}"
        if not ok:
            failed[name] = sig
    if _os.environ.get(SPAWN_GUARD_MUTATION_ENV):
        logger.warning("spawn_guard ran with %s=%s",
                       SPAWN_GUARD_MUTATION_ENV,
                       _os.environ.get(SPAWN_GUARD_MUTATION_ENV))
    if failed:
        return {"ok": False, "error": "; ".join(f"{n}: {s}" for n, s in failed.items())}
    if fingerprint:
        try:
            state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = state_path.with_name(state_path.name + ".tmp")
            tmp.write_text(_json.dumps({
                "generation": fingerprint,
                "verifiedAt": _time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }, indent=2) + "\n", encoding="utf-8")
            _os.replace(tmp, state_path)
        except OSError as exc:
            logger.warning("spawn_guard could not record generation: %s", exc)
    return {"ok": True, "legs": "3/3"}


# ---------------------------------------------------------------------------
# step registries — boot_bootstrap gates each list with the matching record
# ---------------------------------------------------------------------------

HOME_STEPS: tuple = (
    ("adopt_blessed_checkout", step_adopt_blessed_checkout),
    ("migrate_config", step_migrate_config),
    ("sync_skills", step_sync_skills),
    ("state_db_guard", step_state_db_guard),
    ("drop_live_plugin_catalog", step_drop_live_plugin_catalog),
    ("spawn_guard", step_spawn_guard),
    ("expose_cli", expose_cli),
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
