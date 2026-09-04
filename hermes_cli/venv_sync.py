"""Sync the install's Python environment to its tree — and nothing else.

One of the two separately-invocable, stdlib-only-at-import halves of
"make this install runnable":

* ``pm`` — pinned tools and the venv ledger (facts.json), shape-universal.
* THIS module — the pre-venv entry point, whose meaning depends on the
  install shape:

  - **checkout** (git clone + venv): make the venv match
    ``pyproject.toml``/``uv.lock``. For the tree this module runs FROM,
    the work is delegated to :func:`pm.sync_venv` — pm owns the extras
    ledger and records the result in facts.json, so update and fresh
    clone converge on one authority. For a foreign ``--project-root``
    (installer bootstrapping a clone it has not exec'd into yet), uv is
    driven directly with pm's pinned binary and sanitized env.
  - **sealed** (desktop bundle, nix, docker): the interpreter tree is a
    build artifact — syncing it is not possible and not meaningful.
    Exit 0 with ``{"state": "sealed"}``, cleanly.

Stdlib-only at import is a hard contract: this runs on freshly-cloned
trees where the venv does not exist yet, and after tree swaps where the
venv is not trustworthy — exactly the moments a third-party import
would explode. ``pm`` is itself stdlib-only first-party code, and it is
imported lazily, only once a sync is actually due.

Why the sync is not just "run uv every time": ``uv sync`` on an
already-current venv still costs ~1-2s of resolver work, and the boot
path calls this through ``post_update``. The lockfile digest recorded in
``<runtime dir>/cache/venv-sync.json`` makes currency a file read.

Invocation:

    python -m hermes_cli.venv_sync                # sync if stale
    python -m hermes_cli.venv_sync --check        # report, change nothing
    python -m hermes_cli.venv_sync --json         # machine-readable

Exit 0: current/synced/sealed. Exit 1: a sync was needed and failed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

STAMP_NAME = "venv-sync.json"
STAMP_SCHEMA = 1


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _runtime_dir(project_root: Path) -> Path:
    """Where per-install runtime state lives for ``project_root``.

    Mirrors pm.paths.store_root()'s ladder (HERMES_RUNTIME_DIR, then the
    install stamp's runtimeDir) but stays stdlib-local and falls back to
    ``<root>/.hermes-runtime`` instead of the machine-global tool store:
    the venv-sync stamp is per-install, and two checkouts sharing the
    default store must not share a currency claim.
    """
    env = os.environ.get("HERMES_RUNTIME_DIR")
    if env:
        return Path(env)
    try:
        data = json.loads(
            (project_root / "install-stamp.json").read_text(encoding="utf-8-sig")
        )
        stamped = data.get("runtimeDir") if isinstance(data, dict) else None
        if stamped:
            return Path(stamped)
    except (OSError, ValueError):
        pass
    return project_root / ".hermes-runtime"


def _stamp_path(project_root: Path) -> Path:
    return _runtime_dir(project_root) / "cache" / STAMP_NAME


def _lock_digest(project_root: Path) -> str | None:
    """Content hash of what a sync would consume.

    pyproject.toml is part of the key: an extras edit without a lock
    bump must re-sync (same reasoning as pm's venv expected_stamp keying
    on uv.lock — pyproject rides along here because this module cannot
    assume pm's ledger exists yet).
    """
    h = hashlib.sha256()
    found = False
    for name in ("uv.lock", "pyproject.toml"):
        try:
            h.update((project_root / name).read_bytes())
            found = True
        except OSError:
            h.update(b"-")
    return h.hexdigest() if found else None


def _is_sealed(project_root: Path) -> bool:
    """A sealed tree ships its interpreter; only checkouts own a venv.

    The stamp file is the authority (hermes_cli.steward reads the same
    file; restated here to keep the bare import stdlib-and-local). A
    tree with BOTH a stamp and .git is a dev tree — treat as checkout.

    A stamp without a valid ``updateMechanism`` is a build-lane bug and
    must not be silently read as "not sealed" (that is exactly the
    misclassification that made sealed trees look updatable) — same
    guard as hermes_cli.version_info._stamp_version_info.
    """
    if (project_root / ".git").exists():
        return False
    try:
        data = json.loads(
            (project_root / "install-stamp.json").read_text(encoding="utf-8-sig")
        )
    except (OSError, ValueError):
        return False
    if not (isinstance(data, dict) and bool(data)):
        return False
    if data.get("updateMechanism") not in ("self", "electron-updater", "external"):
        raise RuntimeError(
            f"install-stamp.json at {project_root} is missing a valid "
            "'updateMechanism' (one of self, electron-updater, external). The "
            "build lane that wrote this stamp must pass --update-mechanism to "
            "scripts/write_install_stamp.py."
        )
    return True


def _managed_uv() -> tuple:
    """(uv binary path, sanitized env) from pm, or (None, None).

    ``pm.ensure.uv`` IS the resolution: facts from the store the pins
    point at, env sanitized against interpreter hijack (#83914). pm is
    stdlib-only first-party code, so the lazy import costs nothing and
    keeps this module's bare-import surface stdlib-pure.
    """
    try:
        from pm.ensure import uv as pm_uv

        uv_bin, env = pm_uv(realize=True)
    except Exception:
        return None, None
    return uv_bin, env


def read_stamp(project_root: Path) -> dict:
    try:
        data = json.loads(_stamp_path(project_root).read_text(encoding="utf-8-sig"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def write_stamp(project_root: Path, digest: str) -> None:
    path = _stamp_path(project_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(
            {
                "schemaVersion": STAMP_SCHEMA,
                "lockDigest": digest,
                "python": sys.version.split()[0],
            }
        ),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _pm_sync(root: Path) -> dict:
    """Bring ``root``'s venv current. pm's ledger when root IS this tree,
    a direct pinned-uv drive when it is a foreign clone.

    Returns ``{"ok": bool, "detail": str | None}``.
    """
    try:
        from pm import paths as pm_paths

        own_tree = Path(pm_paths.repo_root()).resolve() == root.resolve()
    except Exception:
        own_tree = False

    if own_tree:
        # pm owns the extras ledger; explicit=True because reaching a
        # stale-venv verdict here IS the deliberate remedy (installer /
        # post-update path), the same trust `hermes update` carries.
        try:
            from pm import sync_venv

            sync_venv(explicit=True)
        except Exception as exc:
            return {"ok": False, "detail": str(exc)}
        return {"ok": True, "detail": None}

    uv, env = _managed_uv()
    if uv is None:
        return {
            "ok": False,
            "detail": "managed uv not provisioned (run `hermes pm install`)",
        }
    env = dict(env)
    env["UV_PROJECT_ENVIRONMENT"] = str(root / "venv")
    # A stale VIRTUAL_ENV from the calling shell would win over the
    # project environment and sync the WRONG venv.
    env.pop("VIRTUAL_ENV", None)
    cmd = [str(uv), "sync"]
    if (root / "uv.lock").is_file():
        cmd.append("--frozen")
    proc = subprocess.run(cmd, cwd=str(root), env=env)
    if proc.returncode != 0:
        return {"ok": False, "detail": f"uv sync exited {proc.returncode}"}
    return {"ok": True, "detail": None}


def sync(project_root: Path | None = None, *, check: bool = False) -> dict:
    """Bring the venv up to the tree. Returns a state dict, never raises.

    States: ``sealed`` (nothing to sync, by design), ``current`` (stamp
    matches the lockfile digest), ``synced`` (the sync ran and the stamp
    moved), ``failed`` (the sync did not converge — detail says why),
    ``would-sync`` (check mode found staleness and stopped).
    """
    root = Path(project_root) if project_root else _project_root()

    if _is_sealed(root):
        return {"state": "sealed", "ok": True}

    digest = _lock_digest(root)
    if digest is None:
        return {
            "state": "failed",
            "ok": False,
            "detail": f"no pyproject.toml or uv.lock under {root}",
        }

    if read_stamp(root).get("lockDigest") == digest:
        return {"state": "current", "ok": True}

    if check:
        return {"state": "would-sync", "ok": True}

    result = _pm_sync(root)
    if not result["ok"]:
        # No stamp write: the next run must try again, not skip.
        return {"state": "failed", "ok": False, "detail": result["detail"]}

    write_stamp(root, digest)
    return {"state": "synced", "ok": True}


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hermes_cli.venv_sync")
    parser.add_argument("--project-root", default=None)
    parser.add_argument(
        "--check", action="store_true", help="report; change nothing"
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    result = sync(
        Path(args.project_root) if args.project_root else None, check=args.check
    )

    if args.json:
        print(json.dumps(result))
    else:
        detail = f" ({result['detail']})" if result.get("detail") else ""
        print(f"venv sync: {result['state']}{detail}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
