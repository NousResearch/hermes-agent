"""Send ``hermes update`` back to the install whose interpreter it is running on.

An install's in-tree venv can end up importing another checkout's code: an editable
install recorded against a dev tree (what ``project_venv_dir`` used to cause when a dev
checkout ran on the app install's interpreter) turns ``<install>/venv/bin/hermes`` into
the dev tree's CLI. Every update the Desktop hands to that launcher then pulls the dev
tree, the install never moves, and the app keeps relaunching its stale build.

Running the install's own updater repairs it: it pulls the install and reinstalls the
install into its venv, which rewrites the editable record to point home again.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


def _lease_managed_owner(venv: Path, root: Path) -> Path | None:
    """Return a source checkout recorded for a generated PM environment.

    A PM generation is a dependency snapshot, not an update target.  Its
    ``inputs/.project-root`` record names the source install that created it.
    """
    generation = venv.parent
    if not (generation / ".lease-managed").is_file() or root != generation / "workspace":
        return None
    environments = generation.parent
    if environments.name != "environments":
        return None
    try:
        owner = Path((environments.parent / "inputs" / ".project-root").read_text(
            encoding="utf-8-sig"
        ).strip()).resolve()
    except (OSError, ValueError):
        return None
    if owner == root or not (owner / "hermes_cli" / "main.py").is_file():
        return None
    git = owner / ".git"
    try:
        if not (git.is_dir() or git.read_text(encoding="utf-8-sig").startswith("gitdir:")):
            return None
    except OSError:
        return None
    return owner


def owning_install_root(project_root: Path) -> Path | None:
    """The source checkout that owns this interpreter when it differs from *project_root*.

    Covers both conventional in-tree virtualenvs and PM's lease-managed
    workspace snapshots.  A snapshot has no Git metadata and must never be
    treated as an in-place update target.
    """
    if sys.prefix == sys.base_prefix:
        return None
    venv = Path(sys.prefix).resolve()
    root = Path(project_root).resolve()
    if venv.name not in ("venv", ".venv"):
        return None
    lease_owner = _lease_managed_owner(venv, root)
    if lease_owner is not None:
        return lease_owner
    owner = venv.parent
    if owner == root or not (owner / "hermes_cli" / "main.py").is_file():
        return None
    chosen = (Path(p).resolve() for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p)
    if root in chosen:
        return None
    return owner


def retarget_to_owning_install(project_root: Path) -> None:
    """Re-run this command with the owning install's code; returns only when nothing is redirected."""
    owner = owning_install_root(project_root)
    if owner is None:
        return
    print(f"⚠ {owner / Path(sys.prefix).name} is running the checkout at {project_root}, not its own code.")
    print(f"→ Updating {owner} with its own updater; this also points its venv back at it.")
    sys.stdout.flush()
    # PYTHONPATH entries resolve before the editable finder on sys.meta_path, so the
    # owner's hermes_cli wins; in the child, project_root == owner and this is a no-op.
    env = dict(os.environ, PYTHONPATH=str(owner))
    code = subprocess.call([sys.executable, "-m", "hermes_cli.main", *sys.argv[1:]], cwd=owner, env=env)
    raise SystemExit(code)
