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


def _pm_workspace_owner(root: Path, venv: Path) -> Path | None:
    """Resolve only this generation's workspace through PM's existing input stamp.

    The install key alone is not reversible. Successful PM syncs already persist
    the canonical source in ``inputs/.project-root``; validate it against the key
    before letting the updater execute that checkout. Missing or stale records
    leave the existing update admission path in charge.
    """
    generation = venv.parent
    if root != generation / "workspace" or generation.parent.name != "environments":
        return None
    state = generation.parent.parent
    try:
        from pm.environments import install_key

        recorded = (state / "inputs" / ".project-root").read_text(encoding="utf-8")
        owner = Path(recorded)
        if not recorded or not owner.is_absolute():
            return None
        owner = owner.resolve()
        if (install_key(owner) != state.name or not (owner / ".git").exists()
                or not (owner / "hermes_cli" / "main.py").is_file()):
            return None
        return owner
    except (OSError, ValueError):
        return None


def owning_install_root(project_root: Path) -> Path | None:
    """The checkout owning this interpreter, including a recorded PM generation.

    ``None`` when this process runs on its own checkout's interpreter, a venv outside any
    checkout, or when *project_root* was put on ``PYTHONPATH`` on purpose (a wrapper that
    runs a dev tree on another interpreter chose that tree; it is not a redirected install).
    """
    if sys.prefix == sys.base_prefix:
        return None
    venv = Path(sys.prefix).resolve()
    owner = venv.parent
    root = Path(project_root).resolve()
    if venv.name not in ("venv", ".venv") or owner == root:
        return None
    chosen = (Path(p).resolve() for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p)
    if root in chosen:
        return None
    pm_owner = _pm_workspace_owner(root, venv)
    if pm_owner is not None:
        return pm_owner
    if not (owner / "hermes_cli" / "main.py").is_file():
        return None
    return owner


def retarget_to_owning_install(project_root: Path) -> None:
    """Re-run this command with the owning install's code; returns only when nothing is redirected."""
    owner = owning_install_root(project_root)
    if owner is None:
        return
    print(f"⚠ Interpreter environment {sys.prefix} is running {project_root}, not its owning checkout.")
    print(f"→ Updating {owner} with its own updater.")
    sys.stdout.flush()
    # PYTHONPATH entries resolve before the editable finder on sys.meta_path, so the
    # owner's hermes_cli wins; in the child, project_root == owner and this is a no-op.
    env = dict(os.environ, PYTHONPATH=str(owner))
    code = subprocess.call([sys.executable, "-m", "hermes_cli.main", *sys.argv[1:]], cwd=owner, env=env)
    raise SystemExit(code)
