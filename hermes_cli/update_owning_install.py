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


def owning_install_root(project_root: Path) -> Path | None:
    """The checkout whose in-tree venv this interpreter is, when that is not *project_root*.

    ``None`` when this process runs on its own checkout's interpreter, a venv outside any
    checkout, or when *project_root* was put on ``PYTHONPATH`` on purpose (a wrapper that
    runs a dev tree on another interpreter chose that tree; it is not a redirected install).
    """
    if sys.prefix == sys.base_prefix:
        return None
    venv = Path(sys.prefix).resolve()
    root = Path(project_root).resolve()
    if venv.name not in ("venv", ".venv"):
        return None
    # Under pytest the interpreter legitimately runs from the checkout's venv while the
    # code under test operates on a fixture clone in tmp: that is isolation, not a
    # mis-pointed install, and "retargeting" re-ran the whole test invocation as a
    # `hermes` command (SystemExit 2 on every machine with a real install —
    # 2026-10-08 field incident). The marker is the conftest-exported one from
    # hermes_state_guard, chosen because children inherit it by default.
    if (os.environ.get("PYTEST_CURRENT_TEST") or os.environ.get("PYTEST_VERSION")
            or os.environ.get("HERMES_TEST_ISOLATION")):
        return None
    owner = venv.parent if (venv.parent / "hermes_cli" / "main.py").is_file() else _pm_environment_owner(venv)
    if owner is None or owner == root:
        return None
    chosen = (Path(p).resolve() for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p)
    if root in chosen:
        return None
    return owner


def _pm_environment_owner(venv: Path) -> Path | None:
    """The checkout a PM generation venv (``<installs>/<key>/environments/<gen>/venv``) belongs to.

    Its ``hermes_cli`` is the generation's ``workspace/`` snapshot, which has no ``.git``, so an
    update run from it must go to the checkout (#122627). The install key hashes the checkout
    path, which proves the path recorded beside it.
    """
    from pm.environments import install_key

    state = venv.parent.parent.parent
    if venv.parent.parent.name != "environments":
        return None
    try:
        # pm.environments.record_activation_inputs writes it on every full install.
        owner = Path((state / "inputs" / ".project-root").read_text(encoding="utf-8-sig").strip())
    except OSError:
        return None
    if install_key(owner) != state.name or not (owner / "hermes_cli" / "main.py").is_file():
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
