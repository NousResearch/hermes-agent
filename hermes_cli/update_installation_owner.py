"""A durable data-root identity for source checkouts without committed PM ownership.

An explicit update configuration operation can bind an otherwise unowned
checkout. The binding lives with the checkout, so unrelated HERMES_HOME roots
cannot create competing channel records or scheduler operation locks for it.
Read-only lookups never create it, and conflicting owner evidence fails closed.
"""

from __future__ import annotations

import json
from pathlib import Path

from hermes_constants import get_default_hermes_root, get_hermes_home
from pm.environments import install_key, installed_home_root


def owner_path(project_root: Path) -> Path:
    from hermes_constants import _get_platform_default_hermes_home
    from hermes_cli.update_lock import checkout_lock_path

    root = Path(project_root).resolve()
    metadata = checkout_lock_path(root).parent
    if metadata != root:
        return metadata / f"hermes-update-owner-{install_key(root)}.json"
    # Non-Git ZIP source trees have no protected metadata directory. Keep the
    # binding outside the payload so a failed/successful ZIP swap cannot alter it.
    return _get_platform_default_hermes_home() / "installs" / install_key(root) / "update-owner.json"


def _bound_owner(project_root: Path) -> Path | None:
    path = owner_path(project_root)
    if path.is_symlink():
        raise ValueError(f"Installation update owner must not be a symlink: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        return None
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError(f"Cannot read installation update owner at {path}: {exc}") from exc
    if (not isinstance(data, dict) or data.get("schema") != 1
            or data.get("installationRoot") != str(Path(project_root).resolve())
            or not isinstance(data.get("dataRoot"), str) or not Path(data["dataRoot"]).is_absolute()):
        raise ValueError(f"Invalid installation update owner: {path}")
    home = Path(data["dataRoot"]).resolve()
    if not home.is_dir():
        raise ValueError(f"Installation update data root is unavailable: {home}; inspect {path}")
    return home


def installation_home(project_root: Path, *, home: Path | None = None) -> Path:
    root = Path(project_root).resolve()
    known, bound = installed_home_root(root), _bound_owner(root)
    if known is not None and bound is not None and known.resolve() != bound:
        raise ValueError(f"Conflicting installation owners: PM uses {known}, but {owner_path(root)} records {bound}")
    return (known or bound or get_default_hermes_root(home=home or get_hermes_home())).resolve()


def ensure_installation_home(project_root: Path, *, home: Path | None = None) -> Path:
    """Bind only during an explicit operation; concurrent callers adopt the same owner."""
    from hermes_cli.update_lock import marker_mutex
    from utils import atomic_json_write

    root = Path(project_root).resolve()
    if installed_home_root(root) is not None:
        return installation_home(root, home=home)
    path = owner_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_suffix("")
    if lock.with_name(lock.name + ".lock").is_symlink():
        raise ValueError("Installation update owner lock must not be a symlink")
    with marker_mutex(lock, wait=0):
        selected = installation_home(root, home=home)
        if _bound_owner(root) is None:
            selected.mkdir(parents=True, exist_ok=True)
            atomic_json_write(path, {"schema": 1, "installationRoot": str(root), "dataRoot": str(selected)},
                              indent=2, mode=0o600, fsync_dir=True)
        return selected
