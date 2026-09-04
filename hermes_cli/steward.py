"""Steward classification and refusal text for sealed install trees.

Adapted from the restack branch's ``installation/tree.py``. Our branch has
no ``installation`` package yet — detection here is rewritten onto the two
facts this branch already ships:

* ``.git`` at the project root says the tree is a checkout the uninstaller
  / updater may mutate.
* The build stamp (``install-stamp.json``, written by
  ``scripts/write_install_stamp.py``) names the steward in
  ``distribution`` (``desktop-app``, ``docker``, ``nix``, or a future
  package manager) for sealed trees.

A sealed tree (no ``.git``) belongs to a steward: only the steward removes
or replaces its code. The refusal messages below always leave the user a
working next step for their data (``hermes uninstall --data`` / the
desktop app's Settings -> About page).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

BUILD_INFO_NAME = "install-stamp.json"

STEWARD_DESKTOP = "desktop-app"
STEWARD_DOCKER = "docker"
STEWARD_NIX = "nix"

# What `hermes update` says in a sealed tree, per steward — the refusal
# table shared by the update admission gate. The fallback covers stewards
# this build does not know (a newer package-manager value read by older
# code).
STEWARD_UPDATE_MESSAGES = {
    STEWARD_DESKTOP: (
        "✗ This Hermes runs from inside the desktop app bundle.\n"
        "\n"
        "Manage updates from within the desktop app.\n"
        "Prefer a self-managed source install? See:\n"
        "  https://hermes-agent.nousresearch.com/docs/user-guide/switching-to-source"
    ),
    STEWARD_NIX: (
        "✗ This Hermes runs from the Nix store.\n"
        "\n"
        "The store path is immutable. Update through your flake:\n"
        "  nix flake update && rebuild your profile or system"
    ),
}

_STEWARD_UPDATE_FALLBACK = (
    "✗ This Hermes install is managed by {steward}.\n"
    "\n"
    "The tree has no git checkout, so `hermes update` cannot update it.\n"
    "Update it with the tool that installed it."
)

# What the uninstaller says when it refuses to remove code from a sealed
# tree. The steward put the code there; the steward removes it. The
# desktop-app message is per-OS because each OS owns app removal
# differently.
_STEWARD_DELETE_DATA_PREAMBLE = "To delete your Hermes data (chats, configuration, etc),\n"
_STEWARD_DELETE_DATA_CLI = "run:\n$ hermes uninstall --data\n"
_STEWARD_DELETE_DATA_DESKTOP = "Open Hermes Desktop, go to Settings -> About, and delete your data from there.\n"

_STEWARD_UNINSTALL_MESSAGES = {
    STEWARD_DOCKER: (
        "✗ This Hermes runs from a Docker image.\n"
        "\n"
        "There is no code to uninstall — remove the container and image:\n"
        "  docker rm <container> && docker rmi nousresearch/hermes-agent\n"
        "\n" +
        _STEWARD_DELETE_DATA_PREAMBLE +
        _STEWARD_DELETE_DATA_CLI
    ),
    STEWARD_NIX: (
        "✗ This Hermes was installed by Nix.\n"
        "\n"
        "The store path is immutable — uninstall it the same way you\n"
        "installed it: remove hermes-agent from your flake / profile\n"
        "(e.g. `nix profile remove`), then rebuild.\n"
        "\n" +
        _STEWARD_DELETE_DATA_PREAMBLE +
        _STEWARD_DELETE_DATA_CLI
    ),
}

_STEWARD_MANAGED_BY_DESKTOP = "✗ Hermes is managed by the desktop app.\n"

_STEWARD_DESKTOP_UNINSTALL_BY_PLATFORM = {
    "win32": (
        _STEWARD_MANAGED_BY_DESKTOP +
        "\n"
        "Remove the app from Windows Settings → Apps → Installed apps.\n" +
        _STEWARD_DELETE_DATA_PREAMBLE +
        _STEWARD_DELETE_DATA_DESKTOP
    ),
    "darwin": (
        _STEWARD_MANAGED_BY_DESKTOP +
        "\n"
        "Quit the app and drag Hermes.app from Applications to the Trash.\n" +
        _STEWARD_DELETE_DATA_PREAMBLE +
        _STEWARD_DELETE_DATA_DESKTOP
    ),
}

_STEWARD_DESKTOP_UNINSTALL_DEFAULT = (
    _STEWARD_MANAGED_BY_DESKTOP +
    "\n"
    "Delete the Hermes AppImage (or app directory) from wherever you\n"
    "saved it.\n" +
    _STEWARD_DELETE_DATA_PREAMBLE +
    _STEWARD_DELETE_DATA_DESKTOP
)

_STEWARD_UNINSTALL_FALLBACK = (
    "✗ Hermes is managed by {steward}.\n"
    "\n"
    "The tree has no git checkout, so the uninstaller will not remove it.\n"
    "Remove it with the tool that installed it.\n"
    "\n" +
    # A generic package-manager steward has no desktop app, and this
    # refusal prints in a CLI context — point at the CLI data path.
    _STEWARD_DELETE_DATA_PREAMBLE +
    _STEWARD_DELETE_DATA_CLI
)


def steward_update_message(steward: str) -> str:
    """The `hermes update` refusal text for a sealed tree."""
    message = STEWARD_UPDATE_MESSAGES.get(steward)
    if message is not None:
        return message
    return _STEWARD_UPDATE_FALLBACK.format(steward=steward)


def steward_uninstall_message(steward: str, platform: "str | None" = None) -> str:
    """The uninstall refusal text for a sealed tree."""
    if steward == STEWARD_DESKTOP:
        key = platform if platform is not None else sys.platform
        return _STEWARD_DESKTOP_UNINSTALL_BY_PLATFORM.get(key, _STEWARD_DESKTOP_UNINSTALL_DEFAULT)
    message = _STEWARD_UNINSTALL_MESSAGES.get(steward)
    if message is not None:
        return message
    return _STEWARD_UNINSTALL_FALLBACK.format(steward=steward)


def read_install_stamp(project_root: Path) -> dict:
    """The build stamp of ``project_root``, or ``{}``.

    Tolerant by design: steward classification must not crash a refusal
    path on a malformed stamp — a tree we cannot prove is ours still
    refuses (see :func:`sealed_steward`), so garbage degrades safely.
    """
    try:
        data = json.loads(
            (Path(project_root) / BUILD_INFO_NAME).read_text(encoding="utf-8-sig")
        )
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def is_bundled_payload(project_root: Path) -> bool:
    """True when ``project_root`` is a bundled desktop payload's ``repo/``.

    THE shape predicate for the sealed-bundle branch, and a mirror of the
    Electron shell's ``installShape() === 'bundled'``: the build stamp is
    the authority, so filesystem coincidences (a sibling ``bin/``, a
    surviving ``apps/desktop/package.json``) can never promote a checkout
    into it. A missing, foreign, or unreadable stamp means NOT bundled.
    """
    try:
        return read_install_stamp(Path(project_root)).get("payload") == "bundled"
    except Exception:  # noqa: BLE001 — a bad stamp must not take a caller down
        return False


def sealed_steward(project_root: Path) -> Optional[str]:
    """The steward owning the sealed tree at ``project_root``, or ``None``.

    ``None`` means the tree is a git checkout (``.git`` present — a
    directory, or a worktree/submodule gitfile) and is ours to mutate.
    Everything else is sealed: the stamp's ``distribution`` names the
    steward, and a missing or unreadable stamp gives ``"unknown"`` —
    a tree we cannot prove is ours is not ours to remove.
    """
    root = Path(project_root)
    if (root / ".git").exists():
        return None
    distribution = read_install_stamp(root).get("distribution")
    return distribution if isinstance(distribution, str) and distribution else "unknown"


def classify_install(project_root: Path) -> "tuple[str, bool]":
    """(steward, code_removal_allowed) for the tree at ``project_root``.

    A git checkout reports ``("git", True)``; a sealed tree reports its
    steward and ``False``. Used by ``gui_install_summary`` so the desktop
    UI can gate its destructive options on the same ladder the CLI uses.
    """
    steward = sealed_steward(project_root)
    if steward is None:
        return ("git", True)
    return (steward, False)
