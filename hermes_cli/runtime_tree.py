"""Derive who owns the running tree — no stored mode flags.

The install model has two axes (design:
.hermes/plans/2026-08-07_183000-two-axis-install-model.md):

* A tree with ``.git`` is a **git checkout**: ``hermes update`` owns it.
  The checkout's existence IS the fact; no manifest records it.
* A tree without ``.git`` is **sealed**: something external replaces it
  wholesale. The build stamp (``.hermes_build_info.json``) names that
  steward in its ``distribution`` field: ``desktop-app`` (the embedded
  desktop bundle), ``docker``, ``nix``, or a future package manager.

The update channel (``stable`` or ``main``) lives in config.yaml under
``update.channel``. It applies to git checkouts only — sealed trees
version-track through their stewards.

If a future feature writes to user checkouts (nothing does today), it
must add an explicit opt-out fact FIRST. The old ``manageStyle: ejected``
stickiness guarded against desktop-side adoption and rematerialization;
both are deleted, so the guard went with them.

This is a pure-stdlib leaf module. It does not import hermes_cli.config.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

BUILD_INFO_NAME = ".hermes_build_info.json"

STEWARD_DESKTOP = "desktop-app"
STEWARD_DOCKER = "docker"
STEWARD_NIX = "nix"

CHANNEL_MAIN = "main"
CHANNEL_STABLE = "stable"
_VALID_CHANNELS = (CHANNEL_MAIN, CHANNEL_STABLE)

# What `hermes update` says in a sealed tree, per steward. The fallback
# covers stewards this build does not know (a newer package-manager value
# read by older code).
STEWARD_UPDATE_MESSAGES = {
    STEWARD_DESKTOP: (
        "✗ This Hermes runs from inside the desktop app bundle.\n"
        "\n"
        "The app updates itself, and every app update carries the agent\n"
        "with it. There is nothing for `hermes update` to do here.\n"
        "\n"
        "To manage the agent from the command line with git, run\n"
        "`hermes update --eject`. It installs a source checkout and a\n"
        "desktop app built from it."
    ),
    STEWARD_DOCKER: (
        "✗ This Hermes runs from a Docker image.\n"
        "\n"
        "The image is immutable. Pull the new image to update:\n"
        "  docker pull nousresearch/hermes-agent:latest"
    ),
    STEWARD_NIX: (
        "✗ This Hermes runs from the Nix store.\n"
        "\n"
        "The store path is immutable. Update through your flake:\n"
        "  nix flake update && rebuild your profile or system"
    ),
}

_STEWARD_FALLBACK_MESSAGE = (
    "✗ This Hermes install is managed by {steward}.\n"
    "\n"
    "The tree has no git checkout, so `hermes update` cannot update it.\n"
    "Update it with the tool that installed it."
)


@dataclass(frozen=True)
class GitCheckout:
    """A tree with .git — `hermes update` owns it."""

    root: Path


@dataclass(frozen=True)
class Sealed:
    """A gitless tree — the steward replaces it wholesale."""

    root: Path
    steward: str


def read_build_info(project_root: Path) -> dict:
    """The baked build stamp of ``project_root``, or ``{}``."""
    try:
        data = json.loads((Path(project_root) / BUILD_INFO_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def runtime_tree(project_root: Path) -> GitCheckout | Sealed:
    """Classify the tree at ``project_root``.

    ``.git`` present (a directory, or a worktree/submodule gitfile) means a
    git checkout. Everything else is sealed, with the steward read from the
    build stamp; a missing or unknown stamp gives steward ``"unknown"``.
    """
    root = Path(project_root)
    if (root / ".git").exists():
        return GitCheckout(root=root)

    distribution = read_build_info(root).get("distribution")
    steward = distribution if isinstance(distribution, str) and distribution else "unknown"
    return Sealed(root=root, steward=steward)


def steward_update_message(steward: str) -> str:
    """The `hermes update` refusal text for a sealed tree."""
    message = STEWARD_UPDATE_MESSAGES.get(steward)
    if message is not None:
        return message
    return _STEWARD_FALLBACK_MESSAGE.format(steward=steward)


def managed_install_roots() -> tuple[Path, ...]:
    """The canonical roots where installers create the agent checkout.

    * per-user: ``$HERMES_HOME/hermes-agent`` (usually ``~/.hermes``)
    * FHS root installs (install.sh as root on Linux):
      ``/usr/local/lib/hermes-agent``
    """
    from hermes_constants import get_hermes_home

    return (get_hermes_home() / "hermes-agent", Path("/usr/local/lib/hermes-agent"))


def is_managed_install_root(path: Path) -> bool:
    """True when ``path`` is a canonical installer-created checkout root.

    `hermes update` updates these without a question. A checkout anywhere
    else is somebody's working tree, and update asks first.
    """
    try:
        resolved = Path(path).resolve()
    except OSError:
        return False
    for root in managed_install_roots():
        try:
            if resolved == root.resolve():
                return True
        except OSError:
            continue
    return False


def resolve_update_channel(config: Optional[dict] = None) -> str:
    """The effective update channel for a git checkout.

    ``update.channel`` from config.yaml when it is ``stable`` or ``main``;
    anything else (missing, ``auto``, unknown) means ``main``. Sealed trees
    never ask: their stewards own versioning.
    """
    configured = None
    if isinstance(config, dict):
        update_cfg = config.get("update")
        if isinstance(update_cfg, dict):
            configured = update_cfg.get("channel")
    if isinstance(configured, str) and configured.strip().lower() in _VALID_CHANNELS:
        return configured.strip().lower()
    return CHANNEL_MAIN
