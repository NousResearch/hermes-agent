"""The bundled plugins tree: where it is, and the one rule that decides a hosted process may
not move it (the facade ``hermes_cli.plugins`` re-exports this; call-time importers keep the
monkeypatch seam)."""

from __future__ import annotations

import os
from pathlib import Path


def get_bundled_plugins_dir() -> Path:
    """Bundled ``plugins/`` dir: ``HERMES_BUNDLED_PLUGINS`` (Nix wrapper / packaged installs,
    read-only store paths) on a local install, else the in-repo path. A hosted process ignores
    the override: the tree its in-process trust resolves from is always the one inside the
    installed Hermes package, never a root the process environment names. (The frozen boot
    verdict is consulted, not ``is_hosted()``: this runs inside discovery, which a first config
    read can trigger, so it must never start — or re-enter — one. Until the boot's first read
    decides the process, the override keeps its local meaning, and nothing imports plugin code
    before a hosted verdict froze.)"""
    from hermes_cli.hosted_mode import hosted_verdict_frozen

    env_override = os.getenv("HERMES_BUNDLED_PLUGINS")
    if env_override and hosted_verdict_frozen() is not True:
        return Path(env_override)
    return Path(__file__).resolve().parent.parent / "plugins"
