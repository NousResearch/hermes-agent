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
    read can trigger, so it must never start — or re-enter — one.) Either hosted signal ignores
    it: the start command's ``--hosted`` flag (honoured before any verdict freezes, exactly as
    the shared code-admission predicate honours it — the fence's allowed roots read this tree,
    so a pre-freeze override would otherwise admit a writable root) or a frozen hosted verdict.
    Both reads are import-free beyond the stdlib-only fence."""
    from hermes_cli.hosted_code_fence import armed_by_start_flag, hosted_start_flag_requested
    from hermes_cli.hosted_mode import hosted_verdict_frozen

    env_override = os.getenv("HERMES_BUNDLED_PLUGINS")
    hosted = (hosted_start_flag_requested() or armed_by_start_flag()
              or hosted_verdict_frozen() is True)
    if env_override and not hosted:
        return Path(env_override)
    from hermes_cli._startup_fast import project_root_str

    return Path(project_root_str()) / "plugins"
