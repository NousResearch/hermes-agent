"""Migrate a new profile's copied config document (``hermes_cli.profiles`` creation helper)."""

from __future__ import annotations

import contextlib
from pathlib import Path
from hermes_cli.config_backend import config_exists, supports_file_tooling


def _migrate_profile_config_if_outdated(profile_dir: Path) -> None:
    """Migrate a copied config.yaml to the current schema (non-interactive, scoped to the new
    profile); otherwise the first desktop/doctor view shows a scary ``v0 -> latest`` warning."""
    if not supports_file_tooling():
        return  # no local config.yaml to migrate: a remote backend migrates in memory on read (D12)
    if not config_exists(profile_dir / "config.yaml"):
        return
    # Creation must not fail over an unmigratable old config; `hermes doctor --fix` surfaces
    # the detailed error in the target profile.
    with contextlib.suppress(Exception):
        from hermes_constants import reset_hermes_home_override, set_hermes_home_override
        from hermes_cli.config import check_config_version, migrate_config
        token = set_hermes_home_override(str(profile_dir))
        try:
            current_ver, latest_ver = check_config_version()
            if current_ver < latest_ver:
                migrate_config(interactive=False, quiet=True)
        finally:
            reset_hermes_home_override(token)

