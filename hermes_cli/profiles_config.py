"""A new profile's config document: seed the launch profile's model, migrate a copied config, and
read a profile's config through the config backend (``hermes_cli.profiles`` creation helpers)."""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Optional

from hermes_cli.config_backend import config_exists, read_config_doc, supports_file_tooling


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


def _load_config_dict(profile_dir: Path) -> Optional[dict]:
    """:func:`_load_yaml_dict` for a profile's config.yaml, read through the config backend."""
    from hermes_yaml import YAMLError
    try:
        data = read_config_doc(profile_dir / "config.yaml") or {}
    except (YAMLError, OSError, UnicodeError):  # FileNotFoundError included: no config
        return None
    return data if isinstance(data, dict) else None


def launch_model_seed(source_cfg: dict) -> dict:
    """The config a fresh profile needs to run the launch profile's model: its ``model`` block plus,
    when that block points at a custom ``providers:`` gateway (self-hosted / local endpoint), that
    provider's definition — ``model.provider: my-gateway`` alone is "Unknown provider" on the first
    turn. ``{}`` when the launch profile has no model."""
    model_cfg = source_cfg.get("model")
    if not model_cfg:
        return {}
    seed = {"model": model_cfg}
    providers = source_cfg.get("providers")
    name = model_cfg.get("provider") if isinstance(model_cfg, dict) else None
    if isinstance(providers, dict) and name in providers:
        seed["providers"] = {name: providers[name]}
    return seed


def _seed_model_config(profile_dir: Path) -> None:
    """Copy (not link) the active profile's model block into a fresh profile so it is usable;
    profiles stay independent islands afterwards."""
    config_path = profile_dir / "config.yaml"
    if config_exists(config_path):
        return
    with contextlib.suppress(Exception):  # creation must not fail over this; `hermes model` sets it later
        from hermes_constants import get_hermes_home
        from hermes_cli.config import atomic_config_write, read_user_config_raw
        source = get_hermes_home() / "config.yaml"
        seed = launch_model_seed(read_user_config_raw(source)) if config_exists(source) else {}
        if seed:
            atomic_config_write(config_path, seed)
