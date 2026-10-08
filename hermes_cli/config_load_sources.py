"""Runtime source signatures and recovery before optional overlays are applied."""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any

from utils import file_signature


def load_config_cache_signature(config_path: Path):
    """Keep the eight-integer cache layout while including managed and harness sources."""
    from hermes_cli import config
    from hermes_cli.harness_manifest import manifest_signature

    try:
        user_sig = file_signature(config_path.stat())
    except FileNotFoundError:
        user_sig = None
    managed_dir = config.managed_scope.get_managed_dir()
    try:
        managed_sig = file_signature((managed_dir / "config.yaml").stat()) if managed_dir else (0, 0, 0, 0)
    except OSError:
        managed_sig = (0, 0, 0, 0)
    harness_sig = manifest_signature(config_path.parent)
    if user_sig is None and managed_sig == (0, 0, 0, 0) and harness_sig == (0, 0, 0, 0):
        return None, None
    digest = hashlib.sha256(repr((managed_sig, harness_sig)).encode("ascii")).digest()
    source_sig = tuple(int.from_bytes(digest[offset:offset + 8], "big") for offset in range(0, 32, 8))
    return user_sig, (*(user_sig or (0, 0, 0, 0)), *source_sig)


def apply_runtime_sources(expanded: dict[str, Any], config_path: Path):
    from hermes_cli import config
    from hermes_cli.harness_manifest import apply_active_overlays

    tuned = apply_active_overlays(expanded, path=config_path.parent / "harness.yaml")
    return config._merge_managed_overlay(tuned)


def last_known_good_fallback(config_path: Path, path_key: str, cache_sig, exc: Exception):
    """Recover the user layer, then apply CURRENT harness and managed sources.

    Keeping a final effective snapshot would resurrect reverted or stale harness values when
    config.yaml becomes malformed. Preserve the user's overrides while rereading optional layers.
    """
    from hermes_cli import config

    lkg = config._LAST_EXPANDED_CONFIG_BY_PATH.get(path_key)
    fallback = "last-known-good"
    if lkg is None:
        from hermes_cli.config_backups import load_newest_good_backup
        from hermes_cli.moa_config import apply_user_moa_presets
        raw_good = load_newest_good_backup(config_path)
        if raw_good is not None:
            merged_good = config._deep_merge(copy.deepcopy(config.DEFAULT_CONFIG), raw_good)
            apply_user_moa_presets(merged_good, raw_good)
            lkg = config._canonicalize_config(merged_good)
            fallback = "last-known-good-backup"
    config._warn_config_parse_failure(config_path, exc, fallback=fallback if lkg is not None else "defaults")
    if lkg is None:
        return None
    expanded = config._expand_env_vars(copy.deepcopy(lkg))
    effective, _ = apply_runtime_sources(expanded, config_path)
    lkg_copy = config.FailedConfigRead(effective, error=exc)
    if cache_sig is not None:
        config._LOAD_CONFIG_CACHE[path_key] = (*cache_sig, lkg_copy, {})
    return lkg_copy
