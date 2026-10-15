"""``include:`` / ``includes:`` in config.yaml: merge order, change detection, save-side stripping.

A config file may name other YAML files (paths relative to the declaring file) whose values sit
under its own: the declaring file wins, nested includes recurse, cycles are skipped. Loaders cache
on the root file's signature, so each include's stat signature is recorded and re-checked without
reparsing YAML. An include that exists but cannot be read raises ``ValueError``.
"""

from __future__ import annotations

import copy
import logging
import os
from pathlib import Path
from typing import Any, Optional

from utils import fast_safe_load, file_signature

logger = logging.getLogger(__name__)

CONFIG_INCLUDE_KEYS = ("include", "includes")
# config path -> include signatures recorded by load_config(); see config_includes_unchanged().
INCLUDE_SIGNATURES: dict[str, tuple[tuple[Any, ...], ...]] = {}


def without_include_keys(config: dict[str, Any]) -> dict[str, Any]:
    return {k: copy.deepcopy(v) for k, v in config.items() if k not in CONFIG_INCLUDE_KEYS}


def iter_include_values(config: dict[str, Any]) -> list[Any]:
    values: list[Any] = []
    for key in CONFIG_INCLUDE_KEYS:
        raw = config.get(key)
        if raw is not None:
            values.extend(raw if isinstance(raw, list) else [raw])
    return values


def resolve_include_path(config_path: Path, raw: Any) -> Optional[Path]:
    if not isinstance(raw, str) or not raw.strip():
        logger.warning("Ignoring non-string config include entry in %s", config_path)
        return None
    path = Path(os.path.expandvars(os.path.expanduser(raw.strip())))
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()


def read_config_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8-sig") as f:
        data = fast_safe_load(f) or {}
    return data if isinstance(data, dict) else {}


def merge_config_includes(
    config_path: Path, config: dict[str, Any], *, seen: Optional[set[Path]] = None,
) -> tuple[dict[str, Any], tuple[tuple[Any, ...], ...]]:
    """Merge relative includes in order, with the declaring file taking precedence."""
    from hermes_cli.config import _deep_merge

    seen = set(seen or ()) | {config_path.resolve()}
    merged: dict[str, Any] = {}
    signatures: list[tuple[Any, ...]] = []
    for raw_include in iter_include_values(config):
        include_path = resolve_include_path(config_path, raw_include)
        if include_path is None:
            continue
        if include_path in seen:
            logger.warning("Skipping recursive config include %s from %s", include_path, config_path)
            continue
        try:
            st = include_path.stat()
        except FileNotFoundError:
            logger.warning("Config include %s referenced by %s does not exist", include_path, config_path)
            signatures.append((str(config_path), raw_include, str(include_path), -1, -1, -1, -1))
            continue
        signatures.append((str(config_path), raw_include, str(include_path), *file_signature(st)))
        try:
            include_config = read_config_yaml(include_path)
        except Exception as e:
            raise ValueError(f"Invalid config include {include_path}") from e
        include_merged, include_sig = merge_config_includes(include_path, include_config, seen=seen)
        signatures.extend(include_sig)
        merged = _deep_merge(merged, include_merged)
    return _deep_merge(merged, without_include_keys(config)), tuple(signatures)


def merge_and_record(config_path: Path, path_key: str, config: dict[str, Any]) -> dict[str, Any]:
    """``merge_config_includes`` for load_config(), keeping the signatures its cache re-checks."""
    merged, INCLUDE_SIGNATURES[path_key] = merge_config_includes(config_path, config)
    return merged


def config_includes_unchanged(path_key: str, signatures=None) -> bool:
    """Stat dependencies without reparsing YAML; each reader owns its snapshot."""
    if signatures is None:
        signatures = INCLUDE_SIGNATURES.get(path_key, ())
    for parent, raw, previous_path, *previous_signature in signatures:
        path = resolve_include_path(Path(parent), raw)
        if path is None or str(path) != previous_path:
            return False
        try:
            signature = file_signature(path.stat())
        except FileNotFoundError:
            signature = (-1, -1, -1, -1)
        except OSError:
            return False
        if signature != tuple(previous_signature):
            return False
    return True


def includes_for_save(config_path: Path, raw: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """The include declarations of the file on disk and the values they currently contribute."""
    declarations = {key: raw[key] for key in CONFIG_INCLUDE_KEYS if key in raw}
    included, _ = merge_config_includes(config_path, declarations)
    return declarations, included


def strip_included_config_values(
    config: dict[str, Any], included: dict[str, Any], *,
    preserve_keys: Optional[set[tuple[str, ...]]] = None,
) -> dict[str, Any]:
    """Keep explicit overrides; do not materialize unchanged included values."""
    preserve_keys = set(preserve_keys or ())
    omitted = object()

    def strip(value: Any, baseline: Any, path: tuple[str, ...]) -> Any:
        if path in preserve_keys:
            return copy.deepcopy(value)
        if isinstance(value, dict):
            baseline_dict = baseline if isinstance(baseline, dict) else {}
            out = {}
            for key, child in value.items():
                stripped = strip(child, baseline_dict.get(key, omitted), path + (key,))
                if stripped is not omitted:
                    out[key] = stripped
            return out if out else omitted
        return omitted if baseline is not omitted and value == baseline else copy.deepcopy(value)

    result = {}
    for key, value in config.items():
        stripped = strip(value, included.get(key, omitted), (key,))
        if stripped is not omitted:
            result[key] = stripped
    return result


def strip_for_save(
    normalized: dict[str, Any], included: dict[str, Any], declarations: dict[str, Any],
    preserve_keys: set[tuple[str, ...]],
) -> dict[str, Any]:
    """What save_config writes: included values only where overridden, plus the declarations."""
    if included:
        from hermes_cli.config import _canonicalize_config

        normalized = strip_included_config_values(
            normalized, _canonicalize_config(included), preserve_keys=preserve_keys)
    normalized.update(copy.deepcopy(declarations))
    return normalized
