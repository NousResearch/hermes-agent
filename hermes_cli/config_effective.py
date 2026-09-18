"""The effective USER config: config.yaml + managed overlay + ``${VAR}`` expansion, no defaults.

``load_config()`` merges ``DEFAULT_CONFIG`` first, which is wrong for readers that treat a
missing key as "unset" (the gateway's presence-sensitive env bridge, ``cfg == {}`` sentinels,
cron model pinning) — so nine surfaces used to hand-roll raw-read → overlay → expand in
differing orders and none of them replayed the model-key canonicalization or the last-known-good
recovery ``load_config()`` gained. This module is that one primitive.

Order matches ``_load_config_impl``: the user layer is expanded BEFORE the managed overlay so a
managed ``${VAR}`` resolves against the process environment only (``apply_managed_overlay``
expands it) and can never be re-resolved through a profile's secret scope
(docs/design/managed-scope.md §4.1). ``read_user_config_raw`` stays the write-back primitive.
"""

from __future__ import annotations

import copy
import stat
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, cast

from hermes_cli import config as _config
from hermes_cli import managed_scope
from utils import fast_safe_load

# path -> raw user mapping from the last successful parse in this process; served (through the
# normal pipeline) when the file is later found mid-edit as broken YAML.
_LAST_GOOD_USER_RAW: Dict[str, Dict[str, Any]] = {}
# path -> (*user_signature, *managed_signature, effective, env_snapshot); see utils.file_signature.
_EFFECTIVE_CACHE: Dict[str, Tuple[Any, ...]] = {}
# Explicit-path, fail-closed values use a separate cache so a fail-open
# ``load_user_config_effective`` prewarm can never satisfy a safety-sensitive read.
_STRICT_VALUE_CACHE: Dict[str, Tuple[Tuple[Any, ...], Tuple[Any, ...], Dict[str, Any]]] = {}


class ConfigResolutionError(RuntimeError):
    """An explicit effective-config source could not be resolved safely."""


def _strict_source_signature(path: Path, *, label: str) -> Tuple[Any, ...]:
    """Identity of a config source; only a genuinely absent path is optional."""
    try:
        link_stat = path.lstat()
    except FileNotFoundError:
        return ("absent",)
    except OSError as exc:
        raise ConfigResolutionError(f"{label} at {path} is inaccessible: {exc}") from exc

    def _identity(st) -> Tuple[int, ...]:
        return (
            st.st_mtime_ns, st.st_size, st.st_ctime_ns, st.st_mode,
            st.st_ino, st.st_dev, st.st_uid, st.st_gid,
        )

    if stat.S_ISLNK(link_stat.st_mode):
        try:
            target_stat = path.stat()
        except OSError as exc:
            raise ConfigResolutionError(
                f"{label} at {path} is a dangling or inaccessible symlink: {exc}"
            ) from exc
        if not stat.S_ISREG(target_stat.st_mode):
            raise ConfigResolutionError(f"{label} at {path} does not target a regular file")
        return ("symlink", _identity(link_stat), _identity(target_stat))
    if not stat.S_ISREG(link_stat.st_mode):
        raise ConfigResolutionError(f"{label} at {path} is not a regular file")
    return ("regular", *_identity(link_stat))


def _read_mapping_strict(path: Path, *, label: str) -> Dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as handle:
            value = fast_safe_load(handle)
    except Exception as exc:
        raise ConfigResolutionError(f"Could not read or parse {label} at {path}: {exc}") from exc
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigResolutionError(
            f"{label} at {path} has a non-mapping root ({type(value).__name__})"
        )
    return value


def _strict_managed_source() -> Tuple[Tuple[Any, ...], Optional[Path]]:
    managed_dir = managed_scope.get_managed_dir()
    if managed_dir is None:
        return (("absent",), None)
    path = managed_dir / "config.yaml"
    signature = _strict_source_signature(path, label="Managed config source")
    return signature, (None if signature == ("absent",) else path)


def resolve_effective_config_value(config_path: Path, *keys: str, default: Any = None) -> Any:
    """Strictly resolve one value from an explicit user config plus managed overlay.

    Missing sources are valid empty layers. Present-but-broken sources raise
    :class:`ConfigResolutionError`. Results are cached by complete user and managed
    provenance (including symlink target, inode, ownership and mode).
    """
    config_path = Path(config_path).expanduser()
    user_sig = _strict_source_signature(config_path, label="Config source")
    managed_sig, managed_path = _strict_managed_source()
    path_key = str(config_path)
    cached = _STRICT_VALUE_CACHE.get(path_key)
    if cached is not None and cached[0] == user_sig and cached[1] == managed_sig:
        effective = copy.deepcopy(cached[2])
    else:
        user_raw = {} if user_sig == ("absent",) else _read_mapping_strict(config_path, label="Config")
        managed_raw = {} if managed_path is None else _read_mapping_strict(managed_path, label="Managed config")
        expanded = _config._normalize_root_model_keys(
            cast(Dict[str, Any], _config._expand_env_vars(user_raw))
        )
        managed_expanded = _config._normalize_root_model_keys(
            cast(Dict[str, Any], _config._expand_env_vars(managed_raw))
        )
        if isinstance(managed_expanded.get("model"), str):
            managed_expanded = dict(managed_expanded)
            managed_expanded["model"] = {"default": managed_expanded["model"]}
        effective = _config._deep_merge(expanded, managed_expanded)
        _STRICT_VALUE_CACHE[path_key] = (user_sig, managed_sig, copy.deepcopy(effective))

    value: Any = effective
    for key in keys:
        if not isinstance(value, dict):
            raise ConfigResolutionError(
                f"Config path {'.'.join(keys)} at {config_path} crosses a non-mapping value"
            )
        if key not in value:
            value = default
            break
        value = value[key]
    return copy.deepcopy(value)


def _effective(raw: Dict[str, Any]) -> Dict[str, Any]:
    expanded = _config._expand_env_vars(raw)
    merged = managed_scope.apply_managed_overlay(expanded if isinstance(expanded, dict) else {})
    return _config._normalize_root_model_keys(merged if isinstance(merged, dict) else {})


def _recover_user_raw(config_path: Path, path_key: str, exc: Exception) -> Dict[str, Any]:
    """Last-known-good raw user mapping after a parse failure: this process's last good parse,
    else the newest ``good`` copy in backups/config/, else ``{}`` (warned as defaults)."""
    raw = _LAST_GOOD_USER_RAW.get(path_key)
    fallback = "last-known-good"
    if raw is None:
        from hermes_cli.config_backups import load_newest_good_backup
        raw = load_newest_good_backup(config_path)
        fallback = "last-known-good-backup"
    _config._warn_config_parse_failure(config_path, exc, fallback=fallback if raw is not None else "defaults")
    return copy.deepcopy(raw) if raw is not None else {}


def load_user_config_effective(config_path: Optional[Path] = None, *, fail_closed: bool = False) -> Dict[str, Any]:
    """User ``config.yaml`` → ``${VAR}`` expansion → managed overlay → model-key canonicalization.
    NO ``DEFAULT_CONFIG`` merge: a key absent from the file (and from the managed layer) is absent
    here, so ``{}`` sentinels and presence-sensitive bridges keep working. An absent file is an
    empty user layer (the managed layer still applies). Returns a fresh deepcopy.

    Broken YAML: ``fail_closed=True`` raises the parse error (for callers that keep their own
    last-good state); otherwise the last successfully parsed user file — in-process first, then
    the newest ``backups/config/*.good.*`` copy — is served through the same pipeline, so a
    mid-edit torn write never silently drops user overrides (same contract as ``load_config``).
    Cached on the user + managed file signatures and the values of every referenced env var."""
    if config_path is None:
        config_path = _config.get_config_path()
    path_key = str(config_path)
    with _config._CONFIG_LOCK:
        user_sig, cache_sig = _config._load_config_cache_sig(config_path)
        cached = _EFFECTIVE_CACHE.get(path_key)
        if cached is not None and cache_sig is not None and cached[:8] == cache_sig:
            if all(_config._env_ref_lookup(k) == v for k, v in cached[9].items()):
                return copy.deepcopy(cached[8])

        raw: Dict[str, Any] = {}
        recovered = False
        raw_hit = _config._RAW_CONFIG_CACHE.get(path_key)
        if user_sig is not None and raw_hit is not None and raw_hit[:4] == user_sig:
            raw = copy.deepcopy(raw_hit[4])  # one parse per process, shared with read_raw_config()
            _LAST_GOOD_USER_RAW.setdefault(path_key, copy.deepcopy(raw))
        elif user_sig is not None:
            try:
                with open(config_path, encoding="utf-8") as f:
                    loaded = fast_safe_load(f)
            except Exception as exc:
                if fail_closed:
                    raise
                raw, recovered = _recover_user_raw(config_path, path_key, exc), True
            else:
                raw = loaded if isinstance(loaded, dict) else {}
                _config._RAW_CONFIG_CACHE[path_key] = (*user_sig, copy.deepcopy(raw))
                _LAST_GOOD_USER_RAW[path_key] = copy.deepcopy(raw)
                # Same copy load_config keeps: a fresh process recovers from it (see _recover_user_raw).
                # Only for the ACTIVE home — a read of another profile's file (doctor, TUI cwd lookup)
                # must not create backups/ inside that profile.
                if config_path == _config.get_config_path():
                    from hermes_cli.config_backups import backup_config
                    backup_config(config_path, "good")

        env_snapshot = _config._env_ref_snapshot(raw)
        managed = managed_scope.load_managed_config()
        if managed:
            _config._env_ref_snapshot(managed, env_snapshot)
        effective = _effective(raw)
        # A recovered result is never cached under the corrupt file's signature: a later
        # ``fail_closed`` caller must still see the parse error, not a cache hit.
        if cache_sig is not None and not recovered:
            _EFFECTIVE_CACHE[path_key] = (*cache_sig, copy.deepcopy(effective), env_snapshot)
        else:
            _EFFECTIVE_CACHE.pop(path_key, None)
        return effective
