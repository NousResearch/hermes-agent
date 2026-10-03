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
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from hermes_cli import config as _config
from hermes_cli import managed_scope
from hermes_cli.config_read_errors import _warn_config_parse_failure
from utils import fast_safe_load

# path -> raw user mapping from the last successful parse in this process; served (through the
# normal pipeline) when the file is later found mid-edit as broken YAML.
_LAST_GOOD_USER_RAW: Dict[str, Dict[str, Any]] = {}
# (path, strict) -> (*user_signature, *managed_signature, effective, env_snapshot).
_EFFECTIVE_CACHE: Dict[Tuple[str, bool], Tuple[Any, ...]] = {}
_GOOD_BACKUP_SIGNATURE: Dict[str, Tuple[int, int, int, int]] = {}


def _effective(raw: Dict[str, Any], *, fail_closed: bool = False) -> Dict[str, Any]:
    expanded = _config._expand_env_vars(raw)
    merged = managed_scope.apply_managed_overlay(
        expanded if isinstance(expanded, dict) else {}, fail_closed=fail_closed
    )
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
    _warn_config_parse_failure(config_path, exc, fallback=fallback if raw is not None else "defaults")
    return copy.deepcopy(raw) if raw is not None else {}


def _maybe_backup_good(
    config_path: Path, path_key: str, user_sig, *, side_effect_free: bool
) -> None:
    if (
        side_effect_free
        or user_sig is None
        or config_path != _config.get_config_path()
        or _GOOD_BACKUP_SIGNATURE.get(path_key) == user_sig
    ):
        return
    from hermes_cli.config_backups import backup_config
    backup_config(config_path, "good")
    _GOOD_BACKUP_SIGNATURE[path_key] = user_sig


def load_user_config_effective(
    config_path: Optional[Path] = None,
    *,
    fail_closed: bool = False,
    side_effect_free: bool = False,
) -> Dict[str, Any]:
    """User ``config.yaml`` → ``${VAR}`` expansion → managed overlay → model-key canonicalization.
    NO ``DEFAULT_CONFIG`` merge: a key absent from the file (and from the managed layer) is absent
    here, so ``{}`` sentinels and presence-sensitive bridges keep working. An absent file is an
    empty user layer (the managed layer still applies). Returns a fresh deepcopy.

    Broken YAML: ``fail_closed=True`` raises the parse error (for callers that keep their own
    last-good state); otherwise the last successfully parsed user file — in-process first, then
    the newest ``backups/config/*.good.*`` copy — is served through the same pipeline, so a
    mid-edit torn write never silently drops user overrides (same contract as ``load_config``).
    Strict reads also reject invalid managed roots and preserve explicit nulls. A
    ``side_effect_free`` read suppresses backup writes without suppressing the next ordinary one.
    Cached on the user + managed file signatures and the values of every referenced env var."""
    if config_path is None:
        config_path = _config.get_config_path()
    path_key = str(config_path)
    cache_key = (path_key, fail_closed)
    with _config._CONFIG_LOCK:
        user_sig, cache_sig = _config._load_config_cache_sig(config_path)
        cached = _EFFECTIVE_CACHE.get(cache_key)
        if cached is not None and cache_sig is not None and cached[:8] == cache_sig:
            if all(_config._env_ref_lookup(k) == v for k, v in cached[9].items()):
                _maybe_backup_good(
                    config_path, path_key, user_sig, side_effect_free=side_effect_free
                )
                return copy.deepcopy(cached[8])

        raw: Dict[str, Any] = {}
        recovered = False
        raw_hit = _config._RAW_CONFIG_CACHE.get(path_key)
        if user_sig is not None and raw_hit is not None and raw_hit[:4] == user_sig:
            raw = copy.deepcopy(raw_hit[4])  # one parse per process, shared with read_raw_config()
            _LAST_GOOD_USER_RAW.setdefault(path_key, copy.deepcopy(raw))
        elif user_sig is not None:
            try:
                with open(config_path, encoding="utf-8-sig") as f:
                    loaded = fast_safe_load(f)
            except Exception as exc:
                if fail_closed:
                    raise
                raw, recovered = _recover_user_raw(config_path, path_key, exc), True
            else:
                raw = loaded if isinstance(loaded, dict) else {}
                _config._RAW_CONFIG_CACHE[path_key] = (*user_sig, copy.deepcopy(raw))
                _LAST_GOOD_USER_RAW[path_key] = copy.deepcopy(raw)
        if not recovered:
            _maybe_backup_good(
                config_path, path_key, user_sig, side_effect_free=side_effect_free
            )

        env_snapshot = _config._env_ref_snapshot(raw)
        managed = managed_scope.load_managed_config(fail_closed=fail_closed)
        if managed:
            _config._env_ref_snapshot(managed, env_snapshot)
        effective = _effective(raw, fail_closed=fail_closed)
        # A recovered result is never cached under the corrupt file's signature: a later
        # ``fail_closed`` caller must still see the parse error, not a cache hit.
        if cache_sig is not None and not recovered:
            _EFFECTIVE_CACHE[cache_key] = (*cache_sig, copy.deepcopy(effective), env_snapshot)
        else:
            _EFFECTIVE_CACHE.pop(cache_key, None)
        return effective
