"""What ``save_config`` persists when the process serves a frozen session config.

A managed worker binds its session's creation-time config (``agent.safe_worker_policy``), so its
``load_config()`` is that snapshot, not the profile file. A read-modify-write in such a worker
(an ``always`` approval, a connector enable) must land only what it changed on the LIVE file:
writing the whole snapshot back would revert every profile edit made since the session was
created. Outside a frozen worker this is ``save_config``'s ordinary ``merge_existing`` handling.
"""

from __future__ import annotations

import copy
from typing import Any


def rebase_for_save(config: dict[str, Any], live_raw: dict[str, Any], merge_existing: bool) -> dict[str, Any]:
    from hermes_cli import config as _config

    snapshot = _config._worker_config_snapshot()
    if snapshot is not None:
        return _three_way(live_raw or {}, config, _config.load_config(), _frozen_for_deletion(snapshot))
    if merge_existing and live_raw:
        return _config._merge_partial_save(live_raw, config)
    return config


def _frozen_for_deletion(snapshot: dict[str, Any]) -> Any:
    """The snapshot branches that count as operator state.

    An ordinary snapshot only contains what the profile had at creation, so an explicit
    empty allowlist is a real branch. A bypass worker (safe mode, ``--ignore-user-config``)
    freezes a copy of the schema, so a branch that is still the schema default was never
    an operator choice and must not count as a deletion."""
    from agent.safe_worker_policy import _current_policy

    policy = _current_policy()
    if policy is None or not policy.ignore_user_config:
        return snapshot
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    return _schema_default_branch(snapshot, DEFAULT_CONFIG)


def _schema_default_branch(snapshot: Any, defaults: Any) -> Any:
    """*snapshot* without branches whose value is still the schema default."""
    if not isinstance(snapshot, dict) or not isinstance(defaults, dict):
        return copy.deepcopy(snapshot)
    kept: dict[str, Any] = {}
    for key, value in snapshot.items():
        if key not in defaults:
            kept[key] = copy.deepcopy(value)
            continue
        default = defaults[key]
        if value == default:
            continue
        if isinstance(value, dict) and isinstance(default, dict):
            nested = _schema_default_branch(value, default)
            if nested:
                kept[key] = nested
        else:
            kept[key] = copy.deepcopy(value)
    return kept


_SAME_AS_BASE = object()


def _three_way(live: Any, desired: Any, base: Any, frozen: Any = _SAME_AS_BASE) -> Any:
    """*live* with the caller's edits (*desired* relative to the *base* it loaded) applied.
    Mappings recurse; list edits add/remove items; a key the caller dropped is left alone. A
    mapping/list branch deleted (or retyped) on *live* since *base* stays as live has it: a
    leaf edit must not resurrect the stale snapshot subtree around it.

    *frozen* is the operator-owned snapshot under *base* (``load_config()`` merges the schema
    defaults over the raw snapshot). A branch in neither *frozen* nor *live* came only from the
    defaults, so nothing deleted it: the edit applies to the default (a fresh profile's first
    ``always`` grant, including one made in safe mode). A branch the operator set and the live
    file then lost stays deleted."""
    if frozen is _SAME_AS_BASE:
        frozen = base
    if desired == base:
        return copy.deepcopy(live)
    if isinstance(desired, dict) and isinstance(base, dict) and isinstance(live, dict):
        frozen_map = frozen if isinstance(frozen, dict) else {}
        merged = copy.deepcopy(live)
        for key, value in desired.items():
            if key not in base:
                merged[key] = copy.deepcopy(value)
            elif value != base[key]:
                kind = dict if isinstance(base[key], dict) else list if isinstance(base[key], list) else None
                current = live.get(key)
                if kind is not None and not isinstance(current, kind):
                    if key in live or key in frozen_map:
                        continue  # concurrently deleted/replaced live branch: live wins
                    current = copy.deepcopy(base[key])  # default-only branch
                merged[key] = _three_way(current, value, base[key], frozen_map.get(key, {}))
        return merged
    if isinstance(desired, list) and isinstance(base, list) and isinstance(live, list):
        removed = [item for item in base if item not in desired]
        kept = [item for item in live if item not in removed]
        return kept + [copy.deepcopy(item) for item in desired if item not in base and item not in kept]
    return copy.deepcopy(desired)
