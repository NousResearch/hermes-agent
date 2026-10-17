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

    if _config._worker_config_snapshot() is not None:
        return _three_way(live_raw or {}, config, _config.load_config())
    if merge_existing and live_raw:
        return merge_partial_save(live_raw, config)
    return config


def merge_partial_save(raw: dict, override: dict) -> dict:
    """Merge *override* over *raw* for partial ``save_config`` writes.
    Omitted top-level sections are preserved; shared dict sections deep-merge so one nested key
    can change without dropping siblings on disk. Key REMOVALS are not supported here —
    migrations go through ``_persist_migration`` with a full ``read_raw_config()`` dict."""
    from hermes_cli.config import _deep_merge

    result = copy.deepcopy(override)
    for key, value in raw.items():
        if key not in result:
            result[key] = copy.deepcopy(value)
        elif isinstance(result.get(key), dict) and isinstance(value, dict):
            result[key] = _deep_merge(value, result[key])
    return result


def _three_way(live: Any, desired: Any, base: Any) -> Any:
    """*live* with the caller's edits (*desired* relative to the *base* it loaded) applied.
    Mappings recurse; list edits add/remove items; a key the caller dropped is left alone. A
    mapping/list branch deleted (or retyped) on *live* since *base* stays as live has it: a
    leaf edit must not resurrect the stale snapshot subtree around it."""
    if desired == base:
        return copy.deepcopy(live)
    if isinstance(desired, dict) and isinstance(base, dict) and isinstance(live, dict):
        merged = copy.deepcopy(live)
        for key, value in desired.items():
            if key not in base:
                merged[key] = copy.deepcopy(value)
            elif value != base[key]:
                kind = dict if isinstance(base[key], dict) else list if isinstance(base[key], list) else None
                if kind is not None and not isinstance(live.get(key), kind):
                    continue  # concurrently deleted/replaced live branch: live wins
                merged[key] = _three_way(live.get(key), value, base[key])
        return merged
    if isinstance(desired, list) and isinstance(base, list) and isinstance(live, list):
        removed = [item for item in base if item not in desired]
        kept = [item for item in live if item not in removed]
        return kept + [copy.deepcopy(item) for item in desired if item not in base and item not in kept]
    return copy.deepcopy(desired)
