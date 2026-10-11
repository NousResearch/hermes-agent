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
        return _config._merge_partial_save(live_raw, config)
    return config


def _three_way(live: Any, desired: Any, base: Any) -> Any:
    """*live* with the caller's edits (*desired* relative to the *base* it loaded) applied.
    Mappings recurse; list edits add/remove items; a key the caller dropped is left alone."""
    if desired == base:
        return copy.deepcopy(live)
    if isinstance(desired, dict) and isinstance(base, dict) and isinstance(live, dict):
        merged = copy.deepcopy(live)
        for key, value in desired.items():
            if key not in base:
                merged[key] = copy.deepcopy(value)
            elif value != base[key]:
                merged[key] = _three_way(live.get(key), value, base[key])
        return merged
    if isinstance(desired, list) and isinstance(base, list) and isinstance(live, list):
        removed = [item for item in base if item not in desired]
        kept = [item for item in live if item not in removed]
        return kept + [copy.deepcopy(item) for item in desired if item not in base and item not in kept]
    return copy.deepcopy(desired)
