from __future__ import annotations

import copy


DEFAULT_CONDUCTOR_CONFIG = {
    "enabled": False,
    "state_path": "conductor/state.sqlite",
    "tick_lease_seconds": 30,
    "writer": {"command": [], "provider": "", "model": "", "usage_reporting": "exact"},
    "reviewer": {
        "command": [],
        "provider": "",
        "model": "",
        "usage_reporting": "exact",
    },
    "budgets": {
        "max_processed_tokens_per_run": 200_000,
        "max_processed_tokens_per_day": 1_000_000,
        "max_conductor_turns": 100,
        "max_worker_turns": 90,
        "wall_time_seconds": 3600,
        "max_runs_per_day": 10,
        "max_retries": 2,
        "backoff_base_seconds": 2,
    },
}


def _merge(base: dict, override: dict) -> dict:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def resolve_conductor_config(config: dict) -> dict:
    """Resolve config.yaml's opt-in conductor block without runtime mutation."""
    resolved = _merge(
        copy.deepcopy(DEFAULT_CONDUCTOR_CONFIG), dict(config.get("conductor") or {})
    )
    for key, value in resolved["budgets"].items():
        if not isinstance(value, (int, float)) or value < 0:
            raise ValueError(f"conductor.budgets.{key} must be non-negative")
    if (
        resolved["budgets"]["max_processed_tokens_per_day"]
        < resolved["budgets"]["max_processed_tokens_per_run"]
    ):
        raise ValueError("conductor daily token budget must cover at least one run")
    lease = resolved["tick_lease_seconds"]
    if (
        not isinstance(lease, (int, float))
        or isinstance(lease, bool)
        or not 1 <= lease <= 3600
    ):
        raise ValueError("conductor.tick_lease_seconds must be between 1 and 3600")
    return resolved
