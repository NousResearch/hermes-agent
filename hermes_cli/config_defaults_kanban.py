"""Pure-data defaults for opt-in dispatcher diagnostics and admission."""


def dispatch_defaults():
    return {
        "provider_lanes": {
            "enabled": False,
            "router_path": "~/.agents/model-monarchy/models.toml",
            "minimum_free_bytes": 2 * 1024**3,
            "worker_headroom_bytes": 512 * 1024**2,
        },
        "dispatch_scheduling": {
            "enabled": False,
            "aging_seconds": 900,
            "maximum_bonus": 20,
            "stall_seconds": 3600,
            "oldest_limit": 10,
        },
    }
