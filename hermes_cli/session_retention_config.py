"""Shared CLI/gateway opt-in gates for state.db retention."""

from hermes_state_maintenance import _normalize_source_retention


def auto_prune_kwargs(sessions_cfg) -> dict | None:
    raw = sessions_cfg.get("source_retention_days")
    global_prune = bool(sessions_cfg.get("auto_prune", False))
    if not global_prune and not _normalize_source_retention(raw, warn=False):
        return None
    return dict(retention_days=int(sessions_cfg.get("retention_days", 90)) if global_prune else 90,
                min_interval_hours=int(sessions_cfg.get("min_interval_hours", 24)),
                min_vacuum_interval_days=int(sessions_cfg.get("min_vacuum_interval_days", 30)),
                vacuum=bool(sessions_cfg.get("vacuum_after_prune", True)), global_prune=global_prune, source_retention_days=raw)
