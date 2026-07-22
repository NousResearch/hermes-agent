from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Mapping

import yaml


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SessionPolicy:
    trigram_enabled: bool = True
    auto_prune: bool = False
    retention_days: float = 90.0
    retention_days_by_source: Mapping[str, float] = field(default_factory=dict)
    vacuum_after_prune: bool = True
    min_interval_hours: float = 24.0


def _positive_number(value, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _source_retention(value) -> Dict[str, float]:
    if not isinstance(value, dict):
        return {}
    result: Dict[str, float] = {}
    for raw_source, raw_days in value.items():
        source = str(raw_source or "").strip().lower()
        if not source:
            continue
        try:
            days = float(raw_days)
        except (TypeError, ValueError):
            continue
        if days > 0:
            result[source] = days
    return result


def _strict_bool(value, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "on", "1"}:
            return True
        if normalized in {"false", "no", "off", "0"}:
            return False
    return default


def load_session_policy(hermes_home: Path) -> SessionPolicy:
    sessions = {}
    config_path = Path(hermes_home) / "config.yaml"
    try:
        if config_path.exists():
            loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            if not isinstance(loaded, dict):
                raise ValueError("config root is not a mapping")
            if "sessions" in loaded and not isinstance(loaded["sessions"], dict):
                raise ValueError("sessions config is not a mapping")
            sessions = loaded.get("sessions") or {}
    except Exception as exc:
        logger.warning(
            "Could not read session policy from %s (%s)",
            config_path,
            type(exc).__name__,
        )
        return SessionPolicy(
            trigram_enabled=False,
            auto_prune=False,
            vacuum_after_prune=False,
        )

    trigram_enabled = (
        _strict_bool(sessions["trigram_enabled"], False)
        if "trigram_enabled" in sessions
        else True
    )
    if os.environ.get("HERMES_DISABLE_FTS_TRIGRAM", "").strip().lower() in {
        "1", "true", "yes", "on",
    }:
        trigram_enabled = False

    return SessionPolicy(
        trigram_enabled=trigram_enabled,
        auto_prune=_strict_bool(sessions.get("auto_prune"), False),
        retention_days=_positive_number(sessions.get("retention_days"), 90.0),
        retention_days_by_source=_source_retention(
            sessions.get("retention_days_by_source")
        ),
        vacuum_after_prune=(
            _strict_bool(sessions["vacuum_after_prune"], False)
            if "vacuum_after_prune" in sessions
            else True
        ),
        min_interval_hours=_positive_number(
            sessions.get("min_interval_hours"), 24.0
        ),
    )
