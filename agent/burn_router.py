"""Optional Burn/Rust tool-router integration for Hermes Agent.

This module is deliberately conservative. It shells out to a local
`hermes-burn-tool-router` binary and returns an advisory route result. It never
raises to callers and never hard-gates tools by itself; callers can choose to log
observe-only predictions or use high-confidence `enabled_toolsets` hints.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from dataclasses import dataclass, field
from typing import Any, Mapping

logger = logging.getLogger(__name__)

CATEGORY_TOOLSETS: dict[str, list[str]] = {
    "terminal": ["terminal", "code_execution"],
    "file": ["file"],
    "web": ["web"],
    "x_search": ["x_search"],
    "browser": ["browser"],
    "memory": ["memory", "session_search"],
    "skills": ["skills"],
    "delegation": ["delegation"],
    "media_generation": ["image_gen", "video_gen", "tts"],
    "media_analysis": ["vision", "video"],
    "messaging": ["messaging"],
    "cron": ["cronjob"],
    "hermes_cli": [],
    "todo": ["todo"],
    "smart_home": ["homeassistant"],
    "kanban": ["kanban"],
    "social_platforms": ["discord", "discord_admin", "yuanbao"],
    "productivity": ["feishu_doc", "feishu_drive", "spotify"],
    "computer_use": ["computer_use"],
}


@dataclass(frozen=True)
class BurnRouterConfig:
    """Runtime config for the optional Burn router sidecar."""

    enabled: bool = False
    mode: str = "observe"  # observe | hint | narrow (narrow is treated like hint here)
    binary: str | None = None
    model: str | None = None
    confidence_threshold: float = 0.72
    timeout_seconds: float = 0.25

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None) -> "BurnRouterConfig":
        routing_cfg = (config or {}).get("routing", {}) if isinstance(config, Mapping) else {}
        burn_cfg = routing_cfg.get("burn_router", {}) if isinstance(routing_cfg, Mapping) else {}
        if not isinstance(burn_cfg, Mapping):
            burn_cfg = {}

        def env_bool(name: str, default: bool) -> bool:
            raw = os.getenv(name)
            if raw is None:
                return default
            return raw.strip().lower() in {"1", "true", "yes", "on"}

        return cls(
            enabled=env_bool("HERMES_BURN_ROUTER_ENABLED", bool(burn_cfg.get("enabled", False))),
            mode=str(os.getenv("HERMES_BURN_ROUTER_MODE", burn_cfg.get("mode", "observe"))).lower(),
            binary=os.getenv("HERMES_BURN_ROUTER_BINARY", burn_cfg.get("binary")),
            model=os.getenv("HERMES_BURN_ROUTER_MODEL", burn_cfg.get("model")),
            confidence_threshold=float(
                os.getenv("HERMES_BURN_ROUTER_CONFIDENCE", burn_cfg.get("confidence_threshold", 0.72))
            ),
            timeout_seconds=float(
                os.getenv("HERMES_BURN_ROUTER_TIMEOUT", burn_cfg.get("timeout_seconds", 0.25))
            ),
        )


@dataclass(frozen=True)
class BurnRouterResult:
    """Advisory route prediction returned by the Burn router."""

    category: str
    confidence: float
    time_us: float | None = None
    probabilities: dict[str, float] = field(default_factory=dict)
    enabled_toolsets: list[str] = field(default_factory=list)
    mode: str = "observe"
    raw: dict[str, Any] = field(default_factory=dict)

    def to_log_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "confidence": self.confidence,
            "time_us": self.time_us,
            "enabled_toolsets": self.enabled_toolsets,
            "mode": self.mode,
        }


def _mode_for_prediction(cfg: BurnRouterConfig, confidence: float) -> str:
    if cfg.mode == "observe":
        return "observe"
    if confidence >= cfg.confidence_threshold:
        return cfg.mode if cfg.mode in {"hint", "narrow"} else "hint"
    return "fallback_full_surface"


def observe_burn_router_turn(message: str, config: BurnRouterConfig | Mapping[str, Any] | None = None) -> BurnRouterResult | None:
    """Run the optional router once for observability and log the result.

    This is the first safe integration point for live Hermes: it records what
    the local Burn router *would* have chosen without changing the tool surface
    or model request. It intentionally swallows all failures via
    `get_burn_router_hint()` so a missing/slow sidecar cannot break a user turn.
    """

    result = get_burn_router_hint(message, config)
    if result is None:
        return None
    logger.info("burn_router prediction: %s", result.to_log_dict())
    return result


def get_burn_router_hint(message: str, config: BurnRouterConfig | Mapping[str, Any] | None = None) -> BurnRouterResult | None:
    """Return an advisory Burn router prediction, or None on disabled/failure.

    The caller owns policy. In observe mode this returns category/confidence with
    no `enabled_toolsets`. In hint/narrow modes it only returns toolsets when the
    prediction clears the confidence threshold. Any subprocess error, timeout,
    malformed JSON, or missing binary/model is a safe fallback (`None`).
    """

    cfg = config if isinstance(config, BurnRouterConfig) else BurnRouterConfig.from_config(config)
    if not cfg.enabled:
        return None
    if not cfg.binary or not cfg.model:
        logger.debug("Burn router enabled but binary/model missing; skipping")
        return None

    try:
        completed = subprocess.run(
            [cfg.binary, "predict", message, cfg.model],
            check=False,
            capture_output=True,
            text=True,
            timeout=cfg.timeout_seconds,
        )
    except Exception as exc:
        logger.debug("Burn router invocation failed: %s", exc)
        return None

    if completed.returncode != 0:
        logger.debug("Burn router exited %s: %s", completed.returncode, completed.stderr.strip())
        return None

    try:
        payload = json.loads(completed.stdout)
        category = str(payload["category"])
        confidence = float(payload["confidence"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        logger.debug("Burn router returned malformed JSON: %s", exc)
        return None

    mode = _mode_for_prediction(cfg, confidence)
    enabled_toolsets = CATEGORY_TOOLSETS.get(category, []) if mode in {"hint", "narrow"} else []
    return BurnRouterResult(
        category=category,
        confidence=confidence,
        time_us=float(payload["time_us"]) if payload.get("time_us") is not None else None,
        probabilities=dict(payload.get("all") or {}),
        enabled_toolsets=list(enabled_toolsets),
        mode=mode,
        raw=payload,
    )
