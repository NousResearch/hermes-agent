"""Configuration helpers for the completion-auditor plugin."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home

SCHEMA = "hermes-completion-audit-v1"
_DEFAULT_MAX_RESULT_EXCERPT_CHARS = 800
_DEFAULT_LOG_RETENTION_DAYS = 7
_DEFAULT_MAX_LOG_SIZE_MB = 10


def _default_log_dir() -> Path:
    return get_hermes_home() / "logs" / "completion-auditor"


@dataclass(frozen=True)
class AuditorConfig:
    """Runtime settings for audit-only completion auditing."""

    mode: str = "audit"
    log_verdicts: bool = True
    log_dir: Path = field(default_factory=_default_log_dir)
    include_tool_result_excerpt: bool = False
    max_result_excerpt_chars: int = _DEFAULT_MAX_RESULT_EXCERPT_CHARS
    redact_secrets: bool = True
    log_retention_days: int = _DEFAULT_LOG_RETENTION_DAYS
    max_log_size_mb: int = _DEFAULT_MAX_LOG_SIZE_MB

    @property
    def audit_enabled(self) -> bool:
        return self.mode == "audit" and self.log_verdicts


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _cfg_get(config: dict[str, Any], key: str, default: Any) -> Any:
    plugins_cfg = config.get("plugins", {})
    if not isinstance(plugins_cfg, dict):
        return default
    underscore_cfg = plugins_cfg.get("completion_auditor", {})
    hyphen_cfg = plugins_cfg.get("completion-auditor", {})
    if not isinstance(underscore_cfg, dict):
        underscore_cfg = {}
    if not isinstance(hyphen_cfg, dict):
        hyphen_cfg = {}
    # Prefer the documented Python-friendly key, but accept the plugin id as an
    # alias because users naturally copy it from `hermes plugins enable`.
    merged = {**hyphen_cfg, **underscore_cfg}
    return merged.get(key, default)


def _load_config() -> dict[str, Any]:
    try:
        from hermes_cli.config import load_config

        loaded = load_config()
        return loaded if isinstance(loaded, dict) else {}
    except Exception:
        return {}


def _resolve_log_dir(raw: Any) -> Path:
    if raw is None or str(raw).strip() == "":
        return _default_log_dir()
    expanded = Path(os.path.expandvars(os.path.expanduser(str(raw))))
    if not expanded.is_absolute():
        expanded = get_hermes_home() / expanded
    return expanded


def _positive_int(value: Any, default: int) -> int:
    try:
        return max(0, int(value))
    except Exception:
        return default


def load_settings() -> AuditorConfig:
    """Load profile-safe settings from config.yaml plus env test overrides.

    The plugin itself is opt-in through ``plugins.enabled``. These settings are
    a second, runtime-level configuration surface used only after the plugin has
    loaded.
    """
    config = _load_config()
    raw_log_dir = os.getenv("HERMES_COMPLETION_AUDITOR_LOG_DIR") or _cfg_get(
        config, "log_dir", None
    )
    mode = str(_cfg_get(config, "mode", "audit")).strip().lower() or "audit"
    # MVP is audit-only; fail closed to disabled semantics for future modes.
    if mode != "audit":
        mode = "unsupported"

    max_chars = _positive_int(
        _cfg_get(config, "max_result_excerpt_chars", _DEFAULT_MAX_RESULT_EXCERPT_CHARS),
        _DEFAULT_MAX_RESULT_EXCERPT_CHARS,
    )

    return AuditorConfig(
        mode=mode,
        log_verdicts=_truthy(_cfg_get(config, "log_verdicts", True)),
        log_dir=_resolve_log_dir(raw_log_dir),
        include_tool_result_excerpt=_truthy(
            _cfg_get(config, "include_tool_result_excerpt", False)
        ),
        max_result_excerpt_chars=max_chars,
        redact_secrets=_truthy(_cfg_get(config, "redact_secrets", True)),
        log_retention_days=_positive_int(
            _cfg_get(config, "log_retention_days", _DEFAULT_LOG_RETENTION_DAYS),
            _DEFAULT_LOG_RETENTION_DAYS,
        ),
        max_log_size_mb=_positive_int(
            _cfg_get(config, "max_log_size_mb", _DEFAULT_MAX_LOG_SIZE_MB),
            _DEFAULT_MAX_LOG_SIZE_MB,
        ),
    )
