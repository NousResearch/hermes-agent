"""Clio-specific execution profile helpers.

This module keeps the Buidl MVP execution mode as explicit runtime
configuration instead of baking it into the global Hermes system prompt.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

PROFILE_NAME = "clio-mvp-execution-v1"
PROFILE_ENV_VAR = "CLIO_EXECUTION_PROFILE"
ANTHROPIC_MODEL_ENV_VAR = "CLIO_ANTHROPIC_MODEL"

_PROFILE_PATH = Path(__file__).resolve().parent / "prompts" / "clio" / f"{PROFILE_NAME}.md"
_REPORT_LABELS = frozenset({"GREEN", "RED", "NOISE"})


def configured_clio_profile(config: Mapping[str, Any] | None = None) -> str:
    """Return the requested Clio execution profile name, if any."""
    env_profile = os.getenv(PROFILE_ENV_VAR, "").strip()
    if env_profile:
        return env_profile
    agent_cfg = (config or {}).get("agent", {}) if isinstance(config, Mapping) else {}
    if isinstance(agent_cfg, Mapping):
        return str(
            agent_cfg.get("clio_profile")
            or agent_cfg.get("execution_profile")
            or agent_cfg.get("profile")
            or ""
        ).strip()
    return ""


def is_clio_mvp_execution_enabled(config: Mapping[str, Any] | None = None) -> bool:
    """Return True when the Buidl-specific Clio MVP profile is selected."""
    return configured_clio_profile(config) == PROFILE_NAME


def load_clio_execution_profile(profile_name: str) -> str:
    """Load a bundled Clio execution profile by name."""
    if profile_name != PROFILE_NAME:
        return ""
    try:
        return _PROFILE_PATH.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def append_clio_execution_profile(
    base_prompt: str | None,
    config: Mapping[str, Any] | None = None,
) -> str:
    """Append the configured Clio execution profile to an existing prompt."""
    profile_name = configured_clio_profile(config)
    profile_text = load_clio_execution_profile(profile_name)
    base = (base_prompt or "").strip()
    if not profile_text:
        return base
    if base:
        return f"{base}\n\n{profile_text}"
    return profile_text


def resolve_clio_anthropic_model(
    config: Mapping[str, Any] | None = None,
    *,
    env: Mapping[str, str] | None = None,
    current_model: str = "",
) -> str:
    """Resolve Clio's Anthropic model from env first, then config, then fallback.

    This helper returns only a model name. It never reads, returns or prints API
    keys. ``env`` is injectable so tests can prove CLIO_ANTHROPIC_MODEL drives
    optional Claude Fable 5 selection without requiring that model.
    """
    source = env if env is not None else os.environ
    override = str(source.get(ANTHROPIC_MODEL_ENV_VAR, "")).strip()
    if override:
        return override
    clio_cfg = (config or {}).get("clio", {}) if isinstance(config, Mapping) else {}
    if isinstance(clio_cfg, Mapping):
        configured = str(clio_cfg.get("anthropic_model") or "").strip()
        if configured:
            return configured
    return current_model


def apply_clio_anthropic_model_override(
    model: str,
    provider: str | None,
    config: Mapping[str, Any] | None = None,
) -> tuple[str, str | None]:
    """Apply CLIO_ANTHROPIC_MODEL only for enabled native Anthropic Clio sessions.

    Returns ``(model, startup_notice)``. The notice intentionally contains only
    the selected model name and never includes API keys, tokens or env values.
    """
    current = str(model or "")
    if not is_clio_mvp_execution_enabled(config):
        return current, None
    if str(provider or "").strip().lower() != "anthropic":
        return current, None
    override = os.getenv(ANTHROPIC_MODEL_ENV_VAR, "").strip()
    if not override:
        return current, f"Clio MVP execution profile active with Anthropic model: {current or '(config default)'}"
    return override, f"Clio MVP execution profile active with Anthropic model: {override}"


def classify_clio_report(label: str) -> str:
    """Validate and normalize a Clio MVP report label."""
    normalized = str(label or "").strip().upper()
    if normalized not in _REPORT_LABELS:
        raise ValueError("Clio report label must be GREEN, RED, or NOISE")
    return normalized


def clio_profile_path() -> Path:
    """Expose profile path for tests and diagnostics."""
    return _PROFILE_PATH
