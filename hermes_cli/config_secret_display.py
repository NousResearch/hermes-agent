"""Masking of credential-shaped config values before ``hermes config`` prints them (moved out of
``hermes_cli.config``, which re-exports ``redact_config_value`` for its callers)."""

from __future__ import annotations

from typing import Any


def _maskable_secret_text(value: Any) -> str:
    """Display text of a scalar that must be masked under a secret key, or ``""``. YAML and
    ``config set`` coerce an all-digit value (e.g. a numeric password) to ``int``, so numbers
    count too. Which keys are secret is decided by ``_is_secret_config_key``, not here."""
    if isinstance(value, bool):
        return ""
    if isinstance(value, (int, float)):
        return str(value)
    return value if isinstance(value, str) else ""


def redact_config_value(value: Any, _depth: int = 0) -> Any:
    """Copy of ``value`` with credential-shaped keys masked. ``print`` bypasses the logging
    redactor and opaque tokens miss the vendor-prefix regexes, so structural masking is required."""
    from agent.redact import mask_secret
    from hermes_cli.config import _ENV_PLACEHOLDER_RE, _is_secret_config_key

    if _depth > 20:  # bound recursion for pathological/cyclic configs
        return value
    if isinstance(value, dict):
        return {
            k: mask_secret(text)
            if isinstance(k, str) and _is_secret_config_key(k) and (text := _maskable_secret_text(v))
            and not _ENV_PLACEHOLDER_RE.match(text)
            else redact_config_value(v, _depth + 1)
            for k, v in value.items()}
    if isinstance(value, list):
        return [redact_config_value(v, _depth + 1) for v in value]
    return value
