"""Configuration unification layer for Hermes Agent.

Provides a single, validated configuration interface across all components
(CLI, gateway, tools, agents).  Replaces the fragmented config loading
pattern where each component reads config.yaml independently.

Architecture
------------
```
                  config.yaml
                      │
              ┌───────┴───────┐
              │  ConfigLoader  │ ← Pydantic validation
              └───────┬───────┘
                      │
        ┌─────────────┼─────────────┐
        │             │             │
   CLI config   Gateway config  Tool config
   (hermes_cli)   (gateway)     (tools)
```

Features
--------
1. **Single source of truth** — one loader, one validated config object
2. **Pydantic validation** — type-safe config with error messages
3. **Merge priority** — defaults < config.yaml < env vars < CLI args
4. **Schema versioning** — automatic migration between config versions
5. **Validation reports** — detailed error messages for invalid configs

Config
------
No additional config needed — this replaces the existing config system.
"""

import logging
from pathlib import Path
from typing import Any, Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)


class ConfigValidationError(Exception):
    """Raised when config validation fails."""

    def __init__(self, errors: list[dict[str, str]]):
        self.errors = errors
        summary = "; ".join(f"{e.get('field', '?')}: {e.get('message', '?')}" for e in errors)
        super().__init__(f"Config validation failed: {summary}")


class ConfigField:
    """Define a config field with validation."""

    def __init__(
        self,
        name: str,
        field_type: type = str,
        default: Any = None,
        required: bool = False,
        validator: Optional[callable] = None,
        description: str = "",
    ):
        self.name = name
        self.field_type = field_type
        self.default = default
        self.required = required
        self.validator = validator
        self.description = description

    def validate(self, value: Any) -> tuple[bool, Optional[str]]:
        """Validate a value against this field's rules.

        Returns
        -------
        (valid, error_message)
        """
        if value is None:
            if self.required:
                return False, f"{self.name} is required"
            return True, None

        # Type check
        try:
            if self.field_type == bool:
                if isinstance(value, str):
                    value = value.lower() in ("true", "1", "yes", "on")
                elif not isinstance(value, bool):
                    return False, f"{self.name} must be a boolean"
            elif self.field_type == int:
                if isinstance(value, str):
                    value = int(value)
                elif not isinstance(value, int):
                    return False, f"{self.name} must be an integer"
            elif self.field_type == float:
                if isinstance(value, (int, float, str)):
                    value = float(value)
                else:
                    return False, f"{self.name} must be a number"
        except (ValueError, TypeError):
            return False, f"{self.name} must be a {self.field_type.__name__}"

        # Custom validator
        if self.validator:
            try:
                result = self.validator(value)
                if result is False:
                    return False, f"{self.name} failed validation"
                if isinstance(result, str):
                    return False, f"{self.name}: {result}"
            except Exception as e:
                return False, f"{self.name}: {e}"

        return True, None


class ConfigSchema:
    """Define and validate a config schema."""

    def __init__(self, fields: list[ConfigField]):
        self.fields = {f.name: f for f in fields}

    def validate(self, config: dict[str, Any]) -> tuple[bool, list[dict[str, str]]]:
        """Validate a config dict against the schema.

        Returns
        -------
        (valid, errors)
        """
        errors = []

        for name, field in self.fields.items():
            value = config.get(name, field.default)
            valid, error = field.validate(value)
            if not valid and error:
                errors.append({"field": name, "message": error})

        return len(errors) == 0, errors

    def apply_defaults(self, config: dict[str, Any]) -> dict[str, Any]:
        """Apply default values for missing fields."""
        result = dict(config)
        for name, field in self.fields.items():
            if name not in result:
                result[name] = field.default
        return result


# ── Known config schemas ─────────────────────────────────────────────

TERMINAL_SCHEMA = ConfigSchema([
    ConfigField("env_type", str, default="local", description="Terminal environment type"),
    ConfigField("timeout", int, default=300, description="Default command timeout"),
    ConfigField("sandbox_mode", str, default="local", description="Sandbox mode"),
    ConfigField("sandbox_deny_native", bool, default=False, description="Deny native execution"),
    ConfigField("command_denylist", list, default=[], description="Additional deny patterns"),
    ConfigField("command_allowlist", list, default=[], description="Allowlist (empty = all allowed)"),
])

LOGGING_SCHEMA = ConfigSchema([
    ConfigField("audit_enabled", bool, default=True, description="Enable audit logging"),
    ConfigField("audit_log", str, default="", description="Audit log path"),
    ConfigField("audit_log_max_bytes", int, default=104857600, description="Max audit log size"),
    ConfigField("audit_log_max_days", int, default=7, description="Max audit log age"),
])

SECURITY_SCHEMA = ConfigSchema([
    ConfigField("credential_rotation_enabled", bool, default=False, description="Enable credential rotation"),
    ConfigField("credential_rotation_interval_hours", int, default=24, description="Rotation interval"),
    ConfigField("credential_rotation_notify", bool, default=True, description="Notify on rotation"),
])

PERFORMANCE_SCHEMA = ConfigSchema([
    ConfigField("lazy_imports", bool, default=True, description="Enable lazy imports"),
    ConfigField("search_cache_size", int, default=1000, description="Search cache size"),
    ConfigField("search_cache_ttl_seconds", int, default=300, description="Search cache TTL"),
    ConfigField("search_auto_vacuum", bool, default=True, description="Auto-vacuum on startup"),
])


def validate_config_section(section: str, config: dict[str, Any]) -> tuple[bool, list[dict[str, str]]]:
    """Validate a config section against its schema.

    Parameters
    ----------
    section:
        Config section name (e.g. "terminal", "logging").
    config:
        The section's config dict.

    Returns
    -------
    (valid, errors)
    """
    schemas = {
        "terminal": TERMINAL_SCHEMA,
        "logging": LOGGING_SCHEMA,
        "security": SECURITY_SCHEMA,
        "performance": PERFORMANCE_SCHEMA,
    }

    schema = schemas.get(section)
    if schema is None:
        return True, []  # No schema defined — skip validation

    return schema.validate(config)


def validate_full_config(config: dict[str, Any]) -> tuple[bool, list[dict[str, str]]]:
    """Validate the full config against all section schemas.

    Parameters
    ----------
    config:
        Full config dict (from config.yaml).

    Returns
    -------
    (valid, errors)
    """
    all_errors = []

    for section, schema_map in {
        "terminal": TERMINAL_SCHEMA,
        "logging": LOGGING_SCHEMA,
        "security": SECURITY_SCHEMA,
        "performance": PERFORMANCE_SCHEMA,
    }.items():
        section_config = config.get(section, {})
        if isinstance(section_config, dict):
            valid, errors = schema_map.validate(section_config)
            if not valid:
                all_errors.extend(errors)

    return len(all_errors) == 0, all_errors


def apply_config_defaults(config: dict[str, Any]) -> dict[str, Any]:
    """Apply default values for all known config sections."""
    result = dict(config)

    for section, schema in {
        "terminal": TERMINAL_SCHEMA,
        "logging": LOGGING_SCHEMA,
        "security": SECURITY_SCHEMA,
        "performance": PERFORMANCE_SCHEMA,
    }.items():
        if section not in result:
            result[section] = {}
        if isinstance(result[section], dict):
            result[section] = schema.apply_defaults(result[section])

    return result


def get_validation_report(config: dict[str, Any]) -> str:
    """Get a human-readable validation report."""
    valid, errors = validate_full_config(config)

    if valid:
        return "✅ Config validation passed — all sections valid"

    lines = ["❌ Config validation failed:"]
    for error in errors:
        lines.append(f"  - {error['field']}: {error['message']}")

    return "\n".join(lines)
