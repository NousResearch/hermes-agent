"""Unified effective configuration for the generic webhook listener.

This module is deliberately value-aware only at resolution time.  Callers that
need to display configuration can inspect ``source_map`` without printing any
secret value.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping

from agent.secret_scope import current_secret_scope
from hermes_cli.config import load_config_readonly
from hermes_cli.env_loader import get_secret_source_values
from hermes_cli.profiles import get_profile_dir



WebhookSource = Literal["default", "yaml", "env", "profile"]

DEFAULT_WEBHOOK_ENABLED = False
DEFAULT_WEBHOOK_HOST: str | None = None
DEFAULT_WEBHOOK_PORT = 8644
_DEFAULT_ROUTES_FILENAME = "webhook_subscriptions.json"


@dataclass(frozen=True)
class EffectiveWebhookConfig:
    """The resolved listener settings and non-sensitive provenance metadata."""

    enabled: bool
    host: str | None
    port: int
    profile: str
    global_secret_ref: str | None
    routes_path: Path
    source_map: Mapping[str, WebhookSource]


def _as_mapping(value: object) -> dict:
    return value if isinstance(value, dict) else {}


def _bool_value(value: object, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _int_value(value: object, default: int) -> int:
    try:
        return int(str(value).strip(), 10)
    except (TypeError, ValueError):
        return default


def _yaml_webhook(home: Path) -> dict:
    """Read webhook platform config through the approved config-loading seam.

    ``load_config_readonly`` is the canonical owner for behavioral reads; it
    applies the managed-scope overlay, ``${ENV_VAR}`` expansion, profile-aware
    pathing, and root-model normalization. A raw ``yaml.safe_load`` here would
    trip the config-read-guard lint (raw reads only allowed in owner modules).
    """
    try:
        data = load_config_readonly() or {}
    except Exception:
        return {}
    platforms = _as_mapping(data).get("platforms")
    webhook = _as_mapping(_as_mapping(platforms).get("webhook"))
    extra = _as_mapping(webhook.get("extra"))
    # Accept the documented platform fields and the legacy adapter shape.
    result = dict(extra)
    result.update({key: webhook[key] for key in ("enabled", "host", "port", "secret", "secret_ref", "routes_path") if key in webhook})
    return result


def _profile_env(home: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    try:
        from agent.secret_scope import build_profile_secret_scope

        values.update(build_profile_secret_scope(home))
    except Exception:
        pass
    try:
        values.update(get_secret_source_values(home))
    except Exception:
        pass
    return values


def _env_value(name: str, profile_env: Mapping[str, str], scope: Mapping[str, str] | None) -> tuple[str | None, WebhookSource | None]:
    if scope is not None and name in scope:
        return scope[name], "profile"
    if name in profile_env:
        return profile_env[name], "profile"
    if name in os.environ:
        return os.environ[name], "env"
    return None, None


def resolve_effective_webhook_config(profile: str = "default") -> EffectiveWebhookConfig:
    """Resolve defaults, profile YAML, then profile/env webhook settings.

    ``global_secret_ref`` is a reference name (normally ``WEBHOOK_SECRET``),
    never the resolved secret.  Profile ``.env``/secret-scope values are marked
    ``profile``; process environment values are marked ``env``.
    """
    home = get_profile_dir(profile)
    yaml_values = _yaml_webhook(home)
    profile_env = _profile_env(home)
    scope = current_secret_scope()

    enabled = DEFAULT_WEBHOOK_ENABLED
    host = DEFAULT_WEBHOOK_HOST
    port = DEFAULT_WEBHOOK_PORT
    secret_ref: str | None = None
    source: dict[str, WebhookSource] = {
        "enabled": "default",
        "host": "default",
        "port": "default",
        "global_secret_ref": "default",
        "routes_path": "profile" if profile != "default" else "default",
    }

    if "enabled" in yaml_values:
        enabled = _bool_value(yaml_values["enabled"], enabled)
        source["enabled"] = "yaml"
    if "host" in yaml_values:
        host = str(yaml_values["host"]).strip() or None
        source["host"] = "yaml"
    if "port" in yaml_values:
        port = _int_value(yaml_values["port"], port)
        source["port"] = "yaml"
    if yaml_values.get("secret_ref") or yaml_values.get("secret"):
        secret_ref = str(yaml_values.get("secret_ref") or "WEBHOOK_SECRET")
        source["global_secret_ref"] = "yaml"
    routes_path = home / str(yaml_values.get("routes_path") or _DEFAULT_ROUTES_FILENAME)
    if yaml_values.get("routes_path"):
        source["routes_path"] = "yaml"

    for field, env_name in (("enabled", "WEBHOOK_ENABLED"), ("host", "WEBHOOK_HOST"), ("port", "WEBHOOK_PORT")):
        raw, origin = _env_value(env_name, profile_env, scope)
        if raw is None:
            continue
        if field == "enabled":
            enabled = _bool_value(raw, enabled)
        elif field == "host":
            host = str(raw).strip() or None
        else:
            port = _int_value(raw, port)
        source[field] = origin  # type: ignore[assignment]

    raw_secret, secret_origin = _env_value("WEBHOOK_SECRET", profile_env, scope)
    if raw_secret is not None and str(raw_secret).strip():
        # Expose only the reference identifier, never its value.
        secret_ref = "WEBHOOK_SECRET"
        source["global_secret_ref"] = secret_origin  # type: ignore[assignment]

    return EffectiveWebhookConfig(
        enabled=enabled,
        host=host,
        port=port,
        profile=profile,
        global_secret_ref=secret_ref,
        routes_path=routes_path,
        source_map=source,
    )


def resolve_effective_webhook_secret(profile: str = "default") -> str:
    """Resolve the global HMAC secret for runtime use without exposing it in config."""
    home = get_profile_dir(profile)
    yaml_values = _yaml_webhook(home)
    profile_env = _profile_env(home)
    scope = current_secret_scope()
    raw, _ = _env_value("WEBHOOK_SECRET", profile_env, scope)
    if raw is not None and str(raw).strip():
        return str(raw)
    yaml_secret = yaml_values.get("secret")
    return str(yaml_secret).strip() if yaml_secret else ""
