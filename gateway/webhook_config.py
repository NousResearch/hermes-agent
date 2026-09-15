"""Value-safe projection of the gateway's effective generic-webhook config.

The gateway loader remains the sole merge authority.  This module never applies
configuration itself; it reads the already-resolved ``GatewayConfig`` and adds
non-sensitive provenance for management callers.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal, Mapping

import yaml

from agent.secret_scope import (
    build_profile_secret_scope,
    current_secret_scope,
    reset_secret_scope,
    set_secret_scope,
)
from hermes_constants import (
    get_hermes_home,
    reset_hermes_home_override,
    set_hermes_home_override,
)
from utils import is_truthy_value


WebhookSource = Literal["default", "yaml", "env", "profile"]

DEFAULT_WEBHOOK_ENABLED = False
DEFAULT_WEBHOOK_HOST: str | None = None
DEFAULT_WEBHOOK_PORT = 8644
DEFAULT_WEBHOOK_ROUTES_FILENAME = "webhook_subscriptions.json"


@dataclass(frozen=True)
class EffectiveWebhookConfig:
    """Resolved listener settings plus non-sensitive provenance metadata."""

    enabled: bool
    host: str | None
    port: int
    profile: str
    global_secret_ref: str | None
    routes_path: Path
    source_map: Mapping[str, WebhookSource]


def _yaml_source_fields(home: Path) -> set[str]:
    """Return webhook fields explicitly present in this profile's YAML.

    Values are intentionally discarded.  The gateway loader owns value
    precedence; this helper exists only so the public read model can say which
    fields had an operator-authored YAML source without exposing secrets.
    """
    try:
        data = yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, UnicodeError, yaml.YAMLError):
        return set()
    if not isinstance(data, dict):
        return set()

    blocks: list[dict] = []
    platforms = data.get("platforms")
    if isinstance(platforms, dict) and isinstance(platforms.get("webhook"), dict):
        blocks.append(platforms["webhook"])

    gateway = data.get("gateway")
    if isinstance(gateway, dict):
        gateway_platforms = gateway.get("platforms")
        if (
            isinstance(gateway_platforms, dict)
            and isinstance(gateway_platforms.get("webhook"), dict)
        ):
            blocks.append(gateway_platforms["webhook"])
        if isinstance(gateway.get("webhook"), dict):
            blocks.append(gateway["webhook"])

    if isinstance(data.get("webhook"), dict):
        blocks.append(data["webhook"])

    fields: set[str] = set()
    for block in blocks:
        if "enabled" in block:
            fields.add("enabled")
        extra = block.get("extra")
        extra = extra if isinstance(extra, dict) else {}
        for name in ("host", "port"):
            if name in block or name in extra:
                fields.add(name)
        if any(name in block or name in extra for name in ("secret", "secret_ref")):
            fields.add("global_secret_ref")
    return fields


def _env_source(name: str) -> WebhookSource:
    scope = current_secret_scope()
    if scope is not None and scope.get(name) is not None:
        return "profile"
    return "env"


def _runtime_env(name: str) -> str | None:
    """Read through the exact scope-aware getenv used by gateway config."""
    from gateway.config import _getenv

    return _getenv(name, None)


def _valid_int(raw: object) -> bool:
    if raw is None:
        return False
    try:
        int(str(raw).strip(), 10)
    except (TypeError, ValueError):
        return False
    return True


def _source_map(home: Path) -> dict[str, WebhookSource]:
    yaml_fields = _yaml_source_fields(home)
    sources: dict[str, WebhookSource] = {
        "enabled": "yaml" if "enabled" in yaml_fields else "default",
        "host": "yaml" if "host" in yaml_fields else "default",
        "port": "yaml" if "port" in yaml_fields else "default",
        "global_secret_ref": (
            "yaml" if "global_secret_ref" in yaml_fields else "default"
        ),
        "routes_path": "profile",
    }

    # Current-main runtime semantics only enter the generic-webhook env branch
    # when WEBHOOK_ENABLED is truthy.  A falsy value does not disable YAML, and
    # an explicit YAML enabled:false remains authoritative.  Mirror those exact
    # semantics for provenance without creating a second value merge path.
    raw_enabled = _runtime_env("WEBHOOK_ENABLED")
    env_enables = raw_enabled is not None and is_truthy_value(raw_enabled)
    if env_enables and "enabled" not in yaml_fields:
        sources["enabled"] = _env_source("WEBHOOK_ENABLED")

    raw_port = _runtime_env("WEBHOOK_PORT")
    if env_enables and _valid_int(raw_port):
        sources["port"] = _env_source("WEBHOOK_PORT")

    raw_secret = _runtime_env("WEBHOOK_SECRET")
    if env_enables and raw_secret:
        sources["global_secret_ref"] = _env_source("WEBHOOK_SECRET")

    return sources


@contextmanager
def _profile_config_scope(profile: str) -> Iterator[None]:
    """Bind a management read to one existing profile without env mutation."""
    from hermes_cli.profiles import (
        get_profile_dir,
        normalize_profile_name,
        profile_exists,
        validate_profile_name,
    )

    normalized = normalize_profile_name(profile)
    validate_profile_name(normalized)
    if normalized != "default" and not profile_exists(normalized):
        raise FileNotFoundError(f"Profile {normalized!r} does not exist")

    home = get_profile_dir(normalized)
    home_token = set_hermes_home_override(home)
    secret_token = set_secret_scope(build_profile_secret_scope(home))
    try:
        yield
    finally:
        reset_secret_scope(secret_token)
        reset_hermes_home_override(home_token)


def resolve_effective_webhook_config(
    profile: str | None = None,
) -> EffectiveWebhookConfig:
    """Read the canonical effective webhook settings for an active/named profile."""
    if profile is not None:
        with _profile_config_scope(profile):
            return resolve_effective_webhook_config()

    from gateway.config import Platform, load_gateway_config
    from hermes_cli.profiles import get_active_profile_name

    config = load_gateway_config()
    platform_config = config.platforms.get(Platform.WEBHOOK)
    extra = (
        platform_config.extra
        if platform_config is not None and isinstance(platform_config.extra, dict)
        else {}
    )

    host_value = extra.get("host", DEFAULT_WEBHOOK_HOST)
    host = str(host_value).strip() if host_value else None
    try:
        port = int(extra.get("port", DEFAULT_WEBHOOK_PORT))
    except (TypeError, ValueError):
        port = DEFAULT_WEBHOOK_PORT

    sources = _source_map(get_hermes_home())
    secret_ref = extra.get("secret_ref")
    if isinstance(secret_ref, str):
        secret_ref = secret_ref.strip() or None
    else:
        secret_ref = None
    if (
        secret_ref is None
        and extra.get("secret")
        and sources["global_secret_ref"] in {"env", "profile"}
    ):
        secret_ref = "WEBHOOK_SECRET"

    return EffectiveWebhookConfig(
        enabled=(
            bool(platform_config.enabled)
            if platform_config is not None
            else DEFAULT_WEBHOOK_ENABLED
        ),
        host=host,
        port=port,
        profile=get_active_profile_name(),
        global_secret_ref=secret_ref,
        routes_path=get_hermes_home() / DEFAULT_WEBHOOK_ROUTES_FILENAME,
        source_map=sources,
    )


def resolve_effective_webhook_secret() -> str:
    """Return the active profile's global HMAC secret for runtime auth only."""
    from gateway.config import Platform, load_gateway_config

    config = load_gateway_config()
    platform_config = config.platforms.get(Platform.WEBHOOK)
    if platform_config is None or not isinstance(platform_config.extra, dict):
        return ""
    raw_secret = platform_config.extra.get("secret")
    return str(raw_secret) if raw_secret is not None else ""
