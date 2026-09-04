"""Canonical effective configuration for the generic webhook listener.

The gateway runtime remains the single merge authority.  This module owns the
webhook-specific environment projection and exposes a value-safe read model so
management consumers do not reconstruct a second precedence chain.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Literal, Mapping, Optional

from agent.secret_scope import current_secret_scope
from hermes_constants import (
    get_hermes_home,
    reset_hermes_home_override,
    set_hermes_home_override,
)


WebhookSource = Literal["default", "yaml", "env", "profile"]

DEFAULT_WEBHOOK_ENABLED = False
DEFAULT_WEBHOOK_HOST: str | None = None
DEFAULT_WEBHOOK_PORT = 8644
DEFAULT_WEBHOOK_ROUTES_FILENAME = "webhook_subscriptions.json"

_SOURCE_MAP_KEY = "_webhook_effective_sources"
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off"})


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


def _env_source(name: str) -> WebhookSource:
    scope = current_secret_scope()
    if scope is not None and name in scope:
        return "profile"
    return "env"


def _parse_enabled(raw: object) -> Optional[bool]:
    normalized = str(raw).strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    return None


def _configured_sources(platform_config: object | None) -> dict[str, WebhookSource]:
    sources: dict[str, WebhookSource] = {
        "enabled": "default",
        "host": "default",
        "port": "default",
        "global_secret_ref": "default",
        "routes_path": "profile",
    }
    if platform_config is None:
        return sources

    extra = getattr(platform_config, "extra", None)
    if not isinstance(extra, dict):
        return sources
    if extra.get("_enabled_explicit"):
        sources["enabled"] = "yaml"
    for field in ("host", "port"):
        if field in extra:
            sources[field] = "yaml"
    if "secret" in extra or "secret_ref" in extra:
        sources["global_secret_ref"] = "yaml"
    return sources


def apply_webhook_env_overrides(
    config: object,
    *,
    getenv: Callable[[str, Optional[str]], Optional[str]],
) -> None:
    """Apply legacy ``WEBHOOK_*`` inputs to one ``GatewayConfig`` in place.

    This function is invoked by :func:`gateway.config.load_gateway_config`.
    Keeping the projection here means runtime and management surfaces share the
    exact same precedence and boolean semantics.  Provenance stores only source
    labels; secret values never enter the public read model.
    """
    from gateway.config import Platform, PlatformConfig

    platforms = getattr(config, "platforms")
    platform_config = platforms.get(Platform.WEBHOOK)
    sources = _configured_sources(platform_config)

    raw_enabled = getenv("WEBHOOK_ENABLED", None)
    enabled_override = (
        _parse_enabled(raw_enabled) if raw_enabled is not None else None
    )
    if enabled_override is not None:
        if platform_config is None and enabled_override:
            platform_config = PlatformConfig()
            platforms[Platform.WEBHOOK] = platform_config
        if platform_config is not None:
            platform_config.enabled = enabled_override
            sources["enabled"] = _env_source("WEBHOOK_ENABLED")

    if platform_config is None:
        return

    raw_host = getenv("WEBHOOK_HOST", None)
    if raw_host is not None:
        platform_config.extra["host"] = str(raw_host).strip() or None
        sources["host"] = _env_source("WEBHOOK_HOST")

    raw_port = getenv("WEBHOOK_PORT", None)
    if raw_port is not None:
        try:
            platform_config.extra["port"] = int(str(raw_port).strip(), 10)
        except (TypeError, ValueError):
            pass
        else:
            sources["port"] = _env_source("WEBHOOK_PORT")

    raw_secret = getenv("WEBHOOK_SECRET", None)
    if raw_secret is not None and str(raw_secret):
        platform_config.extra["secret"] = str(raw_secret)
        sources["global_secret_ref"] = _env_source("WEBHOOK_SECRET")

    platform_config.extra[_SOURCE_MAP_KEY] = dict(sources)


@contextmanager
def _profile_config_scope(profile: str) -> Iterator[None]:
    """Temporarily bind a management read to one existing profile home.

    The override is context-local, so concurrent profile reads cannot mutate
    process-global ``os.environ``. The previous scope is restored even when
    configuration loading fails.
    """
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

    token = set_hermes_home_override(get_profile_dir(normalized))
    try:
        yield
    finally:
        reset_hermes_home_override(token)


def resolve_effective_webhook_config(
    profile: str | None = None,
) -> EffectiveWebhookConfig:
    """Read effective webhook settings for the active or named profile.

    Runtime callers normally resolve the already-active scope. Management
    callers may name an existing profile; that read is bound through Hermes'
    context-local home override and the previous scope is restored before
    returning. No process-global environment mutation is used.
    """
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

    raw_sources = extra.get(_SOURCE_MAP_KEY)
    sources = _configured_sources(platform_config)
    if isinstance(raw_sources, dict):
        for field in sources:
            value = raw_sources.get(field)
            if value in {"default", "yaml", "env", "profile"}:
                sources[field] = value

    host_value = extra.get("host", DEFAULT_WEBHOOK_HOST)
    host = str(host_value).strip() if host_value else None
    try:
        port = int(extra.get("port", DEFAULT_WEBHOOK_PORT))
    except (TypeError, ValueError):
        port = DEFAULT_WEBHOOK_PORT

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

    home = get_hermes_home()
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
        routes_path=home / DEFAULT_WEBHOOK_ROUTES_FILENAME,
        source_map=sources,
    )


def resolve_effective_webhook_secret() -> str:
    """Return the active profile's global HMAC secret for runtime auth only.

    Management callers must use :func:`resolve_effective_webhook_config`, whose
    value-safe model cannot contain this value.  Runtime callers enter the same
    profile scope as the adapter before calling this accessor, so the canonical
    gateway loader preserves multiplex isolation.
    """
    from gateway.config import Platform, load_gateway_config

    config = load_gateway_config()
    platform_config = config.platforms.get(Platform.WEBHOOK)
    if platform_config is None or not isinstance(platform_config.extra, dict):
        return ""
    raw_secret = platform_config.extra.get("secret")
    return str(raw_secret) if raw_secret is not None else ""
