"""Freemaxxing model-provider plugin.

Discovery registers one opaque provider/model identity and opens only an
authenticated loopback listener.  Upstream credentials and the backend pool are
resolved lazily, after the first authenticated runtime request.  Multiplexed
profile runtimes are rejected before that resolution boundary; Freemaxxing is
single-profile-only until Hermes has a profile-addressed local-provider wire.
"""

from __future__ import annotations

import logging
import os
import secrets
import threading
from typing import Optional

from providers import register_provider
from providers.base import ProviderProfile

from .proxy import Backend, pool, spawn_proxy, stop_proxy as _stop_proxy

logger = logging.getLogger("freemaxxing")

_NOUS_BASE_URL = "https://inference-api.nousresearch.com/v1"
_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_DEFAULT_PORT = 11435
_ROUTER_MODEL = "freemaxxing"
_NOUS_FREE_MODEL = "deepseek/deepseek-v4-flash-0731"

# The loopback bearer is a capability, not an upstream credential.  It is
# regenerated for every process and never written to disk.
_LOCAL_TOKEN = secrets.token_urlsafe(32)
_listener_lock = threading.Lock()
_listener = None


def _multiplex_active() -> bool:
    try:
        from agent.secret_scope import is_multiplex_active

        return bool(is_multiplex_active())
    except Exception:
        return False


class _LocalCapabilityEnvVars(tuple):
    """Expose the process-local bearer only in a single-profile runtime.

    ``hermes_cli.auth`` snapshots plugin provider metadata into its generic
    API-key registry.  Freemaxxing's bearer is different from an upstream
    credential: it is a process-local capability for an authenticated loopback
    listener.  The generic resolver still needs to carry it to the OpenAI
    client in classic mode, but it must become invisible as soon as multiplex
    mode is active so one profile can never inherit another process-global
    capability.

    A tuple subclass preserves the registry's ordinary sequence contract while
    making iteration scope-aware at the exact secret-resolution boundary.
    """

    def __new__(cls):
        return super().__new__(cls, ("FREEMAXXING_API_KEY",))

    def __iter__(self):
        if _multiplex_active():
            return iter(())
        return super().__iter__()


def _assert_single_profile_runtime() -> None:
    if _multiplex_active():
        raise RuntimeError(
            "Freemaxxing is unavailable while gateway.multiplex_profiles is "
            "enabled. Its loopback router is intentionally fail-closed until "
            "Hermes provides a profile-addressed local-provider capability "
            "wire; use a native provider or a non-multiplexed profile."
        )


def local_token() -> str:
    """Return this process's loopback capability token."""
    return _LOCAL_TOKEN


def _resolve_key(env_names: tuple[str, ...]) -> str:
    """Resolve one upstream key without crossing a profile secret scope."""
    try:
        from agent.secret_scope import (
            UnscopedSecretError,
            current_secret_scope,
            get_secret,
            is_multiplex_active,
        )

        scoped = current_secret_scope() is not None
        multiplex = bool(is_multiplex_active())
        if scoped or multiplex:
            for env_name in env_names:
                value = get_secret(env_name)
                if value and str(value).strip():
                    return str(value).strip()
            # Under multiplexing the active scope is authoritative.  Never
            # consult process-global env or another profile's .env on a miss.
            if multiplex:
                return ""
    except UnscopedSecretError:
        return ""
    except ImportError:
        pass
    except Exception:
        if _multiplex_active():
            return ""

    # Classic/single-profile mode: preserve Hermes' .env-over-shell precedence.
    try:
        from hermes_cli.config import get_env_value_prefer_dotenv

        for env_name in env_names:
            value = get_env_value_prefer_dotenv(env_name)
            if value and str(value).strip():
                return str(value).strip()
    except Exception:
        pass

    if _multiplex_active():
        return ""
    for env_name in env_names:
        value = os.environ.get(env_name, "")
        if value and value.strip():
            return value.strip()
    return ""


def _resolve_nous_credentials() -> tuple[str, str]:
    """Resolve the existing Nous invoke JWT without a static-key downgrade."""
    _assert_single_profile_runtime()
    try:
        from hermes_cli import auth as auth_mod

        credentials = auth_mod.resolve_nous_runtime_credentials()
        key = str(credentials.get("api_key") or "").strip()
        base_url = str(credentials.get("base_url") or _NOUS_BASE_URL).rstrip("/")
        if key:
            return base_url, key
    except Exception as exc:
        logger.debug("freemaxxing: Nous OAuth resolution failed: %s", exc)

    # Backward-compatible static key support remains profile-scoped.  It does
    # not fall through to process env when multiplexing is enabled.
    return _NOUS_BASE_URL, _resolve_key(("NOUS_API_KEY",))


def _build_pool() -> None:
    """Construct the single-profile pool after local bearer authentication."""
    _assert_single_profile_runtime()
    pool.clear()

    base_url, api_key = _resolve_nous_credentials()
    pool.add(
        Backend(
            name="nous-portal",
            base_url=base_url,
            api_key=api_key,
            tier=0,
            refresh=_resolve_nous_credentials,
            default_model=_NOUS_FREE_MODEL,
        )
    )

    openrouter_key = _resolve_key(("OPENROUTER_API_KEY",))
    if openrouter_key:
        pool.add(
            Backend(
                name="openrouter",
                base_url=_OPENROUTER_BASE_URL,
                api_key=openrouter_key,
                tier=1,
            )
        )

    logger.info(
        "freemaxxing: runtime pool initialized with %d backend(s)",
        pool.count(),
    )


def _configured_port() -> int:
    raw = os.environ.get("FREEMAXXING_PORT", str(_DEFAULT_PORT)).strip()
    try:
        port = int(raw)
    except (TypeError, ValueError):
        return _DEFAULT_PORT
    if port == 0:  # test harness / explicit ephemeral listener
        return 0
    if 1 <= port <= 65535:
        return port
    return _DEFAULT_PORT


def _loopback_base_url(port: int) -> str:
    return f"http://127.0.0.1:{port}/v1"


def _ensure_listener():
    """Start the authenticated listener without resolving any upstream key."""
    global _listener
    with _listener_lock:
        if _listener is None:
            _listener = spawn_proxy(
                port=_configured_port(),
                token=_LOCAL_TOKEN,
                pool_initializer=_build_pool,
            )
        return _listener


def stop_proxy() -> None:
    """Stop this process's listener and clear the package lifecycle handle."""
    global _listener
    with _listener_lock:
        target = _listener
        if target is not None:
            _stop_proxy(target)
        _listener = None
        pool.clear()


def ensure_proxy() -> str:
    """Return the live loopback URL, rejecting multiplexed runtimes."""
    _assert_single_profile_runtime()
    server = _ensure_listener()
    return _loopback_base_url(int(server.server_address[1]))


class FreemaxxingProfile(ProviderProfile):
    """Static picker metadata; catalog inspection remains inside the router."""

    def fetch_models(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 8.0,
    ) -> list[str]:
        del api_key, base_url, timeout
        return [_ROUTER_MODEL]


def _register() -> None:
    # Listener startup is credential-free: no auth store, .env, or upstream
    # catalog is read until the first authenticated /models or completion call.
    server = _ensure_listener()
    base_url = _loopback_base_url(int(server.server_address[1]))

    # Generic API-key provider resolution carries the random local capability
    # to the OpenAI client.  In multiplex mode the profile secret scope is
    # authoritative, so this process-global value is intentionally invisible
    # and resolution fails closed before any HTTP request.
    os.environ["FREEMAXXING_API_KEY"] = _LOCAL_TOKEN

    register_provider(
        FreemaxxingProfile(
            name="freemaxxing",
            aliases=("fm", "freemaxxing-auto"),
            display_name="Freemaxxing",
            description=(
                "Authenticated local router across proven-free Nous and "
                "OpenRouter routes"
            ),
            signup_url="",
            env_vars=("FREEMAXXING_API_KEY",),
            base_url=base_url,
            auth_type="api_key",
            api_mode="chat_completions",
            supports_health_check=False,
            supports_vision=False,
            default_aux_model="",
            fallback_models=(_ROUTER_MODEL,),
        )
    )

    # ``hermes_cli.auth`` may already be importing while provider discovery
    # reaches this plugin.  Its registry objects exist before the auto-extension
    # loop, so install the runtime composition bridge here.  This closes the
    # import-order hole where the profile is discoverable but the generic
    # resolver snapshots no usable local capability.
    from hermes_cli.auth import PROVIDER_REGISTRY, ProviderConfig

    runtime_config = ProviderConfig(
        id="freemaxxing",
        name="Freemaxxing",
        auth_type="api_key",
        inference_base_url=base_url,
        api_key_env_vars=_LocalCapabilityEnvVars(),
    )
    PROVIDER_REGISTRY["freemaxxing"] = runtime_config
    PROVIDER_REGISTRY["fm"] = runtime_config
    PROVIDER_REGISTRY["freemaxxing-auto"] = runtime_config

    logger.info(
        "freemaxxing: provider registered at %s; upstream pool remains lazy",
        base_url,
    )


_register()