"""Config/runtime-resolution helpers extracted from gateway/run.py (#54962).

Twelfth slice of the gateway god-file unpacking: profile-runtime scoping
(``_profile_runtime_scope``), process-level gateway config loading for the
runner (``load_gateway_config_for_runner``), open-policy startup validation
(``_own_policy_open_startup_violation``), and per-provider runtime credential
resolution (``_resolve_runtime_agent_kwargs_for_provider``,
``_credential_pool_for_provider``).

All functions moved VERBATIM from gateway/run.py — zero behavior change.
This module never imports gateway.run (that would be circular); gateway.run
imports these names back so ``gateway.run.<name>`` references stay green.

Left in gateway/run.py (documented in the slice PR):
- ``_current_max_iterations`` — its reload helper reads ``_hermes_home``
  module state that tests monkeypatch via ``gateway.run._hermes_home``.
- ``_resolve_runtime_agent_kwargs`` / ``_try_resolve_fallback_provider`` —
  the fallback chain calls ``_load_gateway_runtime_config``, which builds on
  the ``_load_gateway_config`` hub that stays in gateway/run.py.
"""

import logging
import os
from contextlib import contextmanager as _contextmanager
from pathlib import Path
from typing import Optional

from gateway.config import (
    Platform,
    _getenv,
    load_gateway_config,
)
from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)


@_contextmanager
def _profile_runtime_scope(profile_home: "Path"):
    """Scope config/skills/memory AND credentials to a profile for one turn.

    Combines the two seams the multiplexer needs:
      1. ``set_hermes_home_override`` — redirects ``get_hermes_home()`` (config,
         skills, memory, SOUL, sessions) to the profile's home. Contextvar, so
         it propagates into the agent worker thread via ``copy_context()``.
      2. ``set_secret_scope`` — installs the profile's ``.env`` secrets as the
         authoritative credential source, so ``get_secret`` reads this profile's
         keys and never the process-global ``os.environ`` (which in a
         multiplexer may hold another profile's values).

    Only used on the multiplexed inbound path. Single-profile gateways never
    enter this scope, so their behavior is unchanged. Loading the profile's
    ``.env`` here does NOT mutate ``os.environ`` — ``build_profile_secret_scope``
    returns an isolated dict — which is what keeps subprocesses (MCP, kanban)
    from inheriting cross-profile secrets.
    """
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    from agent.secret_scope import (
        build_profile_secret_scope,
        set_secret_scope,
        reset_secret_scope,
    )
    from hermes_cli.env_loader import hydrate_profile_secret_sources

    home_token = set_hermes_home_override(str(profile_home))
    hydrate_profile_secret_sources(Path(profile_home))
    secret_token = set_secret_scope(build_profile_secret_scope(Path(profile_home)))
    try:
        yield
    finally:
        reset_secret_scope(secret_token)
        reset_hermes_home_override(home_token)


def load_gateway_config_for_runner() -> "GatewayConfig":
    """Load gateway config for the process-level GatewayRunner.

    When ``gateway.multiplex_profiles`` is off, this is identical to
    ``load_gateway_config()`` (legacy single-profile path).

    When multiplexing is on, reload under the default/active profile's
    ``_profile_runtime_scope`` so platform tokens in that profile's ``.env``
    resolve through the secret scope — the same path secondary profiles use
    in ``_start_one_profile_adapters``. Without this, primary startup calls
    ``load_gateway_config()`` unscoped: ``_getenv`` falls through to
    ``os.environ``, which often has no ``TELEGRAM_BOT_TOKEN`` once the token
    lives only under ``profiles/<name>/.env`` (#64674).

    Single-profile gateways never set ``multiplex_profiles``, so they keep the
    unscoped load and are unaffected.
    """
    cfg = load_gateway_config()
    if not getattr(cfg, "multiplex_profiles", False):
        return cfg
    try:
        home = get_hermes_home()
    except Exception:
        return cfg
    try:
        with _profile_runtime_scope(Path(home)):
            return load_gateway_config()
    except Exception:
        logger.debug(
            "multiplex default-scope config reload failed; using unscoped load",
            exc_info=True,
        )
        return cfg


_OWN_POLICY_OPEN_ENV = {
    Platform.WECOM: ("WECOM_DM_POLICY", "WECOM_GROUP_POLICY", "WECOM_ALLOW_ALL_USERS"),
    Platform.WEIXIN: ("WEIXIN_DM_POLICY", "WEIXIN_GROUP_POLICY", "WEIXIN_ALLOW_ALL_USERS"),
    Platform.YUANBAO: ("YUANBAO_DM_POLICY", "YUANBAO_GROUP_POLICY", "YUANBAO_ALLOW_ALL_USERS"),
    Platform.QQBOT: (None, None, "QQ_ALLOW_ALL_USERS"),
    Platform.WHATSAPP: ("WHATSAPP_DM_POLICY", "WHATSAPP_GROUP_POLICY", "WHATSAPP_ALLOW_ALL_USERS"),
}


def _own_policy_open_startup_violation(config) -> Optional[str]:
    """Return a startup-abort reason when open policy lacks allow-all opt-in."""
    for platform, platform_config in getattr(config, "platforms", {}).items():
        if not getattr(platform_config, "enabled", False):
            continue
        open_env = _OWN_POLICY_OPEN_ENV.get(platform)
        if not open_env:
            continue
        dm_env, group_env, allow_all_env = open_env
        extra = getattr(platform_config, "extra", None) or {}
        dm_policy = str(
            extra.get("dm_policy")
            or (_getenv(dm_env, "pairing") if dm_env else "pairing")
        ).strip().lower()
        group_policy = str(
            extra.get("group_policy")
            or (_getenv(group_env, "pairing") if group_env else "pairing")
        ).strip().lower()
        if dm_policy != "open" and group_policy != "open":
            continue
        gateway_allow_all = os.getenv(
            "GATEWAY_ALLOW_ALL_USERS", ""
        ).lower() in {"true", "1", "yes"}
        platform_opted_in = gateway_allow_all or (
            allow_all_env
            and _getenv(allow_all_env, "").lower() in {"true", "1", "yes"}
        )
        if platform_opted_in:
            continue
        return f"{platform.value}: open policy without allow-all opt-in"
    return None


def _resolve_runtime_agent_kwargs_for_provider(provider: str) -> dict:
    """Resolve runtime credentials for a specific provider (e.g. from channel override)."""
    from hermes_cli.runtime_provider import (
        resolve_runtime_provider,
        format_runtime_provider_error,
    )
    try:
        runtime = resolve_runtime_provider(requested=provider)
    except Exception as exc:
        raise RuntimeError(format_runtime_provider_error(exc)) from exc
    return {
        "api_key": runtime.get("api_key"),
        "base_url": runtime.get("base_url"),
        "provider": runtime.get("provider"),
        "requested_provider": runtime.get("requested_provider"),
        "api_mode": runtime.get("api_mode"),
        "command": runtime.get("command"),
        "args": list(runtime.get("args") or []),
        "credential_pool": runtime.get("credential_pool"),
    }


def _credential_pool_for_provider(provider: Optional[str]):
    """Return the live credential pool for a provider id (e.g. ``custom:hyper``)."""
    if not provider or not str(provider).strip():
        return None
    try:
        return _resolve_runtime_agent_kwargs_for_provider(str(provider).strip()).get(
            "credential_pool"
        )
    except Exception:
        logger.debug(
            "Failed to resolve credential pool for provider=%s",
            provider,
            exc_info=True,
        )
        return None
