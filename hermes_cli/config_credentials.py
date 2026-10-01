"""Supply configuration and presentation collaborators to authentication.

Configuration writes retain the existing owner and writer. Shared credential
policy is implemented directly in auth.sources.
"""

from __future__ import annotations
import auth.providers.codex_quota as _auth_auth_providers_codex_quota
import auth.providers.nous_store as _auth_auth_providers_nous_store


import auth.constants as _auth_auth_constants
import auth.oauth as _auth_auth_oauth
import auth.providers.codex as _auth_auth_providers_codex
import auth.providers.copilot as _auth_auth_providers_copilot
import auth.providers.nous as _auth_auth_providers_nous
import auth.providers.qwen as _auth_auth_providers_qwen
import auth.providers.xai as _auth_auth_providers_xai

from typing import Any, List
from auth.context import CredentialScope
from auth.sources import CredentialEnvironment
from hermes_constants import get_hermes_home


def _providers_for_env_var(env_var: str) -> List[str]:
    """Provider ids whose registered api_key_env_vars include ``env_var``."""
    try:
        from hermes_cli.auth import PROVIDER_REGISTRY
    except Exception:
        return []
    hits: List[str] = []
    for pid, cfg in PROVIDER_REGISTRY.items():
        try:
            if env_var in (cfg.api_key_env_vars or ()):
                hits.append(pid)
        except Exception:
            continue
    return hits


def _scrub_config_yaml_mirrors(old_value: str, new_value: str | None) -> List[str]:
    """Reconcile config.yaml api_key mirrors holding ``old_value``; return dotted paths touched.

    Value-matched on purpose: only an entry holding the SAME credential that just changed in
    ``.env`` is touched. ``new_value=None`` removes the field. Operates on the RAW user config
    so defaults are never baked into the user's file.
    """
    if not old_value:
        return []
    from hermes_cli.config import (
        atomic_config_write,
        get_config_path,
        read_user_config_raw,
    )

    config_path = get_config_path()
    if not config_path.exists():
        return []
    try:
        user_config = read_user_config_raw(config_path)
    except Exception:
        return []
    if not user_config:
        return []

    touched: List[str] = []

    def _fix(
        section: Any, key_path: str, fields: tuple[str, ...] = ("api_key", "api")
    ) -> None:
        # "api" is the legacy alias for model.api_key in older configs. In the keyed ``providers``
        # schema ``api`` means the base_url, not a credential, so that section passes
        # ``fields=("api_key",)``.
        if not isinstance(section, dict):
            return
        for field in fields:
            current = section.get(field)
            if isinstance(current, str) and current == old_value:
                if new_value:
                    section[field] = new_value
                else:
                    section.pop(field, None)
                touched.append(f"{key_path}.{field}")

    def _items(value: Any, allow_list: bool):
        if isinstance(value, dict):
            return value.items()
        return enumerate(value) if allow_list and isinstance(value, list) else ()

    _fix(user_config.get("model"), "model")
    for task, slot_cfg in _items(user_config.get("auxiliary"), False):
        _fix(slot_cfg, f"auxiliary.{task}")
    for name, entry in _items(user_config.get("custom_providers"), True):
        _fix(entry, f"custom_providers.{name}")

    # ``providers.<id>.api_key`` (v12+) is where dashboard/desktop write custom-endpoint
    # credentials. It is a real inline secret with higher precedence than the env var, so a stale
    # copy shadows a rotation (persistent 401 with a key the UI no longer shows) and survives a
    # removal that promised to clear EVERY store.
    for provider_id, entry in _items(user_config.get("providers"), False):
        _fix(entry, f"providers.{provider_id}", fields=("api_key",))

    if touched:
        atomic_config_write(config_path, user_config)
    return touched


def credential_environment() -> CredentialEnvironment:
    from hermes_cli.config import load_env, save_env_value, remove_env_value

    def clear_models_cache(provider: str):
        from hermes_cli.models import clear_provider_models_cache

        return clear_provider_models_cache(provider)

    def seed_pool(provider: str):
        from auth.credential_pool import load_pool

        return load_pool(provider, environment=credential_pool_environment())

    return CredentialEnvironment(
        scope=CredentialScope(get_hermes_home()),
        read_env=load_env,
        save_env=save_env_value,
        remove_env=remove_env_value,
        reconcile_mirrors=_scrub_config_yaml_mirrors,
        providers_for_env=_providers_for_env_var,
        clear_models_cache=clear_models_cache,
        seed_pool=seed_pool,
    )


def _pool_provider_hooks(provider: str, environment):
    from auth.pool_environment import PoolProviderHooks

    def nous_hooks():
        return PoolProviderHooks(
            resolve_credentials=lambda **kwargs: (
                _auth_auth_providers_nous.resolve_nous_runtime_credentials(
                    **kwargs, environment=environment
                )
            ),
            terminal_error=lambda exc: _auth_auth_oauth._is_terminal_nous_refresh_error(
                exc
            ),
            quarantine_state=lambda *args, **kwargs: (
                _auth_auth_providers_nous_store._quarantine_nous_oauth_state(
                    *args, **kwargs
                )
            ),
            quarantine_pool=lambda *args, **kwargs: (
                _auth_auth_providers_nous_store._quarantine_nous_pool_entries(
                    *args, **kwargs
                )
            ),
        )

    def codex_hooks():
        from hermes_cli.auth_codex import _codex_pool_route_base_url

        return PoolProviderHooks(
            refresh_tokens=lambda *args: (
                _auth_auth_providers_codex.refresh_codex_oauth_pure(
                    *args, environment=environment
                )
            ),
            terminal_error=lambda exc: (
                _auth_auth_oauth._is_terminal_codex_oauth_refresh_error(exc)
            ),
            token_expiring=lambda token: (
                _auth_auth_providers_codex._codex_access_token_is_expiring(
                    token, _auth_auth_constants.CODEX_ACCESS_TOKEN_REFRESH_SKEW_SECONDS
                )
            ),
            quota_shaped=lambda *args: (
                _auth_auth_providers_codex_quota._is_codex_rate_limit_shaped(*args)
            ),
            refresh_probe_token=lambda *args: (
                _auth_auth_providers_codex_quota._refresh_expired_codex_probe_token(
                    *args, environment=environment
                )
            ),
            quota_probe=lambda *args, **kwargs: (
                _auth_auth_providers_codex_quota._probe_codex_quota_restored(
                    *args, **kwargs
                )
            ),
            route_base_url=_codex_pool_route_base_url,
        )

    def xai_hooks():
        return PoolProviderHooks(
            refresh_tokens=lambda *args: (
                _auth_auth_providers_xai.refresh_xai_oauth_pure(*args)
            ),
            terminal_error=lambda exc: (
                _auth_auth_oauth._is_terminal_xai_oauth_refresh_error(exc)
            ),
            token_expiring=lambda token: (
                _auth_auth_providers_xai._xai_access_token_is_expiring(
                    token,
                    _auth_auth_providers_xai._xai_proactive_refresh_skew_seconds(token),
                )
            ),
        )

    def qwen_hooks():
        return PoolProviderHooks(
            resolve_credentials=lambda **kwargs: (
                _auth_auth_providers_qwen.resolve_qwen_runtime_credentials(**kwargs)
            )
        )

    def copilot_hooks():
        return PoolProviderHooks(
            resolve_external_token=lambda: (
                _auth_auth_providers_copilot.resolve_copilot_token()
            ),
            exchange_external_token=lambda token: (
                _auth_auth_providers_copilot.get_copilot_api_token(token)
            ),
            external_env_vars=tuple(_auth_auth_providers_copilot.COPILOT_ENV_VARS),
        )

    builders = {
        "nous": nous_hooks,
        "openai-codex": codex_hooks,
        "xai-oauth": xai_hooks,
        "qwen-oauth": qwen_hooks,
        "copilot": copilot_hooks,
    }
    return builders.get(provider, PoolProviderHooks)()


def credential_pool_environment():
    """Capture the active scope and provide the existing configuration owner."""
    from auth.pool_environment import PoolEnvironment
    from hermes_cli import config
    from hermes_cli import auth as provider_auth
    from hermes_cli.env_loader import get_secret_source
    from hermes_cli.route_identity import normalize_route_base_url

    def key_endpoint(provider, token, fallback, override):
        resolvers = {
            "kimi-coding": provider_auth._resolve_kimi_base_url,
            "zai": provider_auth._resolve_zai_base_url,
        }
        resolver = resolvers.get(provider)
        return resolver(token, fallback, override) if resolver else override or fallback

    environment = PoolEnvironment(
        scope=CredentialScope(get_hermes_home()),
        read_config=lambda: config.load_config_readonly(),
        custom_providers=config.get_compatible_custom_providers,
        read_env=lambda: config.load_env(),
        secret_source=get_secret_source,
        provider_config=lambda provider: provider_auth.PROVIDER_REGISTRY.get(provider),
        provider_configured=lambda provider: (
            provider_auth.is_provider_explicitly_configured(provider)
        ),
        key_endpoint=key_endpoint,
        normalize_endpoint=normalize_route_base_url,
        provider_hooks=lambda provider: _pool_provider_hooks(provider, environment),
        read_secret=config.get_env_value_prefer_dotenv,
        oauth_user_agent=_codex_oauth_user_agent,
        entitlement_message=lambda capability: _nous_entitlement_message(
            capability, environment
        ),
    )
    return environment


def _codex_oauth_user_agent():
    from hermes_cli.version_info import get_version_info

    return f"hermes-cli/{get_version_info().base_version}"


def _nous_entitlement_message(capability, environment):
    from hermes_cli.nous_account import (
        get_nous_portal_account_info,
        format_nous_portal_entitlement_message,
    )

    info = get_nous_portal_account_info(force_fresh=True)
    return format_nous_portal_entitlement_message(info, capability=capability) or ""
