"""Provider readiness checks used by the CLI setup and startup paths.

The facade re-exports ``_has_any_provider_configured`` so callers and tests keep their existing
import and monkeypatch surface. All reads of facade helpers happen at call time for that seam.
"""

from __future__ import annotations

import os


def _configured_model(cfg: dict) -> tuple[object, str, bool]:
    from hermes_cli.config import DEFAULT_CONFIG, split_model_config_default

    model_cfg = cfg.get("model")
    model_name = model_cfg if isinstance(model_cfg, str) else ""
    if isinstance(model_cfg, dict):
        model_name = model_cfg.get("default") or ""
        if isinstance(model_name, dict):
            model_name, _ = split_model_config_default(model_name)
    model_name = str(model_name).strip()
    return model_cfg, model_name, bool(model_name and model_name != DEFAULT_CONFIG.get("model", ""))


def _provider_env_vars(registry, enabled) -> set[str]:
    names = set()
    if enabled("openrouter"):
        names.add("OPENROUTER_API_KEY")
    if enabled("openrouter") or enabled("custom"):
        names.add("OPENAI_API_KEY")
    if enabled("custom"):
        names.add("OPENAI_BASE_URL")
    for provider_id, pconfig in registry.items():
        if pconfig.auth_type == "api_key" and enabled(provider_id):
            names.update(pconfig.api_key_env_vars)
    return names


def _env_provider_configured(env_path, names, strict_profile_scope: bool) -> bool:
    from hermes_cli import main as facade

    if strict_profile_scope:
        from agent.secret_scope import current_secret_scope

        read_env = (current_secret_scope() or {}).get
    else:
        read_env = os.getenv
    return any(read_env(name) for name in names) or facade._dotenv_has_provider_key(env_path, names)


def _model_provider_configured(model_cfg, enabled) -> bool:
    if not isinstance(model_cfg, dict) or not any(
        (model_cfg.get(key) or "").strip() for key in ("provider", "base_url", "api_key")
    ):
        return False
    selected = str(model_cfg.get("provider") or "custom").strip().lower()
    return enabled(selected)


def _auth_store_provider_configured(auth_path, registry, strict_profile_scope: bool) -> bool:
    from hermes_cli import main as facade

    return facade._auth_store_logged_in(auth_path, registry, strict_profile_scope)


def _registry_provider_configured(registry, strict_profile_scope: bool, get_auth_status) -> bool:
    from hermes_cli import main as facade

    if strict_profile_scope:
        return False
    try:
        return any(
            get_auth_status(provider_id).get("logged_in")
            for provider_id, pconfig in registry.items()
            if pconfig.auth_type == "api_key"
        )
    except Exception as exc:  # health: allow BLE001 -- provider auth plugins are third-party boundaries; fail closed after sanitized logging
        facade.logger.exception("provider auth status sweep failed: %s", type(exc).__name__)
        return False


def _claude_code_configured(has_hermes_config: bool, strict_profile_scope: bool, enabled) -> bool:
    from hermes_cli import main as facade

    if not has_hermes_config or strict_profile_scope or not enabled("anthropic"):
        return False
    try:
        from agent.anthropic_credentials import read_claude_code_credentials, is_claude_code_token_valid

        creds = read_claude_code_credentials()
        return bool(creds and (is_claude_code_token_valid(creds) or creds.get("refreshToken")))
    except Exception as exc:  # health: allow BLE001 -- credential readers include optional integrations; fail closed after sanitized logging
        facade.logger.exception("Claude Code credential check failed: %s", type(exc).__name__)
        return False


def _has_any_provider_configured(*, strict_profile_scope: bool = False) -> bool:
    """Check whether at least one configured provider is usable without creating one."""
    from hermes_cli import main as facade
    from hermes_cli.auth import PROVIDER_REGISTRY, _is_provider_enabled, get_auth_status
    from hermes_cli.config import get_env_path, get_hermes_home, load_config

    model_cfg, _model_name, has_hermes_config = _configured_model(load_config())
    if _env_provider_configured(
        get_env_path(), _provider_env_vars(PROVIDER_REGISTRY, _is_provider_enabled), strict_profile_scope
    ):
        return True
    if _auth_store_provider_configured(
        get_hermes_home() / "auth.json", PROVIDER_REGISTRY, strict_profile_scope
    ):
        return True
    if _model_provider_configured(model_cfg, _is_provider_enabled):
        return True
    if _registry_provider_configured(PROVIDER_REGISTRY, strict_profile_scope, get_auth_status):
        return True
    if _claude_code_configured(has_hermes_config, strict_profile_scope, _is_provider_enabled):
        return True
    try:
        from hermes_cli.anon_auth import guest_enabled, has_guest

        return _is_provider_enabled("nous") and guest_enabled() and has_guest()
    except Exception as exc:  # health: allow BLE001 -- free-tier integration is optional; fail closed after sanitized logging
        facade.logger.exception("free tier check on first run skipped: %s", type(exc).__name__)
        return False
