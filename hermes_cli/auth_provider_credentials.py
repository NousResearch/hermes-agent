"""Credential resolution helpers for registry-backed providers.

The auth facade re-exports these names so existing imports and monkeypatch seams remain stable.
Helpers resolve facade functions at call time, preserving patches on ``hermes_cli.auth``.
"""

from __future__ import annotations

from typing import Any


def _resolve_api_key_provider_secret(provider_id: str, pconfig: Any) -> tuple[str, str]:
    """Resolve an API-key provider's token and indicate where it came from."""
    from hermes_cli import auth as facade

    if not facade._is_provider_enabled(provider_id):
        return "", ""

    if provider_id == "copilot":
        # The dedicated copilot auth module does proper token validation/exchange.
        try:
            from hermes_cli.copilot_auth import resolve_copilot_token, get_copilot_api_token
            token, source = resolve_copilot_token()
            if token:
                api_token, _base_url = get_copilot_api_token(token)
                return api_token, source
        except ValueError as exc:
            facade.logger.warning("Copilot token validation failed: %s", exc)
        except Exception:
            pass
        return "", ""

    # Prefer ~/.hermes/.env over os.environ so a deliberate key rotation in .env isn't shadowed by
    # a stale shell export inherited from a parent process (Codex CLI, test runners, etc.).
    from hermes_cli.config import get_env_value_prefer_dotenv

    # Desktop-saved credential pointer: the settings UI persists registry-provider keys as
    # model.key_env → $HERMES_HOME/.env (e.g. HERMES_CUSTOM_LMSTUDIO_API_KEY) while keeping
    # model.provider on the registry id, so the pointer must be honored here or the UI-saved
    # key is silently ignored and lmstudio falls through to its no-auth placeholder (#106336).
    key_env = facade._model_level_key_env(provider_id)
    if key_env:
        val = facade._usable_declared_secret(provider_id, get_env_value_prefer_dotenv(key_env), key_env)
        if val:
            return val, key_env

    for env_var in pconfig.api_key_env_vars:
        val = facade._usable_declared_secret(provider_id, get_env_value_prefer_dotenv(env_var), env_var)
        if val:
            # A provably malformed key (declared prefix mismatch) must not shadow a valid credential-pool
            # entry (#93593). Warn and keep looking instead of returning it.
            return val, env_var

    # Fallback: credential pool (e.g. zai key stored via auth.json). Prefer the pool's own
    # selection (peek) but try the rest too so one malformed entry doesn't block a valid one.
    pool_source = f"credential_pool:{provider_id}"
    try:
        from agent.credential_pool import load_pool
        pool = load_pool(provider_id)
        if pool and pool.has_credentials():
            entry = pool.peek()
            candidates = [entry] if entry is not None else []
            try:
                for extra in pool.entries():
                    if extra is not None and all(extra is not c for c in candidates):
                        candidates.append(extra)
            except Exception:
                pass
            for entry in candidates:
                key = getattr(entry, "access_token", "") or getattr(entry, "runtime_api_key", "")
                val = facade._usable_declared_secret(provider_id, key, pool_source)
                if val:
                    return val, pool_source
    except Exception:
        pass
    return "", ""
