"""API-key source precedence and validation for an already selected provider."""

from __future__ import annotations

import logging
from typing import Any
from auth.pool_environment import PoolEnvironment
from auth.secret_validation import _usable_declared_secret

logger = logging.getLogger(__name__)


def _model_level_key_env(provider_id: str, environment: PoolEnvironment) -> str:
    """``model.key_env`` when config.yaml's main model targets *provider_id*, else ``""``.

    The Desktop settings UI saves registry-provider keys as a credential pointer
    (``model.key_env`` → ``$HERMES_HOME/.env``) instead of the registry's canonical env var,
    so credential resolution must consult it (#106336).
    """
    try:
        model_cfg = (environment.read_config() or {}).get("model")
    except Exception:
        return ""
    if not isinstance(model_cfg, dict):
        return ""
    if str(model_cfg.get("provider") or "").strip().lower() != provider_id:
        return ""
    return str(model_cfg.get("key_env") or model_cfg.get("api_key_env") or "").strip()


def resolve_api_key_provider_secret(provider_id: str, pconfig: Any, *, environment: PoolEnvironment) -> tuple[str, str]:
    """Resolve an API-key provider's token and indicate where it came from."""
    environment.require_current_scope()
    if environment.read_secret is None:
        raise ValueError("API-key resolution requires an explicit secret reader")
    if provider_id == "copilot":
        # The dedicated copilot auth module does proper token validation/exchange.
        try:
            from auth.providers.copilot import resolve_copilot_token, get_copilot_api_token
            token, source = resolve_copilot_token()
            if token:
                api_token, _base_url = get_copilot_api_token(token)
                return api_token, source
        except ValueError as exc:
            logger.warning("Copilot token validation failed: %s", exc)
        except Exception:
            pass
        return "", ""

    # Prefer ~/.hermes/.env over os.environ so a deliberate key rotation in .env isn't shadowed by
    # a stale shell export inherited from a parent process (Codex CLI, test runners, etc.).
    get_env_value_prefer_dotenv = environment.read_secret

    # Desktop-saved credential pointer: the settings UI persists registry-provider keys as
    # model.key_env → $HERMES_HOME/.env (e.g. HERMES_CUSTOM_LMSTUDIO_API_KEY) while keeping
    # model.provider on the registry id, so the pointer must be honored here or the UI-saved
    # key is silently ignored and lmstudio falls through to its no-auth placeholder (#106336).
    key_env = _model_level_key_env(provider_id, environment)
    if key_env:
        val = _usable_declared_secret(provider_id, get_env_value_prefer_dotenv(key_env), key_env)
        if val:
            return val, key_env

    for env_var in pconfig.api_key_env_vars:
        val = _usable_declared_secret(provider_id, get_env_value_prefer_dotenv(env_var), env_var)
        if val:
            # A provably malformed key (declared prefix mismatch) must not shadow a valid credential-pool
            # entry (#93593). Warn and keep looking instead of returning it.
            return val, env_var

    # Fallback: credential pool (e.g. zai key stored via auth.json). Prefer the pool's own
    # selection (peek) but try the rest too so one malformed entry doesn't block a valid one.
    pool_source = f"credential_pool:{provider_id}"
    try:
        from auth.credential_pool import load_pool
        pool = load_pool(provider_id, environment=environment)
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
                val = _usable_declared_secret(provider_id, key, pool_source)
                if val:
                    return val, pool_source
    except Exception:
        pass
    return "", ""


def get_anthropic_key(*, environment: PoolEnvironment) -> str:
    """First usable Anthropic credential (``.env`` preferred over a stale shell export), or ``""``.

    Order mirrors ``PROVIDER_REGISTRY["anthropic"].api_key_env_vars``.

    Checks both the ``.env`` file and the process environment, preferring ``~/.hermes/.env`` so a deliberate
    key rotation isn't shadowed by a stale shell export (matches the api-key resolution path — see #20591).
    """
    environment.require_current_scope()
    env_vars = environment.provider_config("anthropic").api_key_env_vars
    return next((v for v in (environment.read_secret(var) or "" for var in env_vars) if v), "")
