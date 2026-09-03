"""Main-slot model assignment mutation.

Owns the persist/apply seam that used to live in ``hermes_cli/web_server.py``
(~line 2000 apply + ~line 7800 POST /api/model/set). ``web_server`` is the
router: it calls these helpers and re-exports ``apply_main_model_assignment``
as ``_apply_main_model_assignment`` so existing imports keep working.
"""

from __future__ import annotations

from typing import Any

from hermes_cli.config import (
    clear_model_endpoint_credentials,
    custom_endpoint_key_env,
    get_env_value,
    save_env_value,
)


def persist_custom_endpoint_secret(provider: str, base_url: str, api_key: str) -> str:
    """Write a custom/local API key to ``.env``. Returns the ``key_env`` name.

    Empty string means this assignment has no custom-endpoint secret to stash
    (named providers, missing URL, or empty key). Raises if the ``.env`` write
    cannot be verified — callers must not persist a ``key_env`` pointer to a
    secret that is not actually on disk.
    """
    if provider.strip().lower() not in {"custom", "local"}:
        return ""
    if not base_url or not api_key.strip():
        return ""
    secret = api_key.strip()
    env_var = custom_endpoint_key_env(base_url)
    save_env_value(env_var, secret)
    if (get_env_value(env_var) or "").strip() != secret:
        raise RuntimeError(f"failed to persist {env_var} to .env")
    return env_var


def apply_main_model_assignment(
    model_cfg: Any,
    provider: str,
    model: str,
    base_url: str = "",
    api_key: str = "",
    key_env: str = "",
) -> dict:
    """Apply a main-slot model assignment to a ``model`` config dict in place.

    Sets ``provider``/``default``, then reconciles ``base_url``:

    - An explicitly supplied ``base_url`` is always persisted (covers
      ``custom``/local endpoints and any provider whose key is bound to a
      non-default host).
    - Otherwise, a stale ``base_url`` is cleared ONLY when switching to a
      *different* provider — that URL belonged to the old provider. When the
      provider is unchanged and no new URL is supplied, the existing
      ``base_url`` is preserved. This keeps a user's custom endpoint (e.g. a
      Xiaomi MiMo Token Plan host, ``https://token-plan-*.xiaomimimo.com/v1``)
      alive when they merely re-pick a model under the same provider — picking
      a model previously wiped it, forcing the registry default and breaking
      Token Plan keys.

    The runtime resolver reads ``model.base_url`` from config (it ignores
    ``OPENAI_BASE_URL``) and only honors it when the configured provider matches
    and the pool entry is on the registry default, so preserving it here is what
    lets the override actually route. The hardcoded ``context_length`` override
    is always dropped since the new model may have a different context window.

    Returns the same dict (coerced to a fresh dict if the input wasn't one) so
    callers can assign it straight back onto the model config.
    """
    if not isinstance(model_cfg, dict):
        model_cfg = {}
    prev_provider = str(model_cfg.get("provider") or "").strip().lower()
    prev_base_url = str(model_cfg.get("base_url") or "").strip().rstrip("/")
    new_provider = provider.strip().lower()
    new_base_url = base_url.strip().rstrip("/")
    endpoint_changed = bool(
        new_base_url and prev_base_url and new_base_url != prev_base_url
    )
    model_cfg["provider"] = provider
    model_cfg["default"] = model
    if base_url.strip():
        model_cfg["base_url"] = base_url.strip()
    elif model_cfg.get("base_url") and new_provider != prev_provider:
        # Switching providers: the old URL belonged to the old provider, drop
        # it so the new provider's default endpoint is used. Same-provider
        # re-assignment keeps the user's configured base_url intact.
        model_cfg["base_url"] = ""
    # The endpoint key follows the same lifecycle as base_url: an explicit key
    # is always persisted; an existing key is dropped only when switching to a
    # different provider (it belonged to the old endpoint), and preserved on a
    # same-provider re-pick so re-selecting a model doesn't wipe the key.
    if key_env.strip():
        model_cfg["key_env"] = key_env.strip()
        model_cfg.pop("api_key_env", None)
        model_cfg.pop("api_key", None)
        model_cfg.pop("api", None)
    elif api_key.strip():
        model_cfg["api_key"] = api_key.strip()
        model_cfg.pop("api", None)
        model_cfg.pop("key_env", None)
        model_cfg.pop("api_key_env", None)
    elif new_provider != prev_provider or endpoint_changed:
        # A stale endpoint secret can live under the legacy ``api`` alias with
        # no ``api_key`` (the resolver still reads ``model.api`` as a key), so
        # the switch-clears-the-key path must trigger on either field — else the
        # old endpoint's secret survives in config.yaml and contaminates a later
        # custom resolution. clear_model_endpoint_credentials scrubs both.
        clear_model_endpoint_credentials(model_cfg, clear_api_mode=False)
        model_cfg.pop("key_env", None)
        model_cfg.pop("api_key_env", None)
    if new_provider != prev_provider:
        clear_model_endpoint_credentials(model_cfg, clear_api_key=False)
    model_cfg.pop("context_length", None)
    return model_cfg
