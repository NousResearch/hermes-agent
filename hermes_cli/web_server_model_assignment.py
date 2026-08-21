"""Main-slot model assignment helpers for the Hermes dashboard web server.

Extracted verbatim from ``hermes_cli/web_server.py`` (god-file slice R1-C9,
epic #78791). ``web_server`` re-exports both names at the original location,
so existing imports like ``from hermes_cli.web_server import
_normalize_main_model_assignment`` keep working and the identity invariant
``web_server.X is web_server_model_assignment.X`` holds for both names.

Dependencies point only downward (stdlib + ``hermes_cli.config``); the
hermes_cli.models / hermes_cli.providers imports stay deferred inside the
functions so this module imports standalone without pulling the FastAPI
stack.
"""

import logging
from typing import Any

from hermes_cli.config import clear_model_endpoint_credentials, load_config

# Logger-name preservation: web_server.py used ``logging.getLogger(__name__)``;
# keep the same name so log records are indistinguishable from before the move.
logger = logging.getLogger("hermes_cli.web_server")
_log = logger


def _normalize_main_model_assignment(provider: str, model: str) -> tuple[str, str]:
    """Normalize a main-slot (provider, model) pair before persisting.

    The Models page has two assignment paths and only one of them was safe:

    - The "Change" picker sends a real Hermes provider slug — fine.
    - The per-card "Use as → Main model" menu sends ``entry.provider``
      from the analytics rows, falling back to the model's VENDOR prefix
      (``modelVendor("anthropic/claude-opus-4.6") == "anthropic"``) when
      the session row has no ``billing_provider`` (older sessions, NULL
      rows).  That wrote ``provider: anthropic`` +
      ``default: anthropic/claude-opus-4.6`` to config — a vendor-prefixed
      OpenRouter slug on the NATIVE Anthropic provider.  New sessions then
      400 against api.anthropic.com ("model: anthropic/claude-opus-4.6 not
      found") and the user reads it as "changing models does nothing".

    Two repairs, both at this single chokepoint so every caller inherits:

    1. Vendor-name → Hermes-provider mapping: when the provider string is
       not a known Hermes provider/alias (e.g. ``moonshotai``, ``x-ai`` is
       known but ``poolside`` isn't) but the model is a vendor-prefixed
       aggregator slug, keep the user's CURRENT aggregator if they're on
       one, else fall back to openrouter.

       Named custom providers (``custom:litellm``, etc.) are excluded from
       this fallback: ``_KNOWN_PROVIDER_NAMES`` only lists the bare
       ``"custom"`` bucket, never a specific ``custom:<name>`` slug, so
       without this exclusion every named custom provider paired with a
       slash-bearing model (e.g. ``ollama/glm-5.2`` behind a LiteLLM proxy)
       looked exactly like the stray-vendor-prefix case above and got
       silently reassigned to ``openrouter``.
    2. Model-format normalization for the resolved provider via
       ``normalize_model_for_provider`` (e.g. ``anthropic/claude-opus-4.6``
       on native anthropic → ``claude-opus-4-6``).
    """
    from hermes_cli.config import get_compatible_custom_providers
    from hermes_cli.models import _KNOWN_PROVIDER_NAMES, normalize_provider
    from hermes_cli.model_normalize import normalize_model_for_provider
    from hermes_cli.providers import resolve_custom_provider, resolve_user_provider

    prov_in = (provider or "").strip()
    model_in = (model or "").strip()
    canonical = normalize_provider(prov_in)

    # User-declared providers are real routing targets, not analytics vendor
    # labels. Resolve them before the unknown-vendor fallback. ``providers:``
    # keeps its declared bare slug; ``custom_providers:`` canonicalizes both a
    # bare display name and ``custom:<name>`` to the durable custom slug.
    try:
        cfg = load_config()
    except Exception:
        cfg = {}
    user_providers = cfg.get("providers") if isinstance(cfg, dict) else None
    user_provider = resolve_user_provider(
        prov_in, user_providers if isinstance(user_providers, dict) else {}
    )
    custom_provider = resolve_custom_provider(
        prov_in,
        get_compatible_custom_providers(cfg) if isinstance(cfg, dict) else [],
    )
    if user_provider is not None:
        return user_provider.id, model_in
    if custom_provider is not None:
        return custom_provider.id, model_in

    # A named custom provider that didn't resolve above (typo, config
    # mismatch, entry missing from custom_providers/providers) must still
    # not be treated as a stray vendor prefix -- it isn't a known Hermes
    # provider/alias, but it also isn't the analytics-vendor case this
    # fallback exists for. Match only the durable named-custom syntax
    # (bare "custom" bucket, or "custom:<name>" per
    # ``providers.custom_provider_slug``) -- a bare ``startswith("custom")``
    # would also swallow unrelated unconfigured vendor names that merely
    # happen to start with "custom" (e.g. "customproxy").
    is_custom_provider_slug = canonical == "custom" or canonical.startswith("custom:")
    if (
        canonical not in _KNOWN_PROVIDER_NAMES
        and not is_custom_provider_slug
        and "/" in model_in
    ):
        # Vendor prefix posing as a provider (analytics fallback). Resolve
        # against the user's current provider when it's an aggregator that
        # serves vendor-prefixed slugs; otherwise default to openrouter.
        try:
            cur_cfg = cfg.get("model", {})
            cur_provider = (
                str(cur_cfg.get("provider", "") or "").strip().lower()
                if isinstance(cur_cfg, dict) else ""
            )
        except Exception:
            cur_provider = ""
        from hermes_cli.models import _AGGREGATOR_PROVIDERS
        if cur_provider and normalize_provider(cur_provider) in _AGGREGATOR_PROVIDERS:
            canonical = normalize_provider(cur_provider)
            prov_in = cur_provider
        else:
            canonical = "openrouter"
            prov_in = "openrouter"

    # Custom/user-config providers keep the model verbatim — the registry
    # normalizer doesn't know their namespaces.
    if canonical in _KNOWN_PROVIDER_NAMES and not canonical.startswith("custom"):
        try:
            normalized_model = normalize_model_for_provider(model_in, canonical)
            if normalized_model:
                model_in = normalized_model
        except Exception:
            _log.debug("model normalization failed for %s/%s", prov_in, model_in, exc_info=True)

    return prov_in, model_in


def _apply_main_model_assignment(
    model_cfg: "Any", provider: str, model: str, base_url: str = "", api_key: str = ""
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
    new_provider = provider.strip().lower()
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
    if api_key.strip():
        model_cfg["api_key"] = api_key.strip()
        model_cfg.pop("api", None)
    elif (model_cfg.get("api_key") or model_cfg.get("api")) and new_provider != prev_provider:
        # A stale endpoint secret can live under the legacy ``api`` alias with
        # no ``api_key`` (the resolver still reads ``model.api`` as a key), so
        # the switch-clears-the-key path must trigger on either field — else the
        # old endpoint's secret survives in config.yaml and contaminates a later
        # custom resolution. clear_model_endpoint_credentials scrubs both.
        clear_model_endpoint_credentials(model_cfg, clear_api_mode=False)
    if new_provider != prev_provider:
        clear_model_endpoint_credentials(model_cfg, clear_api_key=False)
    model_cfg.pop("context_length", None)
    return model_cfg
