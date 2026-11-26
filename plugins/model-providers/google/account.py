"""Solstice provider profile: a user's own per-user-quota Gemini subscription, signed in with OAuth.

Pre-release (``hidden=True``): no picker, setup list or accounts tab offers it until the user signs in
by name (``hermes auth add solstice``); after that it lists like any signed-in provider. Login and
token refresh go through the generic PKCE plugin flow with a Nous-portal broker as the token endpoint
(``auth.py``); inference goes straight to Google on the per-user-quota methods (``transport.py``).
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any

from hermes_cli.auth_oauth_pkce_plugin import OAuthPKCEConfig, pkce_auth_handler, pkce_refresh_credential
from providers.base import ProviderProfile

from .account_auth import broker_token_request, discover_client

logger = logging.getLogger(__name__)

# Lives here, not in transport.py: discovery imports this module from interpreters without the app's
# HTTP stack (PM's runtime reads config, which lists providers), so the transport loads on first use.
INFERENCE_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

# Hardcoded because the per-user-quota endpoint has no model listing a user token may read (its
# /models answers 403 ACCESS_TOKEN_SCOPE_INSUFFICIENT): the IDs this endpoint serves, nothing
# else. The first entry is the default model. Quota is per model, so Flash-Lite stays last as the reserve
# once the larger models are exhausted (``gemini-flash-lite-latest`` is also the aux model: titles and
# background calls draw from the same pool).
FALLBACK_MODELS = (
    "gemini-4-hydrogen", "gemini-4-neon", "gemini-4-carbon", "gemini-3.5-flash-lite", "gemini-flash-lite-latest",
)

# (allowed ``thinkingLevel`` values weakest first, default) per listed model. The levels differ
# per model and any other one is an HTTP 400: Gemini 4 rejects ``minimal``, Carbon and Flash-Lite
# reject ``xhigh``. Kept here, not in the shared Gemini helpers, which the gemini and vertex profiles use.
_FLASH_LITE_LEVELS = (("minimal", "low", "medium", "high"), "low")
_THINKING_LEVELS: dict[str, tuple[tuple[str, ...], str]] = {
    "gemini-4-hydrogen": (("low", "medium", "high", "xhigh"), "high"),
    "gemini-4-neon": (("low", "medium", "high", "xhigh"), "high"),
    "gemini-4-carbon": (("low", "medium", "high"), "medium"),
    "gemini-3.5-flash-lite": _FLASH_LITE_LEVELS,
    "gemini-flash-lite-latest": _FLASH_LITE_LEVELS,
}


# Ids the earlier list offered that this endpoint does not serve. A user who picked one keeps it in
# config; each is sent as the default model instead of failing every turn. Exactly these, nothing else.
_RETIRED_MODELS = frozenset({"gemini-4-argon", "gemini-3.1-pro", "gemini-3.8-flash"})
_RETIRED_LOGGED: set[str] = set()


def current_model_id(model: str | None) -> str | None:
    """The default model for a retired id (any spelling the client accepts, ``gemini/``/``google/``
    included), logged once per process per id; None for every other id, which stays as typed."""
    from agent.reasoning_effort import bare_gemini_model_id

    bare = bare_gemini_model_id(model).lower()
    if bare not in _RETIRED_MODELS:
        return None
    if bare not in _RETIRED_LOGGED:
        _RETIRED_LOGGED.add(bare)
        logger.info("Solstice no longer serves %s; sending %s instead (pick a model with `hermes model`)",
                    bare, FALLBACK_MODELS[0])
    return FALLBACK_MODELS[0]


def _thinking_levels(model: str | None) -> tuple[tuple[str, ...], str] | None:
    """The listed model's (levels, default), or None for an id typed outside the list. Keyed on the id the
    client sends (``bare_gemini_model_id``: ``gemini/`` and ``google/`` stripped), so a prefixed listed
    model gets its own levels and default."""
    from agent.reasoning_effort import bare_gemini_model_id

    return _THINKING_LEVELS.get(bare_gemini_model_id(model).lower())


def _setup_efforts(model: str) -> tuple[str, ...] | None:
    """Levels the setup effort picker offers: exactly the listed model's own, else the generic ladder."""
    known = _thinking_levels(current_model_id(model) or model)
    return known[0] if known else None


def _thinking_config(model: str | None, reasoning_config: Any) -> dict[str, Any] | None:
    """``thinking_config`` for a listed model, None for any other id. Unset effort sends the model's
    default level with thoughts on, because thought summaries stream only when requested.
    Thinking off cannot switch these models off; it hides thoughts at the model's lowest level (titles)."""
    from agent.reasoning_effort import clamp_effort

    known = _thinking_levels(model)
    if known is None:
        return None
    levels, default = known
    rc = reasoning_config if isinstance(reasoning_config, dict) else {}
    effort = str(rc.get("effort") or "").strip().lower()
    if rc.get("enabled") is False or effort in ("none", "disabled"):
        return {"includeThoughts": False, "thinkingLevel": levels[0]}
    # Nearest weaker allowed level, else the floor (minimal -> low on Gemini 4, xhigh/max -> high on
    # Carbon); a name off the ladder (clamp_effort returns it verbatim) gets the default.
    level = clamp_effort(effort, levels) if effort else default
    return {"includeThoughts": True, "thinkingLevel": level if level in levels else default}


# The broker owns client_id/scope (discovered per login); refresh needs neither, the broker adds them.
_BASE_CFG = OAuthPKCEConfig(
    client_id="", authorize_url="", token_url="", label="Solstice",
    redirect_path="/gemini-auth/callback", token_request=broker_token_request,
    # A refresh token is issued only for offline access, and Google omits it on a repeat consent.
    extra_authorize_params={"access_type": "offline", "prompt": "consent"},
)


def _auth_handler(action: str, args: Any) -> bool:
    cfg = _BASE_CFG
    if action == "add":
        client = discover_client()
        cfg = replace(_BASE_CFG, client_id=client["client_id"], authorize_url=client["authorize_url"],
                      scopes=tuple(client["scope"].split()))
    return pkce_auth_handler(cfg)(action, args)


class SolsticeProfile(ProviderProfile):
    def create_client(self, **client_kwargs: Any) -> Any:
        from .account_client import SolsticeClient

        allowed = {"api_key", "base_url", "default_headers", "timeout", "http_client"}
        return SolsticeClient(**{k: v for k, v in client_kwargs.items() if k in allowed})

    def build_extra_body(self, *, session_id: str | None = None, **context: Any) -> dict[str, Any]:
        """Native ``thinking_config`` (the client reads only that key): the per-model table for a listed
        model, the ``gemini`` provider's shared mapping for any other id a user types."""
        model, reasoning_config = context.get("model") or "", context.get("reasoning_config")
        model = current_model_id(model) or model
        config = _thinking_config(model, reasoning_config)
        if config is not None:
            return {"thinking_config": config}
        from agent.reasoning_effort import gemini_thinking_config

        thinking_config = gemini_thinking_config(model, reasoning_config)
        return {"thinking_config": thinking_config} if thinking_config else {}

    def normalize_model_id(self, model: str, *, api_key: str = "") -> str | None:
        """The default model for a retired id, so a saved one is upgraded where Hermes resolves the model;
        None (core's generic rules) for every other id."""
        return current_model_id(model)

    def get_max_tokens(self, model: str | None) -> int | None:
        return None  # output caps are the backend's; a Hermes default would truncate long answers


solstice = SolsticeProfile(
    name="solstice", aliases=("solstice-oauth",), display_name="Solstice",
    description="Solstice (your own per-user quota, OAuth sign-in)",
    auth_type="oauth_external", base_url=INFERENCE_BASE_URL, hidden=True,
    supports_health_check=False, supports_model_listing=False, supports_vision=True,
    fallback_models=FALLBACK_MODELS, default_aux_model="gemini-flash-lite-latest",
    setup_reasoning_efforts=_setup_efforts,
    auth_handler=_auth_handler, refresh_credential=pkce_refresh_credential(_BASE_CFG),
    picker_after="router", registry_after="router.com",
)
