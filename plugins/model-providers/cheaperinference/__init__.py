"""Cheaper Inference (cheaperinference.com) provider profile.

An OpenAI-compatible gateway: each request is served by one of several upstream
providers for the requested model, ranked by discount, speed or a balance of the
two, at or below the model maker's list price. Declarative — the standard
``chat_completions`` transport covers the wire, so there are no hooks here.

Two gateway behaviours are worth knowing when reading this profile; neither
needs provider-specific code today:

* The gateway also exposes a stateless ``/v1/responses`` layer that REJECTS any
  request not sending ``store=false``. Hermes' ``codex_responses`` transport
  already pins ``store: False`` on every request and refuses to build one
  otherwise (``agent/codex_responses_adapter.py``), so pointing a route at that
  layer needs no change here. The profile stays on ``chat_completions`` because
  that is the gateway's primary surface, and an ``api_mode`` here is a default,
  not a mandate (``hermes_cli.providers.host_mandated_api_mode`` is for hosts
  that accept exactly one wire — this one accepts both).
* Responses carry a ``provider`` field naming the upstream that served them.
  ``agent/turn_usage.py`` already surfaces any such field as ``upstream=`` on the
  per-call log line, so the route is visible without a provider-specific hook.

Model ids are bare (``claude-opus-5``, not ``anthropic/claude-opus-5``), so the
picker's curated list and the per-endpoint context windows in
``agent/model_metadata._ENDPOINT_SCOPED_CONTEXT`` both key off bare ids. Every id in
``fallback_models`` was checked against the live catalogue on 2026-09-17 and all of
them are served.

``supports_vision`` here is the host-level default that lets the vision path engage at
all; it is not a claim about every model. Probed on 2026-09-17 by sending a 32x32
solid-colour PNG as a data: URI and asking for the colour, four colours per route: 34
of the 55 chat routes read the image, and of the ids curated below claude-sonnet-5,
glm-5.3, glm-5.3-flash and gpt-5-mini do not - they either reject the part or answer
without seeing it. Per-model truth comes from the models.dev catalogue this repo
already reads, whose cheaperinference entries carry those probed values; a user can
still pin ``providers.cheaperinference.models.<id>.supports_vision`` in config.
"""

from providers import register_provider
from providers.base import ProviderProfile

cheaperinference = ProviderProfile(
    name="cheaperinference", aliases=("cheaper-inference", "cheaperinference.com"),
    display_name="Cheaper Inference",
    description="Cheaper Inference — OpenAI-compatible gateway; each request routed to a discounted upstream at or below list price",
    signup_url="https://cheaperinference.com/docs",
    env_vars=("CHEAPERINFERENCE_API_KEY", "CHEAPERINFERENCE_BASE_URL"),
    base_url="https://api.cheaperinference.com/v1", auth_type="api_key",
    supports_vision=True, default_aux_model="glm-5.3-flash",
    # Entry [0] is what surfaces that choose without a picker land on
    # (``pick_silent_default_model``), so a capable low-cost model leads and the
    # priciest flagship does not become a silent default — the hazard
    # ``_SILENT_DEFAULT_PROVIDERS`` in hermes_cli/models_catalog_static.py exists for.
    fallback_models=(
        "gpt-5.6-luna", "claude-opus-5", "claude-sonnet-5", "gpt-6-astra", "gpt-5.6-sol", "gpt-5.6-terra",
        "grok-4.5", "kimi-k3", "qwen-3-8-max", "gemini-3.7-flash", "glm-5.3", "gpt-5-mini",
        "deepseek-v4.1-flash", "glm-5.3-flash", "deepseek-v4-flash-0731",
    ),
)

register_provider(cheaperinference)
