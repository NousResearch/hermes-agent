"""OpenRouter provider profile."""

import logging
from typing import Any

from providers import register_provider
from providers.base import ProviderProfile

logger = logging.getLogger(__name__)

_CACHE: list[str] | None = None

# Anthropic model families that still accept an explicit "disable thinking"
# request (the manual ``thinking: {type: "disabled"}`` form OpenRouter emits
# for ``reasoning: {enabled: false}``). Everything Claude 4.6 and newer —
# including future date-stamped / named models (fable, mythos-class, …) —
# mandates reasoning and returns HTTP 400 on any disable form. We therefore
# default *unknown* Anthropic models to "cannot disable" (the modern contract)
# and keep only this explicit legacy allowlist of models that can. Mirrors the
# default-to-newest philosophy in agent/anthropic_adapter._get_anthropic_max_output.
_ANTHROPIC_REASONING_OPTIONAL_SUBSTRINGS = (
    "claude-3",          # 3, 3.5, 3.7
    "claude-opus-4-0", "claude-opus-4.0", "claude-opus-4-1", "claude-opus-4.1",
    "claude-sonnet-4-0", "claude-sonnet-4.0",
    "claude-opus-4-2025", "claude-sonnet-4-2025",  # date-stamped 4.0 IDs
    "claude-opus-4-5", "claude-opus-4.5",
    "claude-sonnet-4-5", "claude-sonnet-4.5",
    "claude-haiku-4-5", "claude-haiku-4.5",
)


def _anthropic_reasoning_is_mandatory(model: str | None) -> bool:
    """Return True for Anthropic models that reject any disable-thinking form.

    Claude 4.6+ (adaptive thinking) and newer named models have no "off"
    switch — sending ``reasoning: {enabled: false}`` makes OpenRouter emit
    ``thinking: {type: "disabled"}``, which these models 400 on. Unknown /
    new Anthropic model names default to mandatory so the next un-numbered
    release doesn't reintroduce the 400.
    """
    m = (model or "").lower()
    if not m.startswith(("anthropic/", "claude")) and "claude" not in m:
        return False
    return not any(sub in m for sub in _ANTHROPIC_REASONING_OPTIONAL_SUBSTRINGS)


class OpenRouterProfile(ProviderProfile):
    """OpenRouter aggregator — provider preferences, reasoning config passthrough."""

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """Fetch from public OpenRouter catalog — no auth required.

        Note: Tool-call capability filtering is applied by hermes_cli/models.py
        via fetch_openrouter_models() → _openrouter_model_supports_tools(), not
        here. The picker early-returns via the dedicated openrouter path before
        reaching this method, so filtering here would be unreachable.
        """
        global _CACHE  # noqa: PLW0603
        if _CACHE is not None:
            return _CACHE
        try:
            result = super().fetch_models(api_key=None, timeout=timeout)
            if result is not None:
                _CACHE = result
            return result
        except Exception as exc:
            logger.debug("fetch_models(openrouter): %s", exc)
            return None

    def build_extra_body(
        self, *, session_id: str | None = None, **context: Any
    ) -> dict[str, Any]:
        body: dict[str, Any] = {}
        if session_id:
            body["session_id"] = session_id
        prefs = context.get("provider_preferences")
        if prefs:
            body["provider"] = prefs

        # Pareto Code router — model-gated. The plugins block is only
        # meaningful for openrouter/pareto-code; sending it on any other
        # model has no documented effect and would be confusing in logs.
        # See: https://openrouter.ai/docs/guides/routing/routers/pareto-router
        model = (context.get("model") or "")
        if model == "openrouter/pareto-code":
            score = context.get("openrouter_min_coding_score")
            if score is not None and score != "":
                try:
                    score_f = float(score)
                except (TypeError, ValueError):
                    score_f = None
                if score_f is not None and 0.0 <= score_f <= 1.0:
                    body["plugins"] = [
                        {"id": "pareto-router", "min_coding_score": score_f}
                    ]
        return body

    def build_api_kwargs_extras(
        self,
        *,
        reasoning_config: dict | None = None,
        supports_reasoning: bool = False,
        model: str | None = None,
        session_id: str | None = None,
        **context: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """OpenRouter passes the full reasoning_config dict as extra_body.reasoning.

        For xAI Grok models routed through OpenRouter, attach the
        ``x-grok-conv-id`` header so that xAI's prompt cache stays pinned to
        the same backend server across turns.
        """
        extra_body: dict[str, Any] = {}
        top_level: dict[str, Any] = {}
        extra_headers: dict[str, Any] = {}
        if supports_reasoning:
            # Reasoning-mandatory Anthropic models (Claude 4.6+ / fable /
            # future named models) use *adaptive* thinking: the model decides
            # how much to think, and OpenRouter ignores ``reasoning.effort`` for
            # them entirely. Sending any ``reasoning`` field is therefore both
            # pointless and actively harmful:
            #   - ``{enabled: false}`` → OpenRouter emits Anthropic's manual
            #     ``thinking: {type: "disabled"}``, which these models 400 on.
            #   - any enabled form, on a tool-continuation turn whose prior
            #     assistant tool_call carries no thinking block (chat_completions
            #     never replays signed thinking blocks), ALSO makes OpenRouter
            #     emit ``thinking: {type: "disabled"}`` → the same 400 on every
            #     turn after the first tool call.
            # The only reliable behavior is to omit ``reasoning`` and let the
            # model default to adaptive. See hermes-agent#42991 (disable case)
            # and the tool-replay follow-up.
            #
            # ``reasoning.effort`` being ignored does NOT mean these models have
            # no effort lever — OpenRouter honors the requested effort on the
            # top-level ``verbosity`` field instead (it maps to Anthropic's
            # ``output_config.effort``; ``reasoning.effort`` is accepted but
            # ignored — confirmed by OpenRouter's Claude migration docs and a
            # live token-spend probe in hermes-agent#43432). Route the existing
            # ``reasoning_config["effort"]`` (sourced from
            # ``agent.reasoning_effort``) onto ``verbosity`` so the knob the user
            # already sets keeps working for these models. We still send NO
            # ``reasoning`` field, preserving the #42991 400 fix.
            if _anthropic_reasoning_is_mandatory(model):
                cfg = reasoning_config or {}
                effort = cfg.get("effort")
                # Only emit when effort is actually requested and reasoning
                # isn't explicitly disabled. Otherwise omit ``verbosity`` so the
                # model keeps its own adaptive default (``high``).
                if cfg.get("enabled", True) is not False and effort and effort != "none":
                    top_level["verbosity"] = effort
            elif reasoning_config is not None:
                extra_body["reasoning"] = dict(reasoning_config)
            else:
                extra_body["reasoning"] = {"enabled": True, "effort": "medium"}

        if session_id and model and model.startswith(("x-ai/grok-", "xai/grok-")):
            extra_headers["x-grok-conv-id"] = session_id
        if extra_headers:
            top_level["extra_headers"] = extra_headers

        return extra_body, top_level


openrouter = OpenRouterProfile(
    name="openrouter",
    aliases=("or",),
    env_vars=("OPENROUTER_API_KEY",),
    display_name="OpenRouter",
    description="OpenRouter — unified API for 200+ models",
    signup_url="https://openrouter.ai/keys",
    base_url="https://openrouter.ai/api/v1",
    models_url="https://openrouter.ai/api/v1/models",
    fallback_models=(
        "anthropic/claude-sonnet-4.6",
        "openai/gpt-5.4",
        "deepseek/deepseek-chat",
        "google/gemini-3-flash-preview",
        "qwen/qwen3-plus",
    ),
)

register_provider(openrouter)
