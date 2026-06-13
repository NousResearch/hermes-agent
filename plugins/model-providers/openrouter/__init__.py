"""OpenRouter provider profile."""

import logging
from typing import Any

from providers import register_provider
from providers.base import ProviderProfile

logger = logging.getLogger(__name__)

_CACHE: list[str] | None = None


def _coerce_bool(value: Any, *, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on", "enabled"}:
            return True
        if normalized in {"0", "false", "no", "off", "disabled"}:
            return False
    return bool(value)


def _positive_int(value: Any, *, max_value: int | None = None) -> int | None:
    if value in {None, ""}:
        return None
    try:
        coerced = int(value)
    except (TypeError, ValueError):
        return None
    if coerced <= 0:
        return None
    if max_value is not None and coerced > max_value:
        return None
    return coerced


def _coerce_temperature(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        coerced = float(value)
    except (TypeError, ValueError):
        return None
    # OpenRouter forwards to many upstreams; keep the common OpenAI-compatible
    # range so Hermes never sends clearly invalid Fusion judge temperatures.
    if 0.0 <= coerced <= 2.0:
        return coerced
    return None


def _coerce_model_list(value: Any) -> list[str]:
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",")]
    elif isinstance(value, (list, tuple)):
        items = [str(item).strip() for item in value]
    else:
        return []
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result[:8]


def normalize_fusion_config(value: Any) -> dict[str, Any] | None:
    """Validate ``openrouter.fusion`` into OpenRouter's Fusion tool shape.

    The supported config mirrors OpenRouter's ``openrouter:fusion`` server
    tool parameters: analysis model panel, judge model, tool-call budget,
    judge completion budget, reasoning config, and judge temperature.
    Invalid optional values are dropped locally.
    """
    if not isinstance(value, dict):
        return None

    if not _coerce_bool(value.get("enabled"), default=False):
        return None

    parameters: dict[str, Any] = {}
    analysis_models = _coerce_model_list(value.get("analysis_models"))
    if analysis_models:
        parameters["analysis_models"] = analysis_models

    judge = value.get("judge")
    if isinstance(judge, dict):
        judge_model = str(judge.get("model") or "").strip()
    else:
        judge_model = str(value.get("judge_model") or value.get("model") or "").strip()
    if judge_model:
        parameters["model"] = judge_model

    coerced = _positive_int(value.get("max_tool_calls"), max_value=16)
    if coerced is not None:
        parameters["max_tool_calls"] = coerced

    coerced = _positive_int(value.get("max_completion_tokens"))
    if coerced is not None:
        parameters["max_completion_tokens"] = coerced

    reasoning = value.get("reasoning")
    if isinstance(reasoning, dict) and reasoning:
        parameters["reasoning"] = dict(reasoning)

    temperature = _coerce_temperature(value.get("temperature"))
    if temperature is not None:
        parameters["temperature"] = temperature

    return {
        "tool": {"type": "openrouter:fusion", "parameters": parameters},
        "tool_choice": "required" if _coerce_bool(value.get("force"), default=False) else None,
    }


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

    def build_server_tools(
        self, *, model: str | None = None, **context: Any
    ) -> list[dict[str, Any]]:
        fusion = normalize_fusion_config(context.get("openrouter_fusion"))
        if not fusion:
            return []
        return [fusion["tool"]]

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
        if supports_reasoning:
            if reasoning_config is not None:
                extra_body["reasoning"] = dict(reasoning_config)
            else:
                extra_body["reasoning"] = {"enabled": True, "effort": "medium"}

        extra_headers: dict[str, Any] = {}
        if session_id and model and model.startswith(("x-ai/grok-", "xai/grok-")):
            extra_headers["x-grok-conv-id"] = session_id

        fusion = normalize_fusion_config(context.get("openrouter_fusion"))
        if fusion and fusion.get("tool_choice"):
            top_level = {"tool_choice": fusion["tool_choice"]}
        else:
            top_level = {}

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
