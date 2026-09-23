"""FastMetal provider profile: request-body ``reasoning`` object, sent only to routes that expose it.

FastMetal (api.fastmetal.ai) is an OpenAI-compatible gateway that relays each model id to its
upstream provider and bills prepaid credit in JPY. Thinking is controlled by a ``reasoning`` object
in the request body (``{"effort": ...}`` or ``{"enabled": false}``); the top-level OpenAI
``reasoning_effort`` string is ignored on several routes (https://fastmetal.ai/docs/api/reasoning).

The object must reach ONLY routes that expose thinking controls. A non-reasoning route answers
any ``reasoning`` field — including ``{"enabled": false}`` — with HTTP 404 "No endpoints found
that can handle the requested parameters" (llama-3.1-8b-instruct, live 2026-09-22), which names no
reasoning field and so is never recovered by the reasoning-rejection retry. Unset reasoning is left
to the route's own default (Kimi K3 thinks at max, Claude Opus 4.8 thinks off): the profile never
invents a level.
"""

from __future__ import annotations

import json
import urllib.request
from typing import Any

from providers import register_provider
from providers.base import ProviderProfile

# Served ids listed under "Per-model defaults" at https://fastmetal.ai/docs/api/reasoning
# (2026-09-22). Their own rule: "A model missing from the table either has no thinking
# controls or the catalog has not been refreshed for it yet" — so an absent id gets no
# ``reasoning`` field rather than a guess.
_REASONING_ROUTES: frozenset[str] = frozenset({
    "anthropic-claude-fable-5", "anthropic-claude-fable-5-1", "anthropic-claude-opus-4-8",
    "anthropic-claude-opus-5", "anthropic-claude-sonnet-5",
    "deepseek-v4-flash", "deepseek-v4-flash-0731", "deepseek-v4-pro", "deepseek-v4.1-flash",
    "fugu-max", "fugu-ultra-v2",
    "gemini-3.1-pro-preview", "gemini-3.5-flash", "gemini-3.7-flash", "gemini-3.8-flash",
    "gemma-4-26b-a4b-it",
    "glm-4.7", "glm-4.7-flash", "glm-5", "glm-5.1", "glm-5.2", "glm-5.3", "glm-5.3-flash", "glm-5.3-flashx",
    "gpt-5-mini", "gpt-5.4-nano", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra",
    "gpt-6-astra", "gpt-6-astra-pro", "gpt-oss-20b",
    "grok-4.5", "grok-4.6", "grok-4.7",
    "hy3", "inkling", "kimi-k2.6", "kimi-k3", "ling-3.0-flash-vl", "mercury-2.5", "minimax-m2.7",
    "muse-glimmer-30b", "muse-spark-1.2", "muse-spark-1.3", "muse-spark-1.3-contributor",
    "nemotron-3-super-120b-a12b", "nex-n2.5-mini-free",
    "qwen3.6-27b", "qwen3.7-max", "qwen3.8-2.4t-a95b", "qwen3.8-27b",
})


def _flat_model_name(model: str | None) -> str:
    """Bare FastMetal id, tolerating a copied ``fastmetal/`` aggregator prefix."""
    return (model or "").strip().rsplit("/", 1)[-1].lower()


def route_exposes_reasoning(model: str | None) -> bool:
    return _flat_model_name(model) in _REASONING_ROUTES


def _amount(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


class FastMetalProfile(ProviderProfile):
    """FastMetal — ``extra_body.reasoning`` on reasoning routes; ``/key/info`` prepaid balance."""

    def build_api_kwargs_extras(
        self, *, reasoning_config: dict | None = None, model: str | None = None,
        supports_reasoning: bool = False, **context: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if not (supports_reasoning or route_exposes_reasoning(model)):
            return {}, {}
        if not isinstance(reasoning_config, dict):
            return {}, {}
        effort = str(reasoning_config.get("effort") or "").strip().lower()
        if reasoning_config.get("enabled") is False or effort == "none":
            return {"reasoning": {"enabled": False}}, {}
        if not effort:
            return {}, {}
        return {"reasoning": {"effort": effort}}, {}

    def fetch_account_usage(self, *, base_url: str | None = None, api_key: str | None = None):
        """Prepaid balance for ``/usage`` from ``GET /key/info`` (``info.spend`` / ``info.max_budget``,
        both in JPY). The route sits at the host root, not under ``/v1``."""
        from datetime import datetime, timezone

        from agent.account_usage import AccountUsageSnapshot, AccountUsageWindow
        from hermes_cli.runtime_provider import resolve_runtime_provider
        from hermes_cli.urllib_security import open_credentialed_url

        runtime = resolve_runtime_provider(requested=self.name, explicit_base_url=base_url, explicit_api_key=api_key)
        token = str(runtime.get("api_key", "") or "").strip()
        if not token:
            return None
        origin = str(runtime.get("base_url") or self.base_url).rstrip("/").removesuffix("/v1")
        request = urllib.request.Request(
            f"{origin}/key/info", headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
        )
        with open_credentialed_url(request, timeout=10.0) as response:
            info = (json.loads(response.read().decode()) or {}).get("info") or {}
        spend, budget = _amount(info.get("spend")), _amount(info.get("max_budget"))
        windows: list[AccountUsageWindow] = []
        details: list[str] = []
        if spend is not None and budget is not None and budget > 0:
            windows.append(AccountUsageWindow(
                label="Prepaid credit", used_percent=min(100.0, spend / budget * 100),
                detail=f"¥{max(0.0, budget - spend):,.2f} of ¥{budget:,.2f} remaining",
            ))
        elif spend is not None:
            details.append(f"Spend on this key: ¥{spend:,.2f}")
        return AccountUsageSnapshot(
            provider=self.name, source="key_info", fetched_at=datetime.now(timezone.utc),
            title="FastMetal credit", windows=tuple(windows), details=tuple(details),
        )


fastmetal = FastMetalProfile(
    name="fastmetal", aliases=("fast-metal", "fastmetal-ai"), display_name="FastMetal",
    description="FastMetal (OpenAI-compatible multi-model gateway, prepaid JPY billing)",
    signup_url="https://fastmetal.ai/dashboard", env_vars=("FASTMETAL_API_KEY", "FASTMETAL_BASE_URL"),
    base_url="https://api.fastmetal.ai/v1", auth_type="api_key",
    # No thinking tax on titles/compression; accepts tools and images (live-verified 2026-09-22).
    default_aux_model="gpt-4.1-nano",
    # Flat ids exactly as /v1/models returns them; the picker is live-first for this provider, so
    # this list is the setup default ([0] = FastMetal's own Hermes guide default) and offline fallback.
    fallback_models=(
        "anthropic-claude-sonnet-5", "anthropic-claude-opus-5", "gpt-6-astra", "gemini-3.1-pro-preview",
        "kimi-k3", "glm-5.3", "deepseek-v4-pro", "grok-4.7", "gemini-3.5-flash", "deepseek-v4-flash",
        "gpt-4.1-nano",
    ),
    # FastMetal is not in models.dev. Reasoning follows the table behind route_exposes_reasoning;
    # vision only where the model page lists ``image`` under input modalities.
    model_capabilities={
        "anthropic-claude-sonnet-5": {"supports_reasoning": True, "supports_vision": True},
        "anthropic-claude-opus-5": {"supports_reasoning": True, "supports_vision": True},
        "gpt-6-astra": {"supports_reasoning": True},
        "gemini-3.1-pro-preview": {"supports_reasoning": True, "supports_vision": True},
        "kimi-k3": {"supports_reasoning": True, "supports_vision": True},
        "glm-5.3": {"supports_reasoning": True},
        "deepseek-v4-pro": {"supports_reasoning": True, "supports_vision": False},
        "grok-4.7": {"supports_reasoning": True},
        "gemini-3.5-flash": {"supports_reasoning": True, "supports_vision": True},
        "deepseek-v4-flash": {"supports_reasoning": True, "supports_vision": False},
        "gpt-4.1-nano": {"supports_reasoning": False, "supports_vision": True},
    },
)

register_provider(fastmetal)
