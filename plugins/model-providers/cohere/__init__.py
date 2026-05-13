"""Native Cohere provider profile.

Cohere ships a v2 chat API at ``api.cohere.com/v2/chat`` that supports
tool calling, streaming, RAG documents with citations, hosted connectors
(e.g. web-search), safety modes, and extended-thinking models. This
profile routes Hermes through the ``cohere`` Python SDK
(``cohere.ClientV2``) via the ``cohere_chat`` api_mode and the
``CohereTransport`` in :mod:`agent.transports.cohere`.

The profile is declarative — it carries auth/endpoint metadata and a
``build_extra_body`` hook that surfaces Cohere-only request fields
(``safety_mode``, ``documents``, ``connectors``, ``citation_options``,
``force_single_step``, ``thinking``). Client construction lives in
``run_agent.py`` (via :mod:`agent.cohere_adapter`).
"""

from __future__ import annotations

import json
import logging
import urllib.request
from typing import Any

from providers import register_provider
from providers.base import ProviderProfile

logger = logging.getLogger(__name__)


# Hermes reasoning effort → Cohere thinking token_budget. Mirrors the
# THINKING_BUDGET map in anthropic_adapter so users get comparable
# behaviour when toggling between providers.
_COHERE_THINKING_BUDGET = {
    "low": 2048,
    "medium": 8192,
    "high": 16384,
    "xhigh": 32768,
}


def _model_supports_thinking(model: str) -> bool:
    """Return True when the model accepts Cohere's ``thinking`` field.

    Cohere documents extended thinking on the ``command-a-reasoning``
    family. Other Command models reject the field with HTTP 400, so
    we only attach it for reasoning-capable models.
    """
    m = (model or "").strip().lower()
    return "command-a-reasoning" in m or "command-r-reasoning" in m


class CohereProfile(ProviderProfile):
    """Cohere v2 chat — native SDK, native tool_plan + citations."""

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """Fetch chat-capable models from Cohere's ``/v1/models`` endpoint.

        Cohere accepts ``Bearer`` auth on the REST models endpoint and
        supports an ``endpoint=chat`` filter so we drop embed/rerank
        models. Returns ``None`` on any failure so the caller falls
        back to ``self.fallback_models``.
        """
        if not api_key:
            return None
        url = "https://api.cohere.com/v1/models?endpoint=chat&page_size=200"
        try:
            req = urllib.request.Request(url)
            req.add_header("Authorization", f"Bearer {api_key}")
            req.add_header("Accept", "application/json")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode())
            models = data.get("models", [])
            return [
                m["name"]
                for m in models
                if isinstance(m, dict) and isinstance(m.get("name"), str)
            ]
        except Exception as exc:
            logger.debug("fetch_models(cohere): %s", exc)
            return None

    def build_extra_body(
        self, *, session_id: str | None = None, **context: Any
    ) -> dict[str, Any]:
        """Surface Cohere-only request fields read from the call context.

        The transport's ``build_kwargs`` reads these straight from
        ``extra_body`` (rather than top-level kwargs) so the same plumbing
        path used by every other provider feeds through cleanly. Keys
        that are unset / empty are omitted so we never send empty
        ``documents=[]`` or ``connectors=[]`` to Cohere (the API treats
        the presence of those fields as an instruction).
        """
        body: dict[str, Any] = {}

        safety_mode = context.get("safety_mode")
        if isinstance(safety_mode, str) and safety_mode.strip():
            body["safety_mode"] = safety_mode.strip().upper()

        documents = context.get("documents")
        if isinstance(documents, list) and documents:
            body["documents"] = documents

        connectors = context.get("connectors")
        if isinstance(connectors, list) and connectors:
            body["connectors"] = connectors

        citation_options = context.get("citation_options")
        if isinstance(citation_options, dict) and citation_options:
            body["citation_options"] = citation_options

        if context.get("force_single_step") is True:
            body["force_single_step"] = True

        # Reasoning: Hermes uses a generic ``reasoning_config`` dict
        # ({"enabled": bool, "effort": "low|medium|high|xhigh"}). For
        # Command A Reasoning models we translate this to Cohere's
        # ``thinking`` field. ``cohere.thinking_token_budget`` from
        # config.yaml overrides the effort-derived budget.
        reasoning_config = context.get("reasoning_config")
        explicit_budget = context.get("thinking_token_budget")
        model = context.get("model") or ""
        if _model_supports_thinking(model):
            token_budget: int | None = None
            if isinstance(explicit_budget, int) and explicit_budget > 0:
                token_budget = explicit_budget
            elif isinstance(reasoning_config, dict) and reasoning_config.get("enabled", True):
                effort = str(reasoning_config.get("effort", "medium") or "medium").strip().lower()
                token_budget = _COHERE_THINKING_BUDGET.get(effort, _COHERE_THINKING_BUDGET["medium"])
            if token_budget is not None:
                body["thinking"] = {"type": "enabled", "token_budget": int(token_budget)}

        return body


cohere = CohereProfile(
    name="cohere",
    aliases=("command", "command-r", "command-a", "cohere-ai"),
    api_mode="cohere_chat",
    display_name="Cohere",
    description="Cohere — native v2 chat API (tool_plan, citations, connectors, safety_mode)",
    signup_url="https://dashboard.cohere.com/api-keys",
    env_vars=("COHERE_API_KEY", "CO_API_KEY"),
    base_url="https://api.cohere.com",
    hostname="api.cohere.com",
    auth_type="api_key",
    default_aux_model="command-r7b-12-2024",
    fallback_models=(
        "command-a-03-2025",
        "command-a-reasoning-08-2025",
        "command-r-plus-08-2024",
        "command-r-08-2024",
        "command-r7b-12-2024",
    ),
)

register_provider(cohere)
