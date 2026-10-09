"""Copilot / GitHub Models provider profile.

Core routes GPT-5+/Codex -> codex_responses and Claude -> anthropic_messages;
this profile covers the chat_completions remainder: editor attribution headers
(providers.github.copilot_request_headers) and catalog-gated GitHub Models reasoning.
"""

import re
from typing import Any

from providers import register_provider
from providers.base import ProviderProfile
from providers.model_normalizers import normalize_copilot_id


class CopilotProfile(ProviderProfile):
    """GitHub Copilot / GitHub Models — editor headers + reasoning."""

    def resolve_route_policy(self, model: str, base_url: str = "", *, options=None) -> str | None:
        """Use Responses for Copilot's GPT-5+ models, except the chat-only mini tier."""
        del base_url
        normalized = normalize_copilot_id(model, ())
        match = re.match(r"^gpt-(\d+)", normalized.lower())
        if match and int(match.group(1)) >= 5 and not normalized.lower().startswith("gpt-5-mini"):
            return "codex_responses"
        return None

    def normalize_model_id(self, model: str, *, known_ids=()) -> str:
        return normalize_copilot_id(model, known_ids)

    def build_api_kwargs_extras(
        self, *, model: str | None = None, reasoning_config: dict | None = None,
        supports_reasoning: bool = False, **ctx,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if not (supports_reasoning and model):
            return {}, {}
        try:
            from models.metadata.github import clamp_github_reasoning_effort, github_model_reasoning_efforts

            supported = github_model_reasoning_efforts(model)
            if not supported:
                return {}, {}
            if not reasoning_config:
                return {"reasoning": {"effort": "medium"}}, {}
            # Never drop straight to medium, which inverted the ladder (ultra < high). See #74295.
            effort = clamp_github_reasoning_effort(reasoning_config.get("effort"), supported)
            return {"reasoning": {"effort": effort}}, {}
        except Exception:
            return {}, {}


copilot = CopilotProfile(
    name="copilot", aliases=("github-copilot", "github-models", "github-model", "github"),
    display_name="GitHub Copilot", description="GitHub Copilot (Uses GITHUB_TOKEN or gh auth token)",
    signup_url="https://github.com/settings/tokens",
    env_vars=("COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN"), base_url="https://api.githubcopilot.com",
    base_url_env_var="COPILOT_API_BASE_URL",
    auth_type="copilot",
)

register_provider(copilot)
