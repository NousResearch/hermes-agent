"""Copilot / GitHub Models provider profile.

Core routes GPT-5+/Codex -> codex_responses and Claude -> anthropic_messages;
this profile covers the chat_completions remainder: editor attribution headers
(copilot_default_headers()) and catalog-gated GitHub Models reasoning.
"""

import logging
from typing import Any

from providers import register_provider
from providers.base import ProviderProfile

# Same logger the resolver used in hermes_cli.auth, so existing log filters keep matching.
logger = logging.getLogger("hermes_cli.auth")


class CopilotProfile(ProviderProfile):
    """GitHub Copilot / GitHub Models — editor headers + reasoning."""

    def resolve_base_url(self, *, api_key: str, default_url: str, env_url: str, probe: bool = True) -> str:
        """Copilot's API base comes from the token-exchange response (endpoints.api, proxy-ep fallback),
        authoritative for Enterprise / proxied accounts; falls back to the registry default.
        ``probe=False`` skips the exchange."""
        base_url = super().resolve_base_url(api_key=api_key, default_url=default_url, env_url=env_url, probe=probe)
        if not probe:
            return base_url
        try:
            from hermes_cli.copilot_auth import resolve_copilot_token, get_copilot_api_token
            raw_token, _ = resolve_copilot_token()
            if raw_token:
                resolved = (get_copilot_api_token(raw_token)[1] or "").strip()
                if resolved:
                    base_url = resolved
        except Exception as exc:
            logger.debug("Copilot base URL resolution fell back to default: %s", exc, exc_info=True)
        return base_url

    def build_api_kwargs_extras(
        self, *, model: str | None = None, reasoning_config: dict | None = None,
        supports_reasoning: bool = False, **ctx,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if not (supports_reasoning and model):
            return {}, {}
        try:
            from hermes_cli.models import clamp_github_reasoning_effort, github_model_reasoning_efforts

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
    env_vars=("COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN"), base_url="https://api.githubcopilot.com",
    auth_type="copilot",
)

register_provider(copilot)
