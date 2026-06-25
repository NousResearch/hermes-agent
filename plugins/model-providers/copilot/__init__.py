"""Copilot / GitHub Models provider profile.

Copilot uses per-model api_mode routing:
  - GPT-5+ / Codex models → codex_responses
  - Claude models → anthropic_messages
  - Everything else → chat_completions (this profile covers that subset)

Key quirks for the chat_completions subset:
  - Editor attribution headers (via copilot_default_headers())
  - GitHub Models reasoning extra_body (model-catalog gated)
"""

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile


class CopilotProfile(ProviderProfile):
    """GitHub Copilot / GitHub Models — editor headers + reasoning."""

    def build_api_kwargs_extras(
        self,
        *,
        model: str | None = None,
        reasoning_config: dict | None = None,
        supports_reasoning: bool = False,
        api_key: str | None = None,
        **ctx,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        extra_body: dict[str, Any] = {}
        if supports_reasoning and model:
            try:
                # Resolve supported efforts through the cached catalog helper, not
                # the bare ``github_model_reasoning_efforts(model)``. The bare call
                # has no catalog/api_key, so it falls through to the static
                # GPT/o-series table and returns ``[]`` for Copilot-hosted Claude,
                # silently dropping ``reasoning_effort`` even though the live
                # ``/models`` catalog advertises it. ``get_copilot_reasoning_efforts``
                # consults the live catalog (1-hour cache) and degrades to the
                # static table only on fetch failure. (PR #51953 fixed the gate and
                # the legacy path but not this registered-profile path.)
                from hermes_cli.models import get_copilot_reasoning_efforts

                supported_efforts = get_copilot_reasoning_efforts(model, api_key)
                if supported_efforts and reasoning_config:
                    effort = reasoning_config.get("effort", "medium")
                    # Normalize non-standard effort levels to the nearest supported
                    if effort == "xhigh":
                        effort = "high"
                    if effort in supported_efforts:
                        extra_body["reasoning"] = {"effort": effort}
                elif supported_efforts:
                    extra_body["reasoning"] = {"effort": "medium"}
            except Exception:
                pass
        return extra_body, {}


copilot = CopilotProfile(
    name="copilot",
    aliases=("github-copilot", "github-models", "github-model", "github"),
    env_vars=("COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN"),
    base_url="https://api.githubcopilot.com",
    auth_type="copilot",
)

register_provider(copilot)
