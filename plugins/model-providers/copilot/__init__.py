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
        **ctx,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        extra_body: dict[str, Any] = {}
        if supports_reasoning and model:
            try:
                from hermes_cli.models import github_model_reasoning_efforts

                supported_efforts = github_model_reasoning_efforts(model)
                if supported_efforts and reasoning_config:
                    effort = reasoning_config.get("effort", "medium")
                    # Honor the requested level when the live Copilot catalog
                    # lists it as supported: gpt-5.5/gpt-5.4 DO support
                    # ``xhigh``. Only downgrade levels the catalog does NOT
                    # list (e.g. ``xhigh``/``max`` on models capped lower, or
                    # ``minimal`` where unsupported), choosing the nearest
                    # weaker supported level rather than forwarding verbatim.
                    #
                    # (Previously this unconditionally mapped xhigh->high, a
                    #  stale guard that silently capped models which do support
                    #  the higher level.)
                    if effort not in supported_efforts:
                        if effort == "xhigh" and "high" in supported_efforts:
                            effort = "high"
                        elif effort == "minimal" and "low" in supported_efforts:
                            effort = "low"
                        elif "medium" in supported_efforts:
                            effort = "medium"
                        else:
                            effort = supported_efforts[0]
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
