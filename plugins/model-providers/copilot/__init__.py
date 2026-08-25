"""Copilot / GitHub Models provider profile.

Core routes GPT-5+/Codex -> codex_responses and Claude -> anthropic_messages;
this profile covers the chat_completions remainder: editor attribution headers
(copilot_default_headers()) and catalog-gated GitHub Models reasoning.
"""

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile


class CopilotProfile(ProviderProfile):
    """GitHub Copilot / GitHub Models — editor headers + reasoning."""

    def build_api_kwargs_extras(
        self, *, model: str | None = None, reasoning_config: dict | None = None,
        supports_reasoning: bool = False, **ctx,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if not (supports_reasoning and model):
            return {}, {}
        try:
            from hermes_cli.models import clamp_reasoning_effort_to_supported, github_model_reasoning_efforts

                supported_efforts = github_model_reasoning_efforts(model)
                if supported_efforts and reasoning_config:
                    effort = reasoning_config.get("effort", "medium")
                    # Honor the requested level when the live Copilot catalog
                    # lists it as supported: gpt-5.5/gpt-5.4 DO support
                    # ``xhigh``. Otherwise clamp to the nearest WEAKER
                    # supported level via the shared ladder helper — the old
                    # ad-hoc rules dropped everything unrecognized to
                    # ``medium``, which inverted the ladder: ``ultra`` (the
                    # strongest ask) resolved weaker than an explicit
                    # ``high`` (#74295).
                    if effort not in supported_efforts:
                        from hermes_cli.models import (
                            clamp_reasoning_effort_to_supported,
                        )

                        effort = clamp_reasoning_effort_to_supported(
                            effort, list(supported_efforts)
                        )
                        if effort not in supported_efforts:
                            # Unrecognized/bespoke level the ladder can't
                            # place — fall back to medium, then to the
                            # catalog's first entry.
                            effort = (
                                "medium"
                                if "medium" in supported_efforts
                                else supported_efforts[0]
                            )
                    if effort in supported_efforts:
                        extra_body["reasoning"] = {"effort": effort}
                elif supported_efforts:
                    extra_body["reasoning"] = {"effort": "medium"}
            except Exception:
                pass
        return extra_body, {}


copilot = CopilotProfile(
    name="copilot", aliases=("github-copilot", "github-models", "github-model", "github"),
    env_vars=("COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN"), base_url="https://api.githubcopilot.com",
    auth_type="copilot",
)

register_provider(copilot)
