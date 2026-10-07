"""Tencent TokenHub provider profile (OpenAI-compatible, tokenhub.tencentmaas.com).

Request quirk: top-level ``reasoning_effort``, clamped to TokenHub's low/medium/high and omitted
when thinking is off; a main-loop turn with no configured effort sends ``high``.

Endpoint, ``TOKENHUB_API_KEY``, aliases and the static Hy model list stay in
``hermes_cli/auth.py`` / ``hermes_cli/models_catalog_static.py``; model listing and the doctor
probe stay off so the picker keeps the curated list.
"""

from typing import Any

from agent.reasoning_effort import TOKENHUB_EFFORTS, clamp_effort, requested_effort
from providers import register_provider
from providers.base import ProviderProfile


class TokenHubProfile(ProviderProfile):
    """TokenHub: top-level ``reasoning_effort``; main-loop turns default to ``high``."""

    def default_reasoning_config(self, model: str | None = None) -> dict | None:
        """Unset effort on a main-loop turn: ``high`` (TokenHub's own default is weaker).
        Auxiliary calls never take this default, so an unset aux call sends no effort."""
        return {"enabled": True, "effort": "high"}

    def build_api_kwargs_extras(
        self, *, reasoning_config: dict | None = None, **context: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        # No config (an auxiliary call with no effort configured): send nothing. Main-loop turns
        # reach here with ``default_reasoning_config`` filled in.
        if not reasoning_config or not isinstance(reasoning_config, dict) or reasoning_config.get("enabled") is False:
            return {}, {}
        effort = requested_effort(reasoning_config)
        return {}, {"reasoning_effort": "high" if effort is None else clamp_effort(effort, TOKENHUB_EFFORTS)}


register_provider(TokenHubProfile(
    name="tencent-tokenhub", display_name="Tencent TokenHub", base_url="https://tokenhub.tencentmaas.com/v1",
    supports_model_listing=False, supports_health_check=False,
))
