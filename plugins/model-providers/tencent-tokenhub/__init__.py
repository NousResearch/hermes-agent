"""Tencent TokenHub provider profile (OpenAI-compatible, tokenhub.tencentmaas.com).

Request quirk: top-level ``reasoning_effort``, clamped to TokenHub's low/medium/high and omitted
when thinking is off; a main-loop turn with no configured effort sends ``high``.

Endpoint, ``TOKENHUB_API_KEY``, aliases and the static Hy model list stay in
``hermes_cli/auth.py`` / ``hermes_cli/models_catalog_static.py``. ``base_url`` is left empty so
model listing keeps today's paths: first-time setup probes ``{base}/models`` through the generic
fallback and the ``/model`` picker serves the curated list. The doctor probe stays off.
"""

from typing import Any

from agent.reasoning_effort import requested_effort, tokenhub_effort
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
        return {}, {"reasoning_effort": tokenhub_effort(requested_effort(reasoning_config))}


register_provider(TokenHubProfile(
    name="tencent-tokenhub", display_name="Tencent TokenHub", supports_health_check=False,
    env_vars=("TOKENHUB_API_KEY", "TOKENHUB_BASE_URL"),
))
