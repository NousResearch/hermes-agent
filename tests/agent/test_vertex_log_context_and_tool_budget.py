"""Regression tests for _client_log_context dynamic routed base_url and _budget_for_agent effective threshold scaling."""

from __future__ import annotations

from types import SimpleNamespace

from agent.client_lifecycle import ClientLifecycleMixin
from agent.tool_executor import _budget_for_agent


class _DummyAgent(ClientLifecycleMixin):
    def __init__(self) -> None:
        self.provider = "vertex"
        self.model = "google/gemini-4-argon"
        self.base_url = "https://aiplatform.googleapis.com/v1beta1/projects/uop-atlas/locations/global/endpoints/openapi"
        self._last_routed_base_url: str | None = None


def test_client_log_context_prefers_routed_base_url() -> None:
    agent = _DummyAgent()
    assert "locations/global" in agent._client_log_context()

    ew2_url = "https://europe-west2-aiplatform.googleapis.com/v1beta1/projects/uop-atlas/locations/europe-west2/endpoints/openapi"
    agent._last_routed_base_url = ew2_url
    ctx = agent._client_log_context()
    assert "locations/europe-west2" in ctx
    assert "locations/global" not in ctx


def test_budget_for_agent_scales_by_effective_threshold_tokens() -> None:
    # A 1M-context model whose compaction threshold is 40k tokens should scale tool budget by 40k, not 1M.
    agent_compacted = SimpleNamespace(
        context_compressor=SimpleNamespace(context_length=1_048_576, threshold_tokens=40_000)
    )
    b_compacted = _budget_for_agent(agent_compacted)
    assert b_compacted.default_result_size == 24_000
    assert b_compacted.turn_budget == 48_000
