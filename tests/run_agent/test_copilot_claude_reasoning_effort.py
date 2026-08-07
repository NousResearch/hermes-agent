"""Claude-on-Copilot reasoning effort, resolved the way production resolves it.

The sibling tests in ``test_run_agent.py`` and ``test_copilot_profile.py`` stub
``github_model_reasoning_efforts`` itself, so they pin the effort *mapping* but
say nothing about whether a Claude slot ever reaches it. It did not: the runtime
called that resolver with neither catalog nor API key, and its keyless fallback
recognizes only o-series and GPT-5 IDs (``hermes_cli/models.py``
``_github_reasoning_efforts_for_model_id``). Every ``claude-*`` model therefore
resolved to ``[]``, the supports-reasoning gate went False, and the payload was
dropped before any degradation ran — the exact route #74295 was filed against.

These tests stub only the network boundary (``fetch_github_model_catalog``) and
drive the real capability-resolution path from there: the agent's cached
resolver, the supports-reasoning gate, and the request the chat_completions
transport actually builds.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent

# Live capabilities from the #74295 report, in the shape the Copilot
# ``/models`` payload uses.
_OPUS_5_EFFORTS = ["low", "medium", "high", "xhigh", "max"]
_OPUS_46_EFFORTS = ["low", "medium", "high", "max"]


def _catalog(*entries: tuple[str, list[str]]) -> list[dict]:
    return [
        {
            "id": model_id,
            "capabilities": {"supports": {"reasoning_effort": list(efforts)}},
        }
        for model_id, efforts in entries
    ]


@pytest.fixture
def copilot_agent():
    """An agent on the Copilot route, with tool loading and the client stubbed."""
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="ghu_test_copilot_token",
            base_url="https://api.githubcopilot.com",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.client = MagicMock()
    agent.provider = "copilot"
    agent.model = "claude-opus-5"
    return agent


@pytest.fixture
def catalog_fetch(monkeypatch):
    """Stub the one network call, and count it."""
    import hermes_cli.models as models_mod

    fetch = MagicMock(return_value=_catalog(
        ("claude-opus-5", _OPUS_5_EFFORTS),
        ("claude-opus-4.6", _OPUS_46_EFFORTS),
    ))
    monkeypatch.setattr(models_mod, "fetch_github_model_catalog", fetch)
    return fetch


class TestClaudeCopilotCapabilityResolution:
    """The catalog reaches the runtime, so Claude slots resolve to real levels."""

    def test_claude_slot_resolves_catalog_efforts(self, copilot_agent, catalog_fetch):
        assert copilot_agent._copilot_reasoning_efforts_cached() == _OPUS_5_EFFORTS

    def test_supports_reasoning_gate_opens_for_claude(self, copilot_agent, catalog_fetch):
        """Without the catalog this gate was False and dropped the payload."""
        assert copilot_agent._supports_reasoning_extra_body() is True

    def test_catalog_is_fetched_once_per_model(self, copilot_agent, catalog_fetch):
        """Recovering the capabilities must not cost a fetch per turn."""
        for _ in range(3):
            copilot_agent._copilot_reasoning_efforts_cached()
        assert catalog_fetch.call_count == 1

    def test_model_switch_refetches(self, copilot_agent, catalog_fetch):
        """The cache is keyed on the model, so a ``/model`` swap re-resolves."""
        copilot_agent._copilot_reasoning_efforts_cached()
        copilot_agent.model = "claude-opus-4.6"
        assert copilot_agent._copilot_reasoning_efforts_cached() == _OPUS_46_EFFORTS

    def test_unavailable_catalog_falls_back_to_heuristics(self, copilot_agent, monkeypatch):
        """A catalog outage degrades to the keyless behaviour, never raises."""
        import hermes_cli.models as models_mod

        monkeypatch.setattr(
            models_mod, "fetch_github_model_catalog", lambda **_kw: None
        )
        assert copilot_agent._copilot_reasoning_efforts_cached() == []
        assert copilot_agent._supports_reasoning_extra_body() is False

    def test_non_copilot_route_does_not_fetch(self, copilot_agent, catalog_fetch):
        """Only Copilot-hosted routes are worth a catalog round-trip."""
        copilot_agent.base_url = "https://openrouter.ai/api/v1"
        copilot_agent._base_url_lower = copilot_agent.base_url.lower()
        copilot_agent.model = "anthropic/claude-opus-5"

        assert copilot_agent._copilot_reasoning_efforts_cached() == []
        assert catalog_fetch.call_count == 0


class TestClaudeCopilotRequestBuilding:
    """End to end: what the chat_completions transport puts on the wire."""

    def _reasoning(self, agent, effort: str) -> dict | None:
        agent.reasoning_config = {"enabled": True, "effort": effort}
        kwargs = agent._build_api_kwargs([{"role": "user", "content": "hi"}])
        return kwargs.get("extra_body", {}).get("reasoning")

    def test_ultra_sends_max_not_medium(self, copilot_agent, catalog_fetch):
        """#74295: the strongest picker entry used to send less than ``high``.

        Pre-fix this assertion failed twice over — ``ultra`` mapped to
        ``medium``, and before that the payload was omitted entirely.
        """
        assert self._reasoning(copilot_agent, "ultra") == {"effort": "max"}

    def test_supported_effort_is_forwarded_verbatim(self, copilot_agent, catalog_fetch):
        assert self._reasoning(copilot_agent, "xhigh") == {"effort": "xhigh"}

    def test_capped_model_steps_down_one_rung(self, copilot_agent, catalog_fetch):
        """opus-4.6 lists no ``xhigh``; the request lands on ``high``, not ``medium``."""
        copilot_agent.model = "claude-opus-4.6"
        assert self._reasoning(copilot_agent, "xhigh") == {"effort": "high"}

    def test_request_ladder_is_monotonic(self, copilot_agent, catalog_fetch):
        """A stronger request never puts a weaker effort on the wire."""
        from hermes_constants import VALID_REASONING_EFFORTS

        sent = [
            self._reasoning(copilot_agent, effort)["effort"]
            for effort in VALID_REASONING_EFFORTS
        ]
        ranks = [_OPUS_5_EFFORTS.index(value) for value in sent]
        assert ranks == sorted(ranks), dict(zip(VALID_REASONING_EFFORTS, sent))
