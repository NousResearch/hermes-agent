"""A cron job's own ``reasoning_effort`` survives a mid-run provider fallback.

The job's level overrides the global ``agent.reasoning_effort`` and the per-model
``agent.reasoning_overrides`` for its runs (website/docs/user-guide/features/cron.md). Fallback
activation re-resolves those two for the fallback model (#21256), so the agent the scheduler builds
must be marked ``reasoning_pinned`` when the job set a level, and only then.
"""

from unittest.mock import MagicMock, patch

import pytest

from cron.scheduler import _construct_cron_agent, _CronAgentSetup, _resolve_job_reasoning_config
from hermes_constants import get_hermes_home
from run_agent import AIAgent

_FALLBACK = {"provider": "openai", "model": "fb-model"}
_CFG = {"agent": {"reasoning_effort": "medium", "reasoning_overrides": {"fb-model": "low"}}}


@pytest.fixture(autouse=True)
def profile_config():
    """Fallback activation reads the profile's levels from config.yaml."""
    (get_hermes_home() / "config.yaml").write_text(
        "agent:\n  reasoning_effort: medium\n  reasoning_overrides:\n    fb-model: low\n", encoding="utf-8")


def _cron_agent(job):
    setup = _CronAgentSetup(
        model="primary-model",
        runtime={"api_key": "k", "base_url": "https://openrouter.ai/api/v1", "provider": "openrouter",
                 "api_mode": "chat_completions"},
        reasoning_config=_resolve_job_reasoning_config(job, _CFG, "primary-model"),
        fallback_model=[_FALLBACK],
    )
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = _construct_cron_agent(AIAgent, job, _CFG, setup, workdir=None, session_id="cron-test",
                                      session_db=None)
    agent.client = MagicMock()
    return agent


def _fallback_level(agent):
    client = MagicMock()
    client.base_url, client.api_key = "https://api.openai.com/v1", "fb-key"
    with patch("agent.auxiliary_client.resolve_provider_client", return_value=(client, _FALLBACK["model"])):
        assert agent._try_activate_fallback() is True
    assert agent.model == _FALLBACK["model"]
    return agent.reasoning_config


def test_a_jobs_own_level_survives_fallback():
    assert _fallback_level(_cron_agent({"id": "j1", "reasoning_effort": "high"})) == {
        "enabled": True, "effort": "high"}


@pytest.mark.parametrize("job", [{"id": "j2"}, {"id": "j3", "reasoning_effort": "turbo"}],
                         ids=["no-level", "unparseable-level"])
def test_a_job_without_its_own_level_resolves_the_fallback_models_config(job):
    assert _fallback_level(_cron_agent(job)) == {"enabled": True, "effort": "low"}
