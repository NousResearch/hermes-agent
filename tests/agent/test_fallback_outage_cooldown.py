"""After an outage-class failover the primary is benched for ``fallback.outage_cooldown_seconds``
and then gets one probe turn, instead of being restored (and re-failing) on every turn
(#46856)."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.error_classifier import FailoverReason
from agent.fallback_cooldown import outage_cooldown_seconds, probe_failed_fast
from hermes_constants import get_hermes_home
from run_agent import AIAgent


def _agent():
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key", provider="openrouter", base_url="https://openrouter.ai/api/v1",
            model="primary/model", quiet_mode=True, skip_context_files=True, skip_memory=True,
            fallback_model=[{"provider": "zai", "model": "glm-5.2"}],
        )
    agent.client = None
    return agent


def _fallback_client():
    client = SimpleNamespace(base_url="https://api.z.ai/api/paas/v4", api_key="fb-key")
    client.chat = SimpleNamespace(completions=SimpleNamespace(create=lambda *a, **k: None))
    return client


def _fail_over(agent, reason, cfg=None, now=1000.0):
    with (
        patch("agent.auxiliary_client.resolve_provider_client", return_value=(_fallback_client(), "glm-5.2")),
        patch("agent.fallback_cooldown.time.monotonic", return_value=now),
        patch("hermes_cli.config.load_config", return_value=cfg if cfg is not None else {}),
    ):
        assert agent._try_activate_fallback(reason=reason) is True


def _restore_at(agent, now):
    notices = []
    agent._emit_diagnostic_status = notices.append
    with (
        patch("agent.agent_runtime_helpers.time.monotonic", return_value=now),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        return agent._restore_primary_runtime(), notices


@pytest.mark.parametrize("reason", [
    FailoverReason.overloaded, FailoverReason.server_error, FailoverReason.timeout,
    FailoverReason.auth, FailoverReason.unknown,
])
def test_outage_failover_holds_the_fallback_then_probes_the_primary_once(reason):
    agent = _agent()
    _fail_over(agent, reason)

    assert _restore_at(agent, 1000.0 + 120)[0] is False
    assert agent.model == "glm-5.2"

    restored, notices = _restore_at(agent, 1000.0 + 301)
    assert restored is True and agent.model == "primary/model"
    assert agent._primary_probe_pending is True
    assert probe_failed_fast(agent, FailoverReason.overloaded) is True
    assert probe_failed_fast(agent, FailoverReason.format_error) is False
    assert notices and notices[0].startswith("🔁 Trying primary model primary/model")


def test_request_shape_failover_still_restores_the_primary_next_turn():
    agent = _agent()
    _fail_over(agent, FailoverReason.format_error)

    restored, notices = _restore_at(agent, 1001.0)

    assert restored is True
    assert agent._primary_probe_pending is False
    assert notices[0].startswith("✅ Primary model restored")


def test_zero_cooldown_keeps_the_restore_every_turn_behaviour():
    agent = _agent()
    _fail_over(agent, FailoverReason.overloaded, cfg={"fallback": {"outage_cooldown_seconds": 0}})

    assert _restore_at(agent, 1001.0)[0] is True


def test_a_longer_provider_reset_wins_over_the_configured_cooldown():
    agent = _agent()
    with (
        patch("agent.auxiliary_client.resolve_provider_client", return_value=(_fallback_client(), "glm-5.2")),
        patch("agent.fallback_cooldown.time.monotonic", return_value=1000.0),
        patch("agent.fallback_cooldown.time.time", return_value=50_000.0),
        patch("hermes_cli.config.load_config", return_value={"fallback": {"outage_cooldown_seconds": 60}}),
    ):
        assert agent._try_activate_fallback(reason=FailoverReason.overloaded, reset_at=50_900.0) is True

    assert agent._rate_limited_until == 1000.0 + 900


@pytest.mark.parametrize(("value", "expected"), [(45, 45.0), (10**9, 14400.0), (-5, 0.0), ("soon", 300.0)])
def test_outage_cooldown_reads_the_profile_config_yaml(value, expected):
    home = get_hermes_home()
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(f"fallback:\n  outage_cooldown_seconds: {value}\n", encoding="utf-8")

    assert outage_cooldown_seconds() == expected
