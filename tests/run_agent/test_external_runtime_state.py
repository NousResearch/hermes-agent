from unittest.mock import patch
import time

from agent.runtime_circuit import open_runtime_circuit
from run_agent import AIAgent


def _make_agent(*, runtime="claude_agent_sdk", fallback_model=None):
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
    ):
        return AIAgent(
            api_key="test-key" if runtime == "hermes" else None,
            base_url="https://example.invalid/v1" if runtime == "hermes" else None,
            provider="anthropic",
            model="claude-sonnet-4-6",
            runtime=runtime,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model=fallback_model,
        )


def test_subscription_runtime_needs_no_api_key_and_is_snapshotted():
    agent = _make_agent()

    assert agent.runtime == "claude_agent_sdk"
    assert agent.api_key == ""
    assert agent.client is None
    assert agent._primary_runtime["runtime"] == "claude_agent_sdk"
    assert agent._current_main_runtime()["runtime"] == "claude_agent_sdk"


def test_each_fallback_keeps_its_own_runtime_configuration():
    agent = _make_agent(
        fallback_model=[
            {
                "provider": "anthropic",
                "model": "claude-opus-4-6",
                "runtime": "claude_agent_sdk",
            },
            {
                "provider": "openai-codex",
                "model": "gpt-5.4",
                "openai_runtime": "codex_app_server",
            },
        ]
    )

    assert [entry["runtime"] for entry in agent._fallback_chain] == [
        "claude_agent_sdk",
        "codex_app_server",
    ]


def test_external_fallback_activates_without_provider_credentials():
    agent = _make_agent(
        runtime="hermes",
        fallback_model={
            "provider": "anthropic",
            "model": "claude-opus-4-6",
            "runtime": "claude_agent_sdk",
        },
    )

    assert agent._try_activate_fallback() is True
    assert agent.runtime == "claude_agent_sdk"
    assert agent.model == "claude-opus-4-6"
    assert agent.client is None


def test_restore_primary_restores_agent_loop_runtime():
    agent = _make_agent(
        fallback_model={
            "provider": "anthropic",
            "model": "claude-opus-4-6",
            "runtime": "claude_agent_sdk",
        }
    )
    agent.runtime = "hermes"
    agent._fallback_activated = True

    assert agent._restore_primary_runtime() is True
    assert agent.runtime == "claude_agent_sdk"


def test_reset_aware_primary_circuit_prevents_early_restore():
    agent = _make_agent(
        fallback_model={
            "provider": "openai-codex",
            "model": "gpt-5.4",
            "runtime": "codex_app_server",
        }
    )
    open_runtime_circuit(agent, reset_at=time.time() + 3600)
    assert agent._try_activate_fallback() is True
    agent._rate_limited_until = 0

    assert agent._restore_primary_runtime() is False
    assert agent.runtime == "codex_app_server"


def test_runtime_circuit_survives_a_fresh_agent_process_state(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    first = _make_agent()
    first._claude_max_attestation = type("Auth", (), {"account_key": "account-a"})()
    expected = open_runtime_circuit(first, reset_at=time.time() + 3600)

    second = _make_agent()
    second._claude_max_attestation = type("Auth", (), {"account_key": "account-a"})()

    from agent.runtime_circuit import runtime_circuit_open_until

    assert runtime_circuit_open_until(second) == expected


def test_fresh_worker_attests_before_account_scoped_circuit_lookup(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    first = _make_agent()
    first._claude_max_attestation = type("Auth", (), {"account_key": "account-a"})()
    open_runtime_circuit(first, reset_at=time.time() + 3600)
    second = _make_agent()

    def attest(agent):
        agent._claude_max_attestation = type(
            "Auth", (), {"account_key": "account-a"}
        )()
        return None

    with (
        patch("agent.external_runtime.prepare_claude_agent_sdk_runtime", side_effect=attest),
        patch("agent.external_runtime.run_claude_agent_sdk_attempt") as attempt,
    ):
        result = second.run_conversation("do the card")

    assert result["completed"] is False
    assert "circuit open" in result["error"]
    attempt.assert_not_called()
