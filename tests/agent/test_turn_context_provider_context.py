"""Provider-native contexts must not trigger Hermes preflight compaction."""

from types import SimpleNamespace

import pytest

from agent.turn_context import provider_owns_context_for_auto_compression


def _agent(**kwargs):
    return SimpleNamespace(**kwargs)


def test_claude_agent_sdk_owns_automatic_context():
    assert provider_owns_context_for_auto_compression(
        _agent(api_mode="claude_agent_sdk")
    )


@pytest.mark.parametrize("mode", ["native", "off", "NATIVE"])
def test_codex_native_modes_remain_provider_owned(mode):
    assert provider_owns_context_for_auto_compression(
        _agent(api_mode="codex_app_server", codex_app_server_auto_compaction=mode)
    )


def test_codex_hermes_mode_remains_eligible_for_preflight():
    assert not provider_owns_context_for_auto_compression(
        _agent(api_mode="codex_app_server", codex_app_server_auto_compaction="hermes")
    )


@pytest.mark.parametrize("api_mode", ["openai", "anthropic", "", None])
def test_other_runtimes_remain_eligible_for_preflight(api_mode):
    assert not provider_owns_context_for_auto_compression(_agent(api_mode=api_mode))
