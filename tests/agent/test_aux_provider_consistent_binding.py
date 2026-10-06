"""Provider-consistent auxiliary routing after a primary-provider fallback.

Incident (t_929b7e1d): a goal_judge lane configured as ``provider: custom`` with its
own task key but no ``base_url`` inherited the turn-start main-runtime snapshot of an
``openai-codex`` primary and POSTed to the codex backend host. The response was a
Cloudflare challenge (PermissionDeniedError), the judge returned a transport failure,
and the goal gate rejected an otherwise valid handoff.

Two invariants are pinned here:

1. ``_resolve_custom_branch`` may reuse the live main-runtime endpoint/credential ONLY
   when that runtime is itself a custom endpoint (#45472 semantics). A non-custom or
   unidentified snapshot must fail safely (fall through to the configured custom
   endpoint resolution) and must never lend its host to a custom aux lane.
2. ``try_activate_fallback`` republishes the main-runtime snapshot, so aux lanes
   resolving through the contextvar for the remainder of the turn bind the activated
   fallback route — endpoint and credential identity from the same provider.

All tests are hermetic: no network, no live inference, temp HERMES_HOME, no env keys.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

PRIMARY_HOST = "chatgpt.com"
PRIMARY_BASE = "https://chatgpt.com/backend-api/codex"
FALLBACK_HOST = "dashscope.example"
FALLBACK_BASE = "https://dashscope.example/compatible-mode/v1"
FALLBACK_ENTRY = {
    "provider": "custom",
    "model": "qwen3.8-max",
    "base_url": FALLBACK_BASE,
    "key_env": "AUX_LANE_TEST_KEY",
}


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    """Hermetic profile: temp HERMES_HOME, no ambient endpoints, clean runtime state."""
    from agent import auxiliary_client as aux

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    (hermes_home / "config.yaml").write_text("model:\n  default: test-model\n")
    for var in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENROUTER_API_KEY",
                "NOUS_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    aux.clear_runtime_main()
    with aux._client_cache_lock:
        aux._client_cache.clear()
    yield aux
    aux.clear_runtime_main()
    with aux._client_cache_lock:
        aux._client_cache.clear()


def _codex_snapshot(aux):
    """Publish the turn-start snapshot of an openai-codex primary (the incident state)."""
    aux.set_runtime_main(
        "openai-codex", "gpt-5.6-sol",
        base_url=PRIMARY_BASE, api_key="primary-runtime-token",
        api_mode="codex_responses", session_id="s-test",
    )


def _make_agent(fallback_model):
    """Minimal offline AIAgent, shaped like tests/agent/test_provider_fallback.py."""
    from run_agent import AIAgent

    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            api_key="primary-runtime-token",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model=fallback_model,
        )
        agent.client = MagicMock()
    agent.provider = "openai-codex"
    agent.base_url = PRIMARY_BASE
    return agent


def _mock_fallback_client():
    client = MagicMock()
    client.base_url = FALLBACK_BASE
    client.api_key = "fallback-lane-token"
    return client


def _activate_fallback(agent):
    """Run one real try_activate_fallback onto the custom entry (client build stubbed)."""
    with (
        patch.dict(os.environ, {"AUX_LANE_TEST_KEY": "fallback-lane-token"}, clear=False),
        patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(_mock_fallback_client(), "qwen3.8-max"),
        ),
    ):
        assert agent._try_activate_fallback() is True


class TestCustomLaneNeverBindsForeignPrimaryEndpoint:
    """Invariant 1: a custom aux lane must not inherit a non-custom primary endpoint."""

    def test_stale_codex_snapshot_does_not_leak_into_custom_lane(self, _isolate):
        """RED regression: explicit non-custom runtime + lane key must NOT produce a
        client at the primary host; hermetic env means nothing else can resolve."""
        from agent.auxiliary_client import resolve_provider_client

        runtime = {
            "provider": "openai-codex",
            "model": "gpt-5.6-sol",
            "base_url": PRIMARY_BASE,
            "api_key": "primary-runtime-token",
        }
        client, _model = resolve_provider_client(
            "custom", model="qwen3.8-max",
            explicit_api_key="lane-task-key", main_runtime=runtime,
        )
        assert client is None or PRIMARY_HOST not in str(getattr(client, "base_url", ""))
        assert client is None, (
            "custom lane with no configured endpoint must fail safely, not bind "
            "the non-custom primary snapshot"
        )

    def test_unidentified_snapshot_fails_safely(self, _isolate):
        """A runtime snapshot without a provider identity is ambiguous → no inheritance."""
        from agent.auxiliary_client import resolve_provider_client

        client, _model = resolve_provider_client(
            "custom", model="qwen3.8-max",
            main_runtime={"base_url": PRIMARY_BASE, "api_key": "primary-runtime-token"},
        )
        assert client is None

    def test_contextvar_lane_resolution_is_safe_before_fallback(self, _isolate):
        """Integration level: with the stale codex snapshot in the contextvar, a
        goal_judge-shaped lane (provider custom, task key, no base_url) resolves to
        nothing instead of contacting the primary endpoint."""
        aux = _isolate
        _codex_snapshot(aux)
        client, _model = aux._get_cached_client(
            "custom", "qwen3.8-max", async_mode=False,
            api_key="lane-task-key", task="goal_judge",
        )
        assert client is None

    def test_custom_snapshot_still_inherits_endpoint(self, _isolate):
        """#45472 regression guard: a custom(-named) main runtime is still reused
        directly by the custom lane."""
        from agent.auxiliary_client import resolve_provider_client

        runtime = {
            "provider": "custom:gateway",
            "model": "glm-5.1",
            "base_url": "https://my-gateway.example.com/v1",
            "api_key": "gateway-key",
        }
        client, model = resolve_provider_client(
            "custom", model="glm-5.1", main_runtime=runtime,
        )
        assert client is not None
        assert model == "glm-5.1"
        assert "my-gateway.example.com" in str(client.base_url)
        assert client.api_key == "gateway-key"


class TestFallbackRepublishesRuntimeSnapshot:
    """Invariant 2: the snapshot tracks the live route after a mid-turn fallback."""

    def test_snapshot_reflects_activated_fallback(self, _isolate):
        aux = _isolate
        agent = _make_agent([dict(FALLBACK_ENTRY)])
        _codex_snapshot(aux)

        _activate_fallback(agent)

        snapshot = aux._normalize_main_runtime(None)
        assert snapshot.get("provider") == "custom"
        assert snapshot.get("model") == "qwen3.8-max"
        assert FALLBACK_HOST in str(snapshot.get("base_url") or "")
        assert PRIMARY_HOST not in str(snapshot.get("base_url") or "")

    def test_custom_lane_binds_fallback_endpoint_after_fallback(self, _isolate):
        """End-to-end: after the fallback activates, the goal_judge-shaped lane
        resolves through the republished contextvar to the fallback's own endpoint
        with the fallback's credential — same provider on both sides."""
        aux = _isolate
        agent = _make_agent([dict(FALLBACK_ENTRY)])
        _codex_snapshot(aux)

        _activate_fallback(agent)

        client, model = aux._get_cached_client(
            "custom", "qwen3.8-max", async_mode=False, task="goal_judge",
        )
        assert client is not None, (
            "after a custom fallback the custom lane must resolve via the "
            "republished runtime snapshot"
        )
        assert model == "qwen3.8-max"
        base = str(getattr(client, "base_url", ""))
        assert FALLBACK_HOST in base
        assert PRIMARY_HOST not in base
