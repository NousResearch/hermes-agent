"""Restoring an ACP session must heal the persisted bare ``custom`` billing class back to the
configured named identity before resolving the runtime provider (#132937)."""

from types import SimpleNamespace

import pytest

from acp_adapter.session import SessionManager


class FakeAgent:
    model = "fake-model"

    def __init__(self, **kwargs):
        self.kwargs = kwargs


@pytest.fixture()
def harness(monkeypatch):
    """Patch _make_agent's collaborators; expose the resolve/heal call records."""
    config = {
        "model": {"default": "fake-model", "provider": "custom:proxy"},
        "mcp_servers": {},
    }
    monkeypatch.setattr("run_agent.AIAgent", FakeAgent)
    monkeypatch.setattr(
        "acp_adapter.session.load_config", lambda: config, raising=False
    )
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: config)
    monkeypatch.setattr(
        "acp_adapter.session._register_task_cwd", lambda task_id, cwd: None
    )
    monkeypatch.setattr(
        "hermes_cli.mcp_startup.ensure_mcp_discovery_before_agent_build",
        lambda **_kw: None,
    )

    def fake_resolve(**kwargs):
        calls["resolve"].append(kwargs)
        return {
            "provider": kwargs.get("requested"),
            "api_mode": "chat_completions",
            "base_url": "https://proxy.example/v1",
            "api_key": "k",
        }

    def fake_heal(**kwargs):
        calls["heal"].append(kwargs)
        return heal_result["value"]

    calls = {"resolve": [], "heal": []}
    heal_result = {"value": None}
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider", fake_resolve
    )
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.canonical_custom_identity", fake_heal
    )
    return SimpleNamespace(calls=calls, heal_result=heal_result)


def test_bare_custom_restore_heals_named_identity(harness):
    """A persisted bare ``custom`` (the resolved billing class of ``custom:proxy``) is healed back
    to the configured entry before resolution, and the persisted endpoint still reaches the
    resolver as the direct-alias fallback — the same contract as the TUI resume path."""
    harness.heal_result["value"] = "custom:proxy"
    agent = SessionManager(db=None)._make_agent(
        session_id="s",
        cwd=".",
        model="m",
        requested_provider="custom",
        base_url="https://proxy.example/v1",
    )

    assert harness.calls["heal"] == [
        {
            "base_url": "https://proxy.example/v1",
            "config_provider": "custom:proxy",
            "model": "m",
        }
    ]
    assert harness.calls["resolve"][0]["requested"] == "custom:proxy"
    assert (
        harness.calls["resolve"][0]["explicit_base_url"] == "https://proxy.example/v1"
    )
    assert agent.kwargs["provider"] == "custom:proxy"
    assert agent.kwargs["base_url"] == "https://proxy.example/v1"


def test_bare_custom_without_recovery_keeps_bare_label_and_endpoint(harness):
    """Identity recovery failing (entry removed) must not silently drop the persisted endpoint:
    the bare label + explicit base_url still reach the resolver so pool/env credentials apply."""
    harness.heal_result["value"] = None
    SessionManager(db=None)._make_agent(
        session_id="s",
        cwd=".",
        model="m",
        requested_provider="custom",
        base_url="https://proxy.example/v1",
    )

    assert harness.calls["resolve"][0]["requested"] == "custom"
    assert (
        harness.calls["resolve"][0]["explicit_base_url"] == "https://proxy.example/v1"
    )


def test_routable_provider_restore_skips_the_heal(harness, monkeypatch):
    """The heal is scoped to the bare ``custom`` label; a routable persisted provider resolves
    directly and must not pay the identity lookup."""

    def _boom(**kwargs):
        raise AssertionError(
            "canonical_custom_identity must not run for routable providers"
        )

    monkeypatch.setattr("hermes_cli.runtime_provider.canonical_custom_identity", _boom)
    SessionManager(db=None)._make_agent(
        session_id="s", cwd=".", model="m", requested_provider="openrouter"
    )

    assert harness.calls["heal"] == []
    assert harness.calls["resolve"][0]["requested"] == "openrouter"
    assert "explicit_base_url" not in harness.calls["resolve"][0]
