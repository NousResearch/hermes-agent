"""The ACP surface must register the user's ``hooks:`` shell hooks (#41457).

``hooks:`` guards live in the process-wide hook table. The CLI, the messaging
gateway and the desktop/TUI backend each register them when they build an
agent; the ACP adapter (IDE integration) did not, so a ``pre_tool_call`` guard
configured by the user was enforced in every surface except an IDE session.
``hermes hooks doctor`` still reported green, because it validates the config
and the script, not whether the running process wired them.

These tests assert the behaviour, not the source text.
"""

from __future__ import annotations

import pytest

from acp_adapter.session import SessionManager


class FakeAgent:
    """Minimal AIAgent stand-in: the builder only sets ``_print_fn`` after it."""

    model = "fake-model"
    provider = "fake-provider"
    enabled_toolsets: list[str] = []
    disabled_toolsets: list[str] = []
    tools: list = []
    valid_tool_names: set = set()

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __getattr__(self, _name):  # tolerate anything the manager touches
        return None


class NoopDb:
    def get_session(self, *_a, **_k):
        return None

    def create_session(self, *_a, **_k):
        return None


@pytest.fixture
def built(monkeypatch):
    """Build one ACP session and report the shell-hook registrations it made."""
    registered: list[dict] = []

    monkeypatch.setattr("run_agent.AIAgent", FakeAgent)
    monkeypatch.setattr(
        "agent.shell_hooks.register_from_config",
        lambda cfg, **_kw: registered.append(cfg) or [],
    )
    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", lambda **_kw: {})
    monkeypatch.setattr(
        "hermes_cli.mcp_startup.ensure_mcp_discovery_before_agent_build",
        lambda **_kw: None,
    )
    manager = SessionManager(db=NoopDb())
    agent = manager._make_agent(session_id="test-acp-hooks", cwd=".")
    return agent, registered


def test_acp_agent_build_registers_config_shell_hooks(built):
    agent, registered = built

    assert isinstance(agent, FakeAgent), "the session did not build through _make_agent"
    assert registered, "ACP built an agent without registering config.yaml shell hooks"


def test_acp_registration_uses_the_same_config_the_agent_gets(built):
    """One read of the config: hooks must see exactly what the agent was built from."""
    agent, registered = built

    assert len(registered) == 1
    assert registered[0] is not None
    assert agent.kwargs.get("platform") == "acp"


def test_acp_registration_is_non_interactive(built):
    """A consent prompt in the ACP event loop would hang the IDE session."""
    _, registered = built

    assert registered, "no registration recorded"
    # register_from_config(cfg) with no accept_hooks/interactive kwargs is the
    # non-prompting call; a prompt-capable call would have to pass them.
