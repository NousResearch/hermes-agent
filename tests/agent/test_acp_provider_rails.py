"""Two core decisions must key on the ``acp://`` scheme, not on one vendor.

An ACP client talks to a CLI over subprocess stdio: it returns a plain
completion object rather than an iterable stream, and it does not implement the
Responses API surface. Both exclusions used to spell out ``acp://copilot``,
which meant the next ACP client silently inherited the wrong defaults — a
Responses upgrade its shim cannot serve, and a streaming call that tries to
iterate a ``SimpleNamespace``.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


class _FakeCompletions:
    """Returns a whole completion — exactly what an ACP shim does.

    ``stream=True`` is not honoured (an ACP turn is one-shot), so if the loop
    ever tries to stream this, iterating the result raises.
    """

    def __init__(self):
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="ok", reasoning=None, tool_calls=[]),
                    finish_reason="stop",
                )
            ],
            usage=None,
        )


class _FakeClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=_FakeCompletions())


def _agent(monkeypatch, base_url: str, **kwargs):
    from run_agent import AIAgent

    client = _FakeClient()
    monkeypatch.setattr("agent.process_bootstrap.OpenAI", lambda **_kw: client)
    monkeypatch.setattr("model_tools.get_tool_definitions", lambda *a, **k: [])
    agent = AIAgent(
        model="gpt-5",  # a model that would normally trigger the Responses upgrade
        api_key="test-key",
        base_url=base_url,
        platform="cli",
        max_iterations=2,
        quiet_mode=True,
        skip_memory=True,
        **kwargs,
    )
    return agent, client


def test_an_acp_base_url_is_not_upgraded_to_the_responses_api(monkeypatch):
    agent, _ = _agent(monkeypatch, "acp://somevendor")
    assert agent.api_mode == "chat_completions"


def test_a_non_acp_url_still_upgrades(monkeypatch):
    """Guard against the exclusion being widened into a blanket opt-out."""
    agent, _ = _agent(monkeypatch, "https://api.openai.com/v1")
    assert agent.api_mode == "codex_responses"


def test_an_acp_provider_turn_never_asks_for_a_stream(monkeypatch):
    """A display consumer is present, so streaming would otherwise be chosen."""
    agent, client = _agent(
        monkeypatch, "acp://somevendor", stream_delta_callback=lambda *_a, **_k: None
    )
    assert agent._has_stream_consumers()

    result = agent.run_conversation("hi")

    assert result["final_response"].startswith("ok")
    assert client.chat.completions.calls, "the client was never called"
    assert not any(c.get("stream") for c in client.chat.completions.calls)


def _register_acp_profile(
    monkeypatch,
    name: str,
    *,
    auth_type: str = "external_process",
    process_command: str = "",
    process_args: tuple = (),
    process_command_env_vars: tuple = (),
) -> None:
    """Register a profile through the real registry, auto-removed by monkeypatch."""
    from providers import _REGISTRY
    from providers.base import ProviderProfile

    monkeypatch.setitem(
        _REGISTRY,
        name,
        ProviderProfile(
            name=name,
            auth_type=auth_type,
            base_url=f"acp://{name}",
            process_command=process_command,
            process_args=process_args,
            process_command_env_vars=process_command_env_vars,
        ),
    )


def _launch_kwargs(provider: str):
    from types import SimpleNamespace

    from agent.agent_init import _explicit_client_kwargs

    agent = SimpleNamespace(provider=provider, acp_command=None, acp_args=None)
    return _explicit_client_kwargs(agent, "placeholder", f"acp://{provider}", None)


def test_an_external_process_profile_supplies_the_launch_command(monkeypatch):
    """Any ACP provider is driven from its own profile — core must not match on one name."""
    _register_acp_profile(
        monkeypatch, "fake-acp", process_command="/usr/bin/fake-acp", process_args=("acp", "--stdio")
    )

    kwargs = _launch_kwargs("fake-acp")

    assert kwargs["command"] == "/usr/bin/fake-acp"
    assert kwargs["args"] == ["acp", "--stdio"]


def test_the_profile_env_override_wins_over_its_static_command(monkeypatch):
    monkeypatch.setenv("HERMES_FAKE_ACP_COMMAND", "/opt/elsewhere/fake-acp")
    _register_acp_profile(
        monkeypatch,
        "fake-acp-env",
        process_command="/usr/bin/fake-acp",
        process_args=("acp",),
        process_command_env_vars=("HERMES_FAKE_ACP_COMMAND",),
    )

    assert _launch_kwargs("fake-acp-env")["command"] == "/opt/elsewhere/fake-acp"


def test_an_http_provider_gets_no_launch_command(monkeypatch):
    """Guard against the ACP branch widening onto ordinary API-key providers."""
    _register_acp_profile(
        monkeypatch, "fake-http", auth_type="api_key", process_command="/usr/bin/nope"
    )

    kwargs = _launch_kwargs("fake-http")

    assert "command" not in kwargs
    assert "args" not in kwargs


def test_an_external_process_provider_never_streams_or_upgrades(monkeypatch):
    """Streaming and the Responses upgrade follow the profile, not the base_url scheme."""
    from types import SimpleNamespace

    from agent.turn_api_call import _should_stream

    _register_acp_profile(
        monkeypatch, "fake-acp-plain", process_command="/usr/bin/fake-acp", process_args=("acp",)
    )

    assert not _should_stream(
        SimpleNamespace(provider="fake-acp-plain", base_url="https://example.invalid/v1")
    )

    agent, _ = _agent(monkeypatch, "https://example.invalid/v1", provider="fake-acp-plain")
    assert agent.api_mode == "chat_completions"
