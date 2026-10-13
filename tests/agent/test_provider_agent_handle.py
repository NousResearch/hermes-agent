"""Which providers get the live agent handed to their ``create_client``.

An agent-as-provider transport (the OMP thin host) executes its own tools, so the
only way that activity can reach Hermes' tool rail — ACP tool calls for Multica,
tool cards in the desktop — is for the transport to hold the agent and call
``agent.tool_progress_callback`` itself.

The handle must therefore reach exactly the profiles that ask for it. It must NOT
travel inside ``client_kwargs``: that mapping is forwarded verbatim to SDK
constructors (``openai.OpenAI`` and friends) which raise on unknown keywords —
widening it for every provider is a regression that has already broken
openai-codex and openrouter once.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import providers  # noqa: E402

from agent.agent_runtime_helpers import _provider_supplied_client  # noqa: E402


class _Profile:
    """Stands in for a registered ProviderProfile."""

    name = "stub"

    def __init__(self, *, wants_agent_handle=False, raises=False):
        if wants_agent_handle:
            self.wants_agent_handle = True
        self._raises = raises
        self.seen_kwargs = None

    def create_client(self, **client_kwargs):
        if self._raises:
            raise RuntimeError("plugin exploded")
        self.seen_kwargs = client_kwargs
        return SimpleNamespace(kind="stub-client", kwargs=client_kwargs)


def _resolve(monkeypatch, profile):
    monkeypatch.setattr(providers, "get_provider_profile", lambda name: profile)
    agent = SimpleNamespace(provider="stub", tool_progress_callback=lambda *a, **k: None)
    client = _provider_supplied_client(agent, {"api_key": "k", "base_url": "rpc-ui://omp"})
    return agent, client


def test_ordinary_profile_never_sees_the_agent(monkeypatch):
    profile = _Profile()
    _agent, client = _resolve(monkeypatch, profile)
    assert client is not None
    assert "_hermes_agent" not in profile.seen_kwargs
    assert set(profile.seen_kwargs) == {"api_key", "base_url"}


def test_opt_in_profile_receives_the_live_agent(monkeypatch):
    profile = _Profile(wants_agent_handle=True)
    agent, client = _resolve(monkeypatch, profile)
    assert client is not None
    assert profile.seen_kwargs["_hermes_agent"] is agent


def test_opt_in_does_not_disturb_the_client_kwargs_it_forwards(monkeypatch):
    profile = _Profile(wants_agent_handle=True)
    _agent, _client = _resolve(monkeypatch, profile)
    assert profile.seen_kwargs["api_key"] == "k"
    assert profile.seen_kwargs["base_url"] == "rpc-ui://omp"


def test_a_raising_profile_is_skipped_rather_than_taking_the_turn_down(monkeypatch):
    # Pre-existing contract: a third-party plugin may only fail to provide a client.
    for wants in (False, True):
        profile = _Profile(wants_agent_handle=wants, raises=True)
        _agent, client = _resolve(monkeypatch, profile)
        assert client is None, wants
