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
