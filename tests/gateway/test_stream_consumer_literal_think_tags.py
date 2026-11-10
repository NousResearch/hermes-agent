"""Gateway previews follow the agent's ``literal_think_tags`` decision.

The consumer's think filter is a second pass over deltas the agent already scrubbed. When the
agent kept a response's think tags as answer text (a route opted in to ``literal_think_tags``),
the preview filter must not hide them; everywhere else it keeps filtering as before.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest

# Stub optional heavy imports so run_agent imports cleanly in isolation.
sys.modules.setdefault("fire", types.SimpleNamespace(Fire=lambda *a, **k: None))
sys.modules.setdefault("firecrawl", types.SimpleNamespace(Firecrawl=object))
sys.modules.setdefault("fal_client", types.SimpleNamespace())

from gateway.stream_consumer import GatewayStreamConsumer

_ROUTE = "https://llm.example.invalid/v1"
_MODEL = "deepseek-v4.1-flash"
_DELTAS = ["<thi", "nk>\nan example block\n</think>\n", "Close it with </th", "ink>."]


def _preview(consumer: GatewayStreamConsumer) -> str:
    for delta in _DELTAS:
        consumer._filter_and_accumulate(delta)
    consumer._flush_think_buffer()
    return consumer._accumulated


def test_preview_keeps_tags_the_agent_kept():
    consumer = GatewayStreamConsumer(MagicMock(), "chat_test")
    consumer.keeps_think_tags = lambda: True
    assert _preview(consumer) == "".join(_DELTAS)


@pytest.mark.parametrize("keeps", [None, MagicMock()])
def test_preview_filter_is_unchanged_by_default_and_for_mock_agents(keeps):
    consumer = GatewayStreamConsumer(MagicMock(), "chat_test")
    if keeps is not None:
        consumer.keeps_think_tags = keeps  # a MagicMock agent's method returns a truthy MagicMock
    preview = _preview(consumer)
    assert "<think>" not in preview and "an example block" not in preview
    assert "Close it with" in preview


def _agent(tmp_path, monkeypatch, *, opted_in: bool):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("", encoding="utf-8")
    capability = "literal_think_tags: true" if opted_in else "context_length: 1000000"
    (tmp_path / "config.yaml").write_text(
        f"providers:\n  local-llm:\n    api: {_ROUTE}\n    models:\n      {_MODEL}:\n        {capability}\n",
        encoding="utf-8")
    from run_agent import AIAgent

    return AIAgent(model=_MODEL, api_key="sk-dummy", base_url=_ROUTE, quiet_mode=True,
                   skip_context_files=True, skip_memory=True, platform="discord")


@pytest.mark.parametrize("opted_in, expected", [
    (True, "".join(_DELTAS)),
    (False, "Close it with ."),
])
def test_agent_to_consumer_preview_follows_the_route(tmp_path, monkeypatch, opted_in, expected):
    """The turn runner binds ``keeps_think_tags`` to the agent's per-response decision; a real agent
    feeding a real consumer shows the tags only on the opted-in route."""
    agent = _agent(tmp_path, monkeypatch, opted_in=opted_in)
    consumer = GatewayStreamConsumer(MagicMock(), "chat_test")
    consumer.keeps_think_tags = agent._stream_keeps_think_tags
    agent.stream_delta_callback = consumer.on_delta
    agent._reset_stream_delivery_tracking()

    for delta in _DELTAS:
        agent._fire_stream_delta(delta)
    agent._reset_stream_delivery_tracking()  # end of response: flushes the agent's held tail
    consumer._drain_queue()
    consumer._flush_think_buffer()

    assert consumer._accumulated == expected
