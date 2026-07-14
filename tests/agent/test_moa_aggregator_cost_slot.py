"""Tests for MoA aggregator-slot exposure used by session cost accounting.

Regression guard for the ~50% MoA cost undercount: on the MoA path the
agent's model/provider are the virtual preset name (e.g. "closed") and "moa",
which have no pricing entry. Session cost accounting must price the
aggregator's acting turn at its REAL model/provider, read from the MoA
client's ``last_aggregator_slot``. Before the fix that slot did not exist and
the aggregator's spend (often >50% of the turn) was silently dropped, leaving
the session cost as advisor-fan-out only.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest


def _response(content="ok"):
    message = SimpleNamespace(content=content, tool_calls=[])
    choice = SimpleNamespace(message=message, finish_reason="stop")
    return SimpleNamespace(choices=[choice], usage=None, model="fake")


@pytest.fixture
def moa_config(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        """
moa:
  default_preset: closed
  presets:
    closed:
      enabled: true
      reference_models:
        - provider: openrouter
          model: anthropic/claude-opus-4.8
        - provider: openrouter
          model: openai/gpt-5.5
      aggregator:
        provider: openrouter
        model: anthropic/claude-opus-4.8
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def test_create_populates_last_aggregator_slot(moa_config, monkeypatch):
    """After a create() turn, last_aggregator_slot carries the REAL aggregator
    model/provider — not the virtual preset name."""
    from agent.moa_loop import MoAChatCompletions

    def fake_call_llm(**kwargs):
        return _response("acted" if kwargs.get("task") != "moa_reference" else "advice")

    monkeypatch.setattr("agent.moa_loop.call_llm", fake_call_llm)

    facade = MoAChatCompletions("closed")
    # Slot is unset before any turn runs.
    assert facade.last_aggregator_slot is None

    facade.create(
        model="closed",
        messages=[{"role": "user", "content": "clean the db"}],
    )

    slot = facade.last_aggregator_slot
    assert slot is not None
    # The virtual preset name / "moa" must NOT leak into the priced slot.
    assert slot["model"] == "anthropic/claude-opus-4.8"
    assert slot["provider"] == "openrouter"
    assert slot["model"] != "closed"


def test_client_exposes_last_aggregator_slot(moa_config, monkeypatch):
    """MoAClient delegates last_aggregator_slot to its completions facade so
    session accounting can read it without touching internals."""
    from agent.moa_loop import MoAClient

    def fake_call_llm(**kwargs):
        return _response("acted" if kwargs.get("task") != "moa_reference" else "advice")

    monkeypatch.setattr("agent.moa_loop.call_llm", fake_call_llm)

    client = MoAClient("closed")
    assert client.last_aggregator_slot is None

    client.chat.completions.create(
        model="closed",
        messages=[{"role": "user", "content": "clean the db"}],
    )

    slot = client.last_aggregator_slot
    assert slot is not None
    assert slot["model"] == "anthropic/claude-opus-4.8"
    assert slot["provider"] == "openrouter"


def test_reference_pricing_calls_preserve_each_advisor_route(moa_config, monkeypatch):
    """Blackbox can retrieve physical advisor usage without repricing it as MoA."""
    from agent.moa_loop import MoAClient

    def fake_call_llm(**kwargs):
        response = _response(
            "acted" if kwargs.get("task") != "moa_reference" else "advice"
        )
        response.usage = SimpleNamespace(
            prompt_tokens=1000,
            completion_tokens=100,
            total_tokens=1100,
        )
        return response

    monkeypatch.setattr("agent.moa_loop.call_llm", fake_call_llm)
    client = MoAClient("closed")
    client.chat.completions.create(
        model="closed",
        messages=[{"role": "user", "content": "clean the db"}],
    )

    calls = client.consume_reference_pricing_calls()
    assert [c["model"] for c in calls] == [
        "anthropic/claude-opus-4.8",
        "openai/gpt-5.5",
    ]
    assert [c["provider"] for c in calls] == ["openrouter", "openrouter"]
    assert [c["input_tokens"] for c in calls] == [1000, 1000]
    assert [c["output_tokens"] for c in calls] == [100, 100]

    # Consume-once matches consume_reference_usage: cached tool iterations do
    # not multiply advisor billing rows.
    assert client.consume_reference_pricing_calls() == []
