"""Regression test for aggregate_moa_context's reference/aggregator max_tokens split.

PR #53580 removed a hardcoded ``max_tokens`` cap from the aggregator's
synthesis call because it truncated long aggregator syntheses. PR #56756
(feat(moa): add reference_max_tokens to cap advisor output and cut turn
latency) later reintroduced a single ``max_tokens`` parameter shared by BOTH
the reference fan-out and the aggregator call in ``aggregate_moa_context`` —
silently regressing the exact bug #53580 fixed: setting
``reference_max_tokens`` to speed up the advisors also truncates the
aggregator's own synthesis, which is the context the main agent actually
uses.

``aggregate_moa_context`` and its reference/aggregator calls both go through
``call_llm`` (task="moa_reference" vs task="moa_aggregator"), so mocking
just that one function exercises the real fan-out/aggregation code path.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest


def _response(content: str = "ok"):
    message = SimpleNamespace(content=content, tool_calls=[])
    choice = SimpleNamespace(message=message, finish_reason="stop")
    return SimpleNamespace(choices=[choice], usage=None, model="fake")


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def test_aggregator_call_never_receives_reference_max_tokens(hermes_home, monkeypatch):
    """reference_max_tokens must cap only the reference fan-out — the
    aggregator's own call_llm invocation must not receive max_tokens at all
    (call_llm omits it entirely when None; see its own docstring)."""
    from agent.moa_loop import aggregate_moa_context

    calls: list[dict] = []

    def fake_call_llm(**kwargs):
        calls.append(kwargs)
        return _response("advice" if kwargs.get("task") == "moa_reference" else "synthesis")

    monkeypatch.setattr("agent.moa_loop.call_llm", fake_call_llm)

    aggregate_moa_context(
        user_prompt="clean the db",
        api_messages=[{"role": "user", "content": "clean the db"}],
        reference_models=[{"provider": "openrouter", "model": "openai/gpt-5.5"}],
        aggregator={"provider": "openrouter", "model": "anthropic/claude-opus-4.8"},
        reference_max_tokens=600,
    )

    reference_calls = [c for c in calls if c.get("task") == "moa_reference"]
    aggregator_calls = [c for c in calls if c.get("task") == "moa_aggregator"]
    assert len(reference_calls) == 1
    assert len(aggregator_calls) == 1

    # The reference fan-out is capped as configured.
    assert reference_calls[0]["max_tokens"] == 600
    # The aggregator's synthesis call must be uncapped — not even max_tokens=None,
    # the kwarg must be absent entirely (matches call_llm's omit-when-None contract).
    assert "max_tokens" not in aggregator_calls[0]


def test_aggregator_call_uncapped_when_reference_max_tokens_unset(hermes_home, monkeypatch):
    """Sanity check: with no reference_max_tokens configured, the reference
    call still explicitly passes max_tokens=None (call_llm itself decides
    whether to omit it on the wire), while the aggregator call structurally
    never carries a max_tokens kwarg at all — the pre-#56756 default MoA
    behavior for the aggregator, preserved regardless of the reference cap."""
    from agent.moa_loop import aggregate_moa_context

    calls: list[dict] = []

    def fake_call_llm(**kwargs):
        calls.append(kwargs)
        return _response("advice" if kwargs.get("task") == "moa_reference" else "synthesis")

    monkeypatch.setattr("agent.moa_loop.call_llm", fake_call_llm)

    aggregate_moa_context(
        user_prompt="clean the db",
        api_messages=[{"role": "user", "content": "clean the db"}],
        reference_models=[{"provider": "openrouter", "model": "openai/gpt-5.5"}],
        aggregator={"provider": "openrouter", "model": "anthropic/claude-opus-4.8"},
    )

    reference_calls = [c for c in calls if c.get("task") == "moa_reference"]
    aggregator_calls = [c for c in calls if c.get("task") == "moa_aggregator"]
    assert reference_calls[0]["max_tokens"] is None
    assert "max_tokens" not in aggregator_calls[0]
