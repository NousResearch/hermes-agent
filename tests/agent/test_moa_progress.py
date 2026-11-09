"""Tests for the MoA progress indicator added in issue #59546.

The MoA facade (``MoAChatCompletions.create``) emits a sequence of display
events so a TUI / CLI / desktop surface can render progress as references
complete and the phase transitions from ``reference`` to ``aggregator``:

  - ``moa.progress`` — fired once per settled reference slot with
                       ``state="completed"`` and monotonic ``refs_done`` /
                       ``refs_total`` counters; separate ``state="waiting"``
                       heartbeats report dispatch and pending work
  - ``moa.phase``    — fired once per phase transition (currently only the
                       ``phase="aggregator"`` transition right before the
                       aggregator acts)

These tests exercise the real callback surface end-to-end through the
display hook (``reference_callback``) — no mocks on the dispatch path.
The LLM is stubbed via ``call_llm`` so the test does not depend on any
real provider.
"""

from __future__ import annotations

import threading
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
        - provider: openrouter
          model: google/gemini-3-pro
      aggregator:
        provider: openrouter
        model: anthropic/claude-opus-4.8
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def _collect_emits(facade):
    """Pull every event the facade dispatches into a flat list of (event, kwargs)."""
    captured: list[tuple[str, dict]] = []

    def _capture(event: str, **kwargs):
        captured.append((event, kwargs))

    facade.reference_callback = _capture
    return captured




def test_moa_progress_emits_dispatch_and_waiting_heartbeats(moa_config, monkeypatch):
    """A slow advisor is visible immediately and remains visibly alive."""
    from agent import moa_loop

    release = threading.Event()
    second_heartbeat = threading.Event()
    monkeypatch.setattr(moa_loop, "_REFERENCE_POLL_INTERVAL_S", 0.02)
    monkeypatch.setattr(moa_loop, "_reference_context_length", lambda *_args: None)

    def slow_call_llm(**kwargs):
        if kwargs.get("task") == "moa_reference":
            release.wait(timeout=2)
            return _response("advice")
        return _response("acted")

    monkeypatch.setattr(moa_loop, "call_llm", slow_call_llm)
    facade = moa_loop.MoAChatCompletions("closed")
    captured = []

    def capture(event, **kwargs):
        captured.append((event, kwargs))
        if event == "moa.progress" and kwargs.get("state") == "waiting":
            waiting = [
                item for item in captured
                if item[0] == "moa.progress" and item[1].get("state") == "waiting"
            ]
            if len(waiting) >= 2:
                second_heartbeat.set()

    facade.reference_callback = capture
    worker = threading.Thread(
        target=lambda: facade.create(
            model="closed",
            messages=[{"role": "user", "content": "pressure-test this"}],
        )
    )
    worker.start()

    try:
        assert second_heartbeat.wait(timeout=1), "a pending fan-out must emit a periodic heartbeat"
        waiting = [
            kwargs
            for event, kwargs in captured
            if event == "moa.progress" and kwargs.get("state") == "waiting"
        ]
        assert waiting[0]["refs_done"] == 0
        assert waiting[0]["refs_total"] == 3
        assert waiting[0]["elapsed_seconds"] == 0
        assert waiting[0]["label"].startswith("waiting on 3 advisors")
        assert not any(event == "moa.reference" for event, _ in captured)
    finally:
        release.set()
        worker.join(timeout=2)

    assert not worker.is_alive()
    completed = [
        kwargs
        for event, kwargs in captured
        if event == "moa.progress" and kwargs.get("state") == "completed"
    ]
    assert [item["refs_done"] for item in completed] == [1, 2, 3]


def test_failed_references_emit_terminal_progress(moa_config, monkeypatch):
    """A failed provider call settles its slot and advances terminal progress."""
    from agent import moa_loop

    monkeypatch.setattr(moa_loop, "_reference_context_length", lambda *_args: None)

    def fail_references(**kwargs):
        if kwargs.get("task") == "moa_reference":
            raise RuntimeError("advisor unavailable")
        return _response("acted")

    monkeypatch.setattr(moa_loop, "call_llm", fail_references)
    facade = moa_loop.MoAChatCompletions("closed")
    captured = _collect_emits(facade)
    facade.create(
        model="closed",
        messages=[{"role": "user", "content": "handle unavailable advisors"}],
    )

    completed = [
        kwargs
        for event, kwargs in captured
        if event == "moa.progress" and kwargs.get("state") == "completed"
    ]
    assert [item["refs_done"] for item in completed] == [1, 2, 3]
    assert all(text.startswith("[failed:") for _, text, _ in facade._ref_cache_outputs)


def test_moa_phase_transitions_to_aggregator(moa_config, monkeypatch):
    """A single ``moa.phase`` event with ``phase="aggregator"`` fires after the fan-out."""
    from agent.moa_loop import MoAChatCompletions

    def fake_call_llm(**kwargs):
        if kwargs.get("task") == "moa_reference":
            return _response("advice")
        return _response("acted")

    monkeypatch.setattr("agent.moa_loop.call_llm", fake_call_llm)

    facade = MoAChatCompletions("closed")
    captured = _collect_emits(facade)

    facade.create(
        model="closed",
        messages=[{"role": "user", "content": "plan the migration"}],
    )

    phase_events = [(e, k) for (e, k) in captured if e == "moa.phase"]
    # Exactly one phase event per turn, identifying the aggregator.
    assert len(phase_events) == 1
    _event, kwargs = phase_events[0]
    assert kwargs["phase"] == "aggregator"
    assert kwargs["aggregator"] == "openrouter:anthropic/claude-opus-4.8"
    # counts match the configured reference count
    assert kwargs["refs_done"] == 3
    assert kwargs["refs_total"] == 3






def test_moa_progress_callback_none_safe(moa_config, monkeypatch):
    """A missing ``reference_callback`` does not break the fan-out or create()."""
    from agent.moa_loop import MoAChatCompletions

    def fake_call_llm(**kwargs):
        if kwargs.get("task") == "moa_reference":
            return _response("advice")
        return _response("acted")

    monkeypatch.setattr("agent.moa_loop.call_llm", fake_call_llm)

    # No callback attached — the facade's _emit is a no-op in that case.
    facade = MoAChatCompletions("closed")
    assert facade.reference_callback is None
    facade.create(
        model="closed",
        messages=[{"role": "user", "content": "noop"}],
    )
    # Turn still resolved cleanly; aggregator slot populated as usual.
    assert facade.last_aggregator_slot is not None
    assert facade.last_aggregator_slot["model"] == "anthropic/claude-opus-4.8"
