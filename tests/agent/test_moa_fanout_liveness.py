"""MoA fan-out must stamp the turn-liveness activity clock (#110015).

The reference fan-out runs in worker threads outside the main agent loop, and nothing
in it used to stamp ``_last_activity_ts`` — so during a healthy multi-minute fan-out
the turn-liveness watchdog read the clock frozen at the last main-loop stamp
("starting API call #1") and force-aborted the turn at ``agent.turn_liveness.timeout_s``
(default 600s), while the aux stream layer deliberately permits
``max(600, 4 × timeout)`` = 3600s for the same advisors.

The fix bridges both progress signals into ``AIAgent._touch_activity`` (the single
activity write path the watchdog samples):

- each advisor's streaming chunks tick the thread-local aux progress hook
  (``call_llm`` ticks it per substantive chunk on every dispatch path);
- each advisor completion and the aggregator synthesis start stamp milestones.

A silent fan-out (no chunks, no completions) still stamps nothing, so a genuinely
wedged advisor turn remains abortable by the watchdog.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest


def _response(content="ok"):
    message = SimpleNamespace(content=content, tool_calls=[])
    choice = SimpleNamespace(message=message, finish_reason="stop")
    return SimpleNamespace(choices=[choice], usage=None, model="fake")


class _Agent:
    """AIAgent double recording every activity stamp (what the watchdog samples)."""

    def __init__(self):
        self.touches = []
        self._interrupt_requested = False
        self._cache_disabled = None
        self._cache_ttl = None

    def _touch_activity(self, desc, **kwargs):
        self.touches.append(desc)


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


_SLOTS = [
    {"provider": "openrouter", "model": "anthropic/claude-opus-4.8"},
    {"provider": "openrouter", "model": "openai/gpt-5.5"},
]


def _tick_current_aux_progress():
    """Simulate one substantive stream chunk: tick the thread-local aux progress hook."""
    from agent.auxiliary_client import _aux_progress

    hook = getattr(_aux_progress, "hook", None)
    if callable(hook):
        hook()


class TestFanoutStampsActivity:
    def test_streaming_advisor_chunks_stamp_activity(self, moa_config, monkeypatch):
        """Per-chunk ticks from a streaming advisor must reach ``_touch_activity`` —
        the watchdog samples that clock, so chunks re-arm it (#110015)."""
        from agent import moa_loop

        agent = _Agent()
        seen_threads = []
        chunks_per_call = [3, 2]

        def fake_call_llm(**kwargs):
            seen_threads.append(threading.current_thread().name)
            if kwargs.get("task") == "moa_reference":
                for _ in range(chunks_per_call.pop(0)):
                    _tick_current_aux_progress()
            return _response("advice")

        monkeypatch.setattr(moa_loop, "call_llm", fake_call_llm)
        facade = moa_loop.MoAChatCompletions("closed", agent=agent)
        facade.create(messages=[{"role": "user", "content": "hello"}])

        stream_stamps = [t for t in agent.touches if "stream progress" in t]
        assert len(stream_stamps) == 5, agent.touches
        assert any("anthropic/claude-opus-4.8" in t for t in stream_stamps)
        assert any("openai/gpt-5.5" in t for t in stream_stamps)
        # The fan-out really ran on worker threads (the bridge is thread-local).
        assert any(not n.startswith("MainThread") for n in seen_threads)

    def test_each_reference_completion_stamps_activity(self, moa_config, monkeypatch):
        """Non-streaming advisors tick nothing per chunk, so each completion must
        itself be fan-out progress the watchdog can see."""
        from agent import moa_loop

        agent = _Agent()
        monkeypatch.setattr(moa_loop, "call_llm", lambda **kwargs: _response("advice"))
        facade = moa_loop.MoAChatCompletions("closed", agent=agent)
        facade.create(messages=[{"role": "user", "content": "hello"}])

        assert "MoA: 1 of 2 references complete" in agent.touches
        assert "MoA: 2 of 2 references complete" in agent.touches

    def test_silent_fanout_stamps_nothing_until_completion(
        self, moa_config, monkeypatch
    ):
        """No chunks and no completions = no activity stamps: a wedged advisor turn
        must stay abortable by the watchdog (the fix adds progress, never fakes it)."""
        from agent import moa_loop

        agent = _Agent()

        def silent_call_llm(**kwargs):
            assert not any("stream progress" in t for t in agent.touches), (
                "a silent call in flight must not have stamped stream progress"
            )
            return _response("advice")

        monkeypatch.setattr(moa_loop, "call_llm", silent_call_llm)
        facade = moa_loop.MoAChatCompletions("closed", agent=agent)
        facade.create(messages=[{"role": "user", "content": "hello"}])

        assert not any("stream progress" in t for t in agent.touches)

    def test_aggregate_moa_context_stamps_synthesis(self, moa_config, monkeypatch):
        """The one-shot /moa path (no facade) must stamp the aggregator synthesis
        too: it is the longest single call of that path."""
        from agent import moa_loop

        agent = _Agent()
        monkeypatch.setattr(moa_loop, "call_llm", lambda **kwargs: _response("advice"))
        moa_loop.aggregate_moa_context(
            user_prompt="hello",
            api_messages=[{"role": "user", "content": "hello"}],
            reference_models=_SLOTS,
            aggregator={"provider": "openrouter", "model": "anthropic/claude-opus-4.8"},
            agent=agent,
        )

        assert any("synthesizing aggregator guidance" in t for t in agent.touches)

    def test_fanout_without_agent_still_works(self, moa_config, monkeypatch):
        """agent=None (aux/preview callers, test doubles) keeps the old behaviour."""
        from agent import moa_loop

        monkeypatch.setattr(moa_loop, "call_llm", lambda **kwargs: _response("advice"))
        outputs = moa_loop._run_references_parallel(
            _SLOTS, [{"role": "user", "content": "hello"}]
        )
        assert len(outputs) == 2
        assert all(
            not isinstance(text, str) or isinstance(text, str)
            for _lbl, text, _acct in outputs
        )
