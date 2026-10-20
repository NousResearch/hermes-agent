"""A recursion-guard-skipped MoA reference slot is a terminal result (#71243).

``_run_references_parallel`` skips every slot whose ``provider`` is ``moa`` (MoA
presets must not recursively reference MoA) and substitutes a placeholder output.
That slot *always* yields a result, so it must advance the fan-out progress exactly
like a completed reference.

It used to not do so: the skip branch only wrote ``results[idx]`` and never
incremented ``completed`` nor fired ``progress_callback``. In a *mixed* fan-out
(some real slots, some skipped) the terminal ``refs_done`` therefore never reached
``M/M`` — the aggregator still ran with every slot, but any consumer driving a
progress display saw a fan-out that stayed permanently incomplete.

These tests pin the contract: every slot of a fan-out — ran, failed, or skipped —
is a terminal result the progress counter must account for, and the counter must
end at ``M/M`` for an all-skipped fan-out too (where no call is ever dispatched).
"""

from __future__ import annotations

from types import SimpleNamespace

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


_SKIP_NOTE = "[skipped: MoA presets cannot recursively reference MoA]"


def _slots(*specs):
    """``("openrouter", "a")`` -> slot dict; ``("moa", "nested")`` -> recursion guard."""
    return [{"provider": p, "model": m} for p, m in specs]


def _collect_events(monkeypatch):
    """Patch ``call_llm`` with a stub and capture (done, total, label) triples."""
    from agent import moa_loop

    monkeypatch.setattr(moa_loop, "call_llm", lambda **kwargs: _response("advice"))
    events: list[tuple[int, int, str]] = []

    def progress_callback(refs_done, refs_total, label):
        events.append((refs_done, refs_total, label))

    return moa_loop, events, progress_callback


class TestSkippedSlotAdvancesFanoutProgress:
    def test_mixed_fanout_progress_reaches_total(self, moa_config, monkeypatch):
        """The regression: 3 slots where the middle one is skipped must still end at 3/3."""
        moa_loop, events, cb = _collect_events(monkeypatch)

        outputs = moa_loop._run_references_parallel(
            _slots(("openrouter", "a"), ("moa", "nested"), ("openrouter", "b")),
            [{"role": "user", "content": "hello"}],
            progress_callback=cb,
        )

        assert len(outputs) == 3
        assert len(events) == 3, f"every slot is terminal, got {events}"
        assert [done for done, _t, _l in events] == [1, 2, 3]
        assert {total for _d, total, _l in events} == {3}
        assert events[-1][0] == events[-1][1], "refs_done must reach M/M"

    def test_skipped_slot_reports_its_own_label(self, moa_config, monkeypatch):
        """The skipped slot must be reported by label, like any other completion."""
        moa_loop, events, cb = _collect_events(monkeypatch)

        moa_loop._run_references_parallel(
            _slots(("openrouter", "a"), ("moa", "nested")),
            [{"role": "user", "content": "hello"}],
            progress_callback=cb,
        )

        assert "moa:nested" in [label for _d, _t, label in events]

    def test_all_skipped_fanout_reaches_total_without_dispatching(self, moa_config, monkeypatch):
        """No slot dispatches a call, yet progress must still complete at M/M."""
        from agent import moa_loop

        dispatched = []
        monkeypatch.setattr(
            moa_loop, "call_llm", lambda **kwargs: dispatched.append(kwargs) or _response("x")
        )
        events: list[tuple[int, int, str]] = []

        outputs = moa_loop._run_references_parallel(
            _slots(("moa", "x"), ("moa", "y")),
            [{"role": "user", "content": "hello"}],
            progress_callback=lambda d, t, l: events.append((d, t, l)),
        )

        assert dispatched == [], "recursion-guard slots must never be dispatched"
        assert len(outputs) == 2
        assert [(d, t) for d, t, _l in events] == [(1, 2), (2, 2)]

    def test_skipped_slot_still_yields_the_placeholder(self, moa_config, monkeypatch):
        """Progress accounting must not change the substituted output."""
        moa_loop, _events, cb = _collect_events(monkeypatch)

        outputs = moa_loop._run_references_parallel(
            _slots(("moa", "nested"),), [{"role": "user", "content": "hello"}],
            progress_callback=cb,
        )

        label, text, _acct = outputs[0]
        assert label == "moa:nested"
        assert text == _SKIP_NOTE

    def test_normal_fanout_is_unaffected(self, moa_config, monkeypatch):
        """No skipped slot: behaviour is exactly what it was before the fix."""
        moa_loop, events, cb = _collect_events(monkeypatch)

        moa_loop._run_references_parallel(
            _slots(("openrouter", "a"), ("openrouter", "b")),
            [{"role": "user", "content": "hello"}],
            progress_callback=cb,
        )

        assert [done for done, _t, _l in events] == [1, 2]

    def test_progress_callback_failure_never_breaks_the_fanout(self, moa_config, monkeypatch):
        """A raising display callback is swallowed on the skip path too."""
        moa_loop, _events, _cb = _collect_events(monkeypatch)

        def boom(*_a, **_k):
            raise RuntimeError("display exploded")

        outputs = moa_loop._run_references_parallel(
            _slots(("moa", "nested"), ("openrouter", "a")),
            [{"role": "user", "content": "hello"}],
            progress_callback=boom,
        )

        assert len(outputs) == 2
