"""Regression for t_317282ed: the kanban auto-decompose tick must back off a board after a
rate-limited aux call instead of re-probing the same quota wall every ~60s tick (observed:
~2,700 429s in one day against a single triage task while the weekly Anthropic limit held).

``decompose_task`` returning ``ok=False, reason="rate_limited"`` must:
  1. arm a per-board backoff window (``_decompose_rate_limit_until``), sized off the SAME
     ``HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS`` knob the dispatcher's own worker-respawn
     cooldown uses (no second knob to discover/keep in sync);
  2. stop attempting further triage tasks on that board for the rest of THIS tick (don't burn
     the per-tick budget retrying siblings behind the same wall);
  3. skip the board entirely on the NEXT tick while still within the backoff window, without
     calling ``decompose_task`` again.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

from gateway import kanban_watchers_dispatcher as kwd


def _dispatcher():
    settings = kwd._DispatcherSettings(60.0, None, None, 2, 0, True, None, None)
    return kwd._KanbanDispatcher(SimpleNamespace(DEFAULT_BOARD="default"), settings)


def test_rate_limited_decompose_arms_backoff_and_skips_siblings(monkeypatch):
    kwd._decompose_rate_limit_until.clear()
    monkeypatch.setattr(kwd, "_board_slugs", lambda kb: ["default"])
    monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", "300")

    calls = []

    def fake_decompose(task_id, author=None):
        calls.append(task_id)
        return SimpleNamespace(ok=False, fanout=False, child_ids=None, reason="rate_limited")

    fake = SimpleNamespace(
        list_triage_ids=lambda: ["t1", "t2", "t3"],
        decompose_task=fake_decompose,
    )
    import sys
    import hermes_cli
    monkeypatch.setitem(sys.modules, "hermes_cli.kanban_decompose", fake)
    monkeypatch.setattr(hermes_cli, "kanban_decompose", fake, raising=False)

    dispatcher = _dispatcher()
    decomposed = dispatcher.auto_decompose_tick(auto_decompose_per_tick=5)

    assert decomposed == 0
    # Only the FIRST triage task on the board is attempted this tick — t2/t3 are
    # skipped once the board is marked rate-limited, not hammered too.
    assert calls == ["t1"]
    assert "default" in kwd._decompose_rate_limit_until
    assert kwd._decompose_rate_limit_until["default"] > time.time()

    # Next tick, still within the backoff window: the board must be skipped
    # entirely — decompose_task must not be called again.
    calls.clear()
    decomposed2 = dispatcher.auto_decompose_tick(auto_decompose_per_tick=5)
    assert decomposed2 == 0
    assert calls == []

    kwd._decompose_rate_limit_until.clear()


def test_rate_limit_backoff_clears_and_resumes_after_window(monkeypatch):
    kwd._decompose_rate_limit_until.clear()
    monkeypatch.setattr(kwd, "_board_slugs", lambda kb: ["default"])
    monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", "300")

    calls = []

    def fake_decompose(task_id, author=None):
        calls.append(task_id)
        return SimpleNamespace(ok=True, fanout=False, child_ids=None, reason=None)

    fake = SimpleNamespace(list_triage_ids=lambda: ["t1"], decompose_task=fake_decompose)
    import sys
    import hermes_cli
    monkeypatch.setitem(sys.modules, "hermes_cli.kanban_decompose", fake)
    monkeypatch.setattr(hermes_cli, "kanban_decompose", fake, raising=False)

    # Simulate a backoff window that already expired (set in the past).
    kwd._decompose_rate_limit_until["default"] = time.time() - 1

    dispatcher = _dispatcher()
    decomposed = dispatcher.auto_decompose_tick(auto_decompose_per_tick=5)

    assert decomposed == 1
    assert calls == ["t1"]

    kwd._decompose_rate_limit_until.clear()
