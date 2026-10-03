"""`curator.min_idle_hours` is the "let the agent settle" guard, and it must survive a bad value.

`maybe_run_curator` gates on ``idle_for_seconds < get_min_idle_hours() * 3600.0``. A negative
configured value makes that comparison false at every idle measurement, so the guard stops
existing and the review pass can fork while the agent is mid-turn — the opposite of failing
closed. ``0`` is a coherent "no idle requirement" and stays honoured.

Kept in its own file rather than appended to ``test_curator.py`` so it does not collide with the
lifecycle-ordering work in #122746, which edits that file.
"""

import pytest

from agent import curator


@pytest.fixture(autouse=True)
def _clean_curator(monkeypatch):
    monkeypatch.setattr(curator, "_load_config", lambda: {})
    curator._warned_bad_values.clear()
    yield
    curator._warned_bad_values.clear()


def _gate_would_run(monkeypatch, cfg, idle_for_seconds):
    """True when maybe_run_curator gets past the idle gate and starts a pass.

    ``should_run_now`` is forced True so that a ``None`` return can only mean the idle gate
    blocked — otherwise the interval gate would mask the thing under test.
    """
    started = []
    monkeypatch.setattr(curator, "_load_config", lambda: cfg)
    monkeypatch.setattr(curator, "should_run_now", lambda: True)
    monkeypatch.setattr(curator, "_claim_run", lambda: True)
    monkeypatch.setattr(curator, "_release_run_claim", lambda: None)
    monkeypatch.setattr(curator, "run_curator_review",
                        lambda on_summary=None: started.append(True) or {"ok": True})
    result = curator.maybe_run_curator(idle_for_seconds=idle_for_seconds)
    assert bool(started) == (result is not None), "spy and return value disagree"
    return bool(started)


def test_a_negative_min_idle_hours_does_not_disable_the_idle_gate(monkeypatch):
    """With ``min_idle_hours: -1`` the product is negative, so every idle measurement compared
    false and a busy agent's curator pass ran anyway. It now falls back to the default."""
    # An agent that has been idle for one second is, by any reading, not settled.
    assert _gate_would_run(monkeypatch, {"min_idle_hours": -1}, 1.0) is False
    # And the same agent under the default is blocked, so the fallback is what blocked it.
    assert _gate_would_run(monkeypatch, {}, 1.0) is False
    # Past the default window it runs, proving the gate is not simply stuck closed.
    assert _gate_would_run(monkeypatch, {"min_idle_hours": -1}, 3 * 3600.0) is True


def test_zero_min_idle_hours_still_means_no_idle_requirement(monkeypatch):
    """Control: 0 is a coherent setting and is honoured, so the floor is at 0 and not at 1."""
    assert _gate_would_run(monkeypatch, {"min_idle_hours": 0}, 0.0) is True

    monkeypatch.setattr(curator, "_load_config", lambda: {"min_idle_hours": 0})
    assert curator.get_min_idle_hours() == 0.0


def test_a_nan_min_idle_hours_falls_back_and_warns_once(monkeypatch, caplog):
    """YAML can express ``.nan``, and NaN makes the gate's comparison false exactly like a
    negative does. The warn-once marker is keyed on repr() because NaN never equals itself —
    keyed on the float it would miss the set on every poll and warn per dashboard read."""
    monkeypatch.setattr(curator, "_load_config", lambda: {"min_idle_hours": float("nan")})

    with caplog.at_level("WARNING", logger=curator.logger.name):
        for _ in range(4):
            assert curator.get_min_idle_hours() == float(curator.DEFAULT_MIN_IDLE_HOURS)
        warnings = [r for r in caplog.records if "min_idle_hours" in r.getMessage()]

    assert len(warnings) == 1, [r.getMessage() for r in warnings]


def test_a_usable_min_idle_hours_is_returned_verbatim_and_silent(monkeypatch, caplog):
    """Control: a fractional hour is a legitimate setting, returned as-is with nothing logged."""
    monkeypatch.setattr(curator, "_load_config", lambda: {"min_idle_hours": 0.5})

    with caplog.at_level("WARNING", logger=curator.logger.name):
        assert curator.get_min_idle_hours() == 0.5

    assert [r.getMessage() for r in caplog.records] == []
