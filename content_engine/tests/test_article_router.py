"""Tests for article_router — mode decision + ledger bookkeeping."""
import article_router as ar


def _sig(sid, stype="harness_change", prio=8, sha="abc", summary=None):
    return {
        "signal_id": sid,
        "signal_type": stype,
        "priority": prio,
        "summary": summary or f"signal {sid}",
        "repo": "KenseiAgent",
        "sha": sha,
        "variables": {"summary": summary or f"signal {sid}"},
    }


def test_deep_dive_when_top_signal_meets_threshold(monkeypatch):
    """3 signals, top is priority 9: deep_dive on the single best signal."""
    sigs = [_sig("a", prio=4), _sig("b", prio=9), _sig("c", prio=5)]
    monkeypatch.setattr(ar, "collect_signals", lambda state: sigs)
    plan = ar.choose({"used": []})
    assert plan is not None
    assert plan["mode"] == "deep_dive"
    assert len(plan["signals"]) == 1
    assert plan["signals"][0]["signal_id"] == "b"


def test_digest_when_minor_signals_accumulate(monkeypatch):
    """5 minor signals (all under threshold): digest rolls them up."""
    sigs = [_sig(f"m{i}", prio=4) for i in range(5)]
    monkeypatch.setattr(ar, "collect_signals", lambda state: sigs)
    plan = ar.choose({"used": []})
    assert plan is not None
    assert plan["mode"] == "digest"
    assert len(plan["signals"]) == 5


def test_skip_when_no_signals(monkeypatch):
    monkeypatch.setattr(ar, "collect_signals", lambda state: [])
    assert ar.choose({"used": []}) is None


def test_skip_when_below_thresholds(monkeypatch):
    """A single weak signal (priority 4) is not a deep dive, and 1 is below
    the digest min (4). Skip the day."""
    sigs = [_sig("weak", prio=4)]
    monkeypatch.setattr(ar, "collect_signals", lambda state: sigs)
    assert ar.choose({"used": []}) is None


def test_chosen_signals_marked_used(monkeypatch):
    sigs = [_sig("a", prio=9), _sig("b", prio=8)]
    monkeypatch.setattr(ar, "collect_signals", lambda state: sigs)
    state = {"used": []}
    plan = ar.choose(state)
    assert plan is not None
    # Deep dive on top signal only, so one marked.
    assert state["used"] == ["a"]


def test_digest_min_signals_floor(monkeypatch):
    """3 minor signals is below the digest min of 4: skip."""
    sigs = [_sig(f"m{i}", prio=4) for i in range(3)]
    monkeypatch.setattr(ar, "collect_signals", lambda state: sigs)
    assert ar.choose({"used": []}) is None


def test_no_duplicate_with_short_post_track(monkeypatch):
    """Signals already in the in-memory ledger are filtered out before scoring."""
    sigs = [_sig("a", prio=9), _sig("b", prio=8)]
    monkeypatch.setattr(ar, "collect_signals", lambda state: sigs)
    state = {"used": ["a"]}
    plan = ar.choose(state)
    assert plan is not None
    # "a" is filtered; "b" wins at priority 8 -> deep dive.
    assert plan["signals"][0]["signal_id"] == "b"
    assert state["used"] == ["a", "b"]


def test_choose_returns_none_when_disabled(monkeypatch):
    """ARTICLE_ENABLED=False short-circuits before any signal fetch."""
    monkeypatch.setattr(ar, "ARTICLE_ENABLED", False)
    assert ar.choose({"used": []}) is None
