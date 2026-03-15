from collections import deque

from radio import level_meter


def test_feature_snapshot_returns_safe_defaults_when_meter_inactive(monkeypatch):
    monkeypatch.setattr(level_meter, '_levels', deque([], maxlen=64))
    monkeypatch.setattr(level_meter, '_running', False)
    monkeypatch.setattr(level_meter, '_process', None)

    snap = level_meter.get_feature_snapshot(16)

    assert snap.levels == [0.0] * 16
    assert snap.energy == 0.0
    assert snap.peak == 0.0
    assert snap.transient == 0.0
    assert snap.motion == 0.0
    assert snap.decay == 0.0
    assert snap.active is False


def test_feature_snapshot_resamples_recent_levels_and_computes_motion(monkeypatch):
    monkeypatch.setattr(level_meter, '_levels', deque([-50.0, -40.0, -20.0, -10.0], maxlen=64))
    monkeypatch.setattr(level_meter, '_running', True)

    class _Proc:
        def poll(self):
            return None

    monkeypatch.setattr(level_meter, '_process', _Proc())

    snap = level_meter.get_feature_snapshot(8)

    assert len(snap.levels) == 8
    assert all(0.0 <= value <= 1.0 for value in snap.levels)
    assert 0.0 <= snap.energy <= 1.0
    assert 0.0 <= snap.peak <= 1.0
    assert snap.motion > 0.0
    assert snap.transient > 0.0
    assert snap.decay >= 0.0
    assert snap.active is True
