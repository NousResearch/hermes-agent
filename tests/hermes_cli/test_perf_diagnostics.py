from __future__ import annotations

from hermes_cli import perf_diagnostics


def setup_function() -> None:
    perf_diagnostics._reset_for_tests()


def test_snapshot_contains_only_bounded_activity_metadata(monkeypatch) -> None:
    monkeypatch.setenv("HERMES_PROFILE", "coder")
    token = perf_diagnostics.begin_activity("rpc", "model.options", pool=True)

    snapshot = perf_diagnostics.diagnostics_snapshot()

    assert snapshot["pid"] > 0
    assert snapshot["profile"] == "coder"
    assert snapshot["active"][0]["category"] == "rpc"
    assert snapshot["active"][0]["name"] == "model.options"
    assert snapshot["active"][0]["attributes"] == {"pool": True}
    perf_diagnostics.finish_activity(token)


def test_slow_activity_is_retained_after_completion(monkeypatch) -> None:
    clock = iter((10.0, 12.5, 12.5))
    monkeypatch.setattr(perf_diagnostics.time, "monotonic", lambda: next(clock))
    token = perf_diagnostics.begin_activity("backup", "snapshot")

    perf_diagnostics.finish_activity(token, error=True)
    snapshot = perf_diagnostics.diagnostics_snapshot()

    assert snapshot["active"] == []
    assert snapshot["recent_slow"][0]["duration_ms"] == 2500.0
    assert snapshot["recent_slow"][0]["error"] is True
