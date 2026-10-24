"""Provider quota durations keep their source and never borrow positional meaning."""
from agent import account_usage as usage


def test_codex_fetch_keeps_duration_provenance_and_unknown_labels(monkeypatch):
    monkeypatch.setattr(usage, "_resolve_codex_usage_credentials", lambda *a, **k: ("token", "https://example.test", "account"))
    monkeypatch.setattr(usage, "_get_json", lambda *a, **k: {"rate_limit": {
        "primary_window": {"used_percent": 25, "limit_window_seconds": 604800},
        "secondary_window": {"used_percent": 30, "limit_window_seconds": 7200},
    }})
    snapshot = usage._fetch_codex_account_usage()
    assert [(w.source_key, w.duration_seconds, w.duration_source) for w in snapshot.windows] == [
        ("primary_window", 604800, "limit_window_seconds"),
        ("secondary_window", 7200, "limit_window_seconds"),
    ]
    assert snapshot.windows[0].label == "Weekly"
    assert snapshot.windows[1].label == "Secondary window"
    for value in (None, True, -1, float("inf"), float("nan")):
        labels = usage._codex_window_labels({"primary_window": {"limit_window_seconds": value}})
        windows = usage._usage_windows({"primary_window": {"used_percent": 10, "limit_window_seconds": value}},
                                      labels, "used_percent", "reset_at", duration_key="limit_window_seconds")
        assert windows[0].label == "Primary window"
        assert windows[0].duration_seconds is None
        assert windows[0].duration_source is None


def test_generic_windows_preserve_model_scope_without_guessed_durations():
    windows = usage._usage_windows(
        {"account": {"used": 0.2, "period": 600}, "model": {"used": 1}},
        (("model", "Model"), ("account", "Account")), "used", "reset",
        fraction=True, model_scoped={"model"}, duration_key="period",
    )
    assert [(w.scope, w.source_key, w.duration_seconds) for w in windows] == [
        ("model", "model", None), ("account", "account", 600),
    ]
    assert windows[0].used_percent == 100
    assert windows[1].duration_source == "period"
