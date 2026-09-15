from types import SimpleNamespace


def test_disabled_guard_does_not_fetch(monkeypatch):
    from agent import codex_quota_guard as guard

    monkeypatch.setattr(guard, "configured_daily_budget", lambda: None)
    assert guard.check_daily_budget() is None


def test_blocks_after_daily_delta(monkeypatch, tmp_path):
    from agent import account_usage, codex_quota_guard as guard

    monkeypatch.setattr(guard, "_state_path", lambda: tmp_path / "state.json")
    monkeypatch.setattr(guard, "configured_daily_budget", lambda: 10.0)
    values = iter((20.0, 31.0))

    def fake_usage(*args, **kwargs):
        return SimpleNamespace(windows=[SimpleNamespace(label="Weekly", used_percent=next(values))])

    monkeypatch.setattr(account_usage, "fetch_account_usage", fake_usage)
    assert guard.check_daily_budget() is None
    message = guard.check_daily_budget()
    assert message is not None
    assert "11.0%" in message
    assert "blocked until tomorrow" in message


def test_state_survives_process_style_reload(monkeypatch, tmp_path):
    from agent import account_usage, codex_quota_guard as guard

    path = tmp_path / "state.json"
    monkeypatch.setattr(guard, "_state_path", lambda: path)
    monkeypatch.setattr(guard, "configured_daily_budget", lambda: 5.0)
    monkeypatch.setattr(
        account_usage,
        "fetch_account_usage",
        lambda *a, **k: SimpleNamespace(
            windows=[SimpleNamespace(label="Weekly", used_percent=40.0)]
        ),
    )
    assert guard.check_daily_budget() is None
    assert path.exists()

    monkeypatch.setattr(
        account_usage,
        "fetch_account_usage",
        lambda *a, **k: SimpleNamespace(
            windows=[SimpleNamespace(label="Weekly", used_percent=45.1)]
        ),
    )
    assert guard.check_daily_budget() is not None

def test_weekly_reset_restarts_same_day_baseline(monkeypatch, tmp_path):
    from agent import account_usage, codex_quota_guard as guard

    path = tmp_path / "state.json"
    path.write_text(
        '{"day": "' + __import__("datetime").date.today().isoformat() +
        '", "baseline_weekly_percent": 80.0}',
        encoding="utf-8",
    )

    monkeypatch.setattr(guard, "_state_path", lambda: path)
    monkeypatch.setattr(guard, "configured_daily_budget", lambda: 10.0)
    monkeypatch.setattr(
        account_usage,
        "fetch_account_usage",
        lambda *a, **k: SimpleNamespace(
            windows=[SimpleNamespace(label="Weekly", used_percent=5.0)]
        ),
    )

    assert guard.check_daily_budget() is None

    state = __import__("json").loads(path.read_text(encoding="utf-8"))
    assert state["baseline_weekly_percent"] == 5.0
