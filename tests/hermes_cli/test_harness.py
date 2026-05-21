from hermes_cli import harness


def test_ensure_harness_running_respects_env_opt_out(monkeypatch):
    calls = []

    monkeypatch.setenv("HYPURA_HARNESS_AUTO_START", "0")
    monkeypatch.setattr(
        harness,
        "load_config",
        lambda: {"harness": {"enabled": True, "auto_start": True}},
    )
    monkeypatch.setattr(harness, "is_harness_running", lambda: calls.append("probe") or False)
    monkeypatch.setattr(harness, "start_harness_daemon", lambda: calls.append("start") or True)

    harness.ensure_harness_running()

    assert calls == []


def test_ensure_harness_running_starts_when_enabled(monkeypatch):
    calls = []

    monkeypatch.delenv("HYPURA_HARNESS_AUTO_START", raising=False)
    monkeypatch.delenv("HERMES_HARNESS_AUTO_START", raising=False)
    monkeypatch.setattr(
        harness,
        "load_config",
        lambda: {"harness": {"enabled": True, "auto_start": True}},
    )
    monkeypatch.setattr(harness, "is_harness_running", lambda: calls.append("probe") or False)
    monkeypatch.setattr(harness, "start_harness_daemon", lambda: calls.append("start") or True)

    harness.ensure_harness_running()

    assert calls == ["probe", "start"]


def test_ensure_harness_running_respects_config_auto_start(monkeypatch):
    calls = []

    monkeypatch.delenv("HYPURA_HARNESS_AUTO_START", raising=False)
    monkeypatch.setattr(
        harness,
        "load_config",
        lambda: {"harness": {"enabled": True, "auto_start": False}},
    )
    monkeypatch.setattr(harness, "is_harness_running", lambda: calls.append("probe") or False)
    monkeypatch.setattr(harness, "start_harness_daemon", lambda: calls.append("start") or True)

    harness.ensure_harness_running()

    assert calls == []
