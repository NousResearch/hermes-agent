from types import SimpleNamespace

from hermes_cli import status as status_mod


def test_show_status_includes_tavily_key(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-1...cdef")

    status_mod.show_status(SimpleNamespace(all=False, deep=False))

    output = capsys.readouterr().out
    assert "Tavily" in output
    assert "tvly...cdef" in output


def test_show_status_reports_degraded_gateway_runtime(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(status_mod, "get_env_path", lambda: tmp_path / ".env", raising=False)
    monkeypatch.setattr(status_mod, "get_hermes_home", lambda: tmp_path, raising=False)
    monkeypatch.setattr(status_mod, "load_config", lambda: {}, raising=False)
    monkeypatch.setattr(status_mod, "get_env_value", lambda name: {"TELEGRAM_BOT_TOKEN": "123:abc"}.get(name, ""), raising=False)
    monkeypatch.setattr(status_mod, "resolve_requested_provider", lambda requested=None: "auto", raising=False)
    monkeypatch.setattr(status_mod, "resolve_provider", lambda requested=None, **kwargs: "openrouter", raising=False)
    monkeypatch.setattr(status_mod, "provider_label", lambda provider: provider, raising=False)
    monkeypatch.setattr(
        status_mod.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="active\n", returncode=0),
    )
    monkeypatch.setattr(
        status_mod,
        "_load_gateway_runtime_snapshot",
        lambda: {
            "gateway_state": "running",
            "platforms": {
                "telegram": {
                    "state": "reconnecting",
                    "error_message": "Timed out",
                }
            },
        },
        raising=False,
    )

    status_mod.show_status(SimpleNamespace(all=False, deep=False))

    output = capsys.readouterr().out
    assert "running (degraded)" in output
    assert "Telegram" in output
    assert "runtime: reconnecting (Timed out)" in output
    assert "Runtime:      ⚠ telegram: reconnecting — Timed out" in output


def test_show_status_ignores_stale_disconnected_and_unconfigured_runtime(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(status_mod, "get_env_path", lambda: tmp_path / ".env", raising=False)
    monkeypatch.setattr(status_mod, "get_hermes_home", lambda: tmp_path, raising=False)
    monkeypatch.setattr(status_mod, "load_config", lambda: {}, raising=False)
    monkeypatch.setattr(
        status_mod,
        "get_env_value",
        lambda name: {
            "TELEGRAM_BOT_TOKEN": "123:abc",
            "SLACK_BOT_TOKEN": "xoxb-123",
        }.get(name, ""),
        raising=False,
    )
    monkeypatch.setattr(status_mod, "resolve_requested_provider", lambda requested=None: "auto", raising=False)
    monkeypatch.setattr(status_mod, "resolve_provider", lambda requested=None, **kwargs: "openrouter", raising=False)
    monkeypatch.setattr(status_mod, "provider_label", lambda provider: provider, raising=False)
    monkeypatch.setattr(
        status_mod.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="active\n", returncode=0),
    )
    monkeypatch.setattr(
        status_mod,
        "_load_gateway_runtime_snapshot",
        lambda: {
            "gateway_state": "running",
            "platforms": {
                "telegram": {"state": "disconnected", "error_message": ""},
                "whatsapp": {"state": "fatal", "error_message": "stale whatsapp failure"},
                "webhook": {"state": "fatal", "error_message": "stale webhook failure"},
            },
        },
        raising=False,
    )

    status_mod.show_status(SimpleNamespace(all=False, deep=False))

    output = capsys.readouterr().out
    assert "running (degraded)" not in output
    assert "runtime: disconnected" not in output
    assert "stale whatsapp failure" not in output
    assert "stale webhook failure" not in output
