from types import SimpleNamespace

from hermes_cli.status import show_status


def _patch_common_status_deps(monkeypatch, status_mod, tmp_path):
    import hermes_cli.auth as auth_mod

    monkeypatch.setattr(status_mod, "get_env_path", lambda: tmp_path / ".env", raising=False)
    monkeypatch.setattr(status_mod, "get_hermes_home", lambda: tmp_path, raising=False)
    monkeypatch.setattr(status_mod, "load_config", lambda: {"model": "gpt-5.4"}, raising=False)
    monkeypatch.setattr(status_mod, "resolve_requested_provider", lambda requested=None: "openai-codex", raising=False)
    monkeypatch.setattr(status_mod, "resolve_provider", lambda requested=None, **kwargs: "openai-codex", raising=False)
    monkeypatch.setattr(status_mod, "provider_label", lambda provider: "OpenAI Codex", raising=False)
    monkeypatch.setattr(auth_mod, "get_nous_auth_status", lambda: {}, raising=False)
    monkeypatch.setattr(auth_mod, "get_codex_auth_status", lambda: {}, raising=False)
    monkeypatch.setattr(auth_mod, "get_qwen_auth_status", lambda: {}, raising=False)


def test_show_status_includes_tavily_key(monkeypatch, capsys, tmp_path):
    from hermes_cli import status as status_mod

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-1...cdef")
    _patch_common_status_deps(monkeypatch, status_mod, tmp_path)
    monkeypatch.setattr(status_mod, "_gateway_process_running", lambda: False, raising=False)
    monkeypatch.setattr(status_mod, "_gateway_service_loaded", lambda: False, raising=False)
    monkeypatch.setattr(status_mod, "_load_gateway_runtime_status", lambda: {}, raising=False)
    monkeypatch.setattr(status_mod, "_cron_jobs_summary", lambda: (0, 0, False), raising=False)
    monkeypatch.setattr(status_mod, "_cron_next_run", lambda: "(none)", raising=False)

    show_status(SimpleNamespace(all=False, deep=False))

    output = capsys.readouterr().out
    assert "Tavily" in output
    assert "tvly...cdef" in output


def test_show_status_termux_gateway_section_skips_systemctl(monkeypatch, capsys, tmp_path):
    from hermes_cli import status as status_mod
    import hermes_cli.gateway as gateway_mod

    monkeypatch.setenv("TERMUX_VERSION", "0.118.3")
    monkeypatch.setenv("PREFIX", "/data/data/com.termux/files/usr")
    _patch_common_status_deps(monkeypatch, status_mod, tmp_path)
    monkeypatch.setattr(gateway_mod, "find_gateway_pids", lambda exclude_pids=None: [], raising=False)
    monkeypatch.setattr(status_mod, "_gateway_process_running", lambda: False, raising=False)
    monkeypatch.setattr(status_mod, "_gateway_service_loaded", lambda: None, raising=False)
    monkeypatch.setattr(status_mod, "_load_gateway_runtime_status", lambda: {}, raising=False)
    monkeypatch.setattr(status_mod, "_cron_jobs_summary", lambda: (0, 0, False), raising=False)
    monkeypatch.setattr(status_mod, "_cron_next_run", lambda: "(none)", raising=False)

    def _unexpected_systemctl(*args, **kwargs):
        raise AssertionError("systemctl should not be called in the Termux status view")

    monkeypatch.setattr(status_mod.subprocess, "run", _unexpected_systemctl)

    status_mod.show_status(SimpleNamespace(all=False, deep=False))

    output = capsys.readouterr().out
    assert "Manager:      Termux / manual process" in output
    assert "Start with:   hermes gateway" in output
    assert "systemd (user)" not in output


def test_show_status_prints_consolidated_health_for_gateway_signal_discord_cron(monkeypatch, capsys, tmp_path):
    from hermes_cli import status as status_mod

    _patch_common_status_deps(monkeypatch, status_mod, tmp_path)
    monkeypatch.setattr(status_mod, "_gateway_process_running", lambda: True, raising=False)
    monkeypatch.setattr(status_mod, "_gateway_service_loaded", lambda: True, raising=False)
    monkeypatch.setattr(
        status_mod,
        "_load_gateway_runtime_status",
        lambda: {
            "platforms": {
                "signal": {"state": "connected"},
                "discord": {"state": "retrying"},
            }
        },
        raising=False,
    )
    monkeypatch.setattr(status_mod, "_cron_jobs_summary", lambda: (2, 3, False), raising=False)
    monkeypatch.setattr(status_mod, "_cron_next_run", lambda: "2026-04-19T07:00:00Z", raising=False)

    show_status(SimpleNamespace(all=False, deep=False))

    output = capsys.readouterr().out
    assert "◆ Consolidated Health" in output
    assert "Gateway:      running" in output
    assert "Signal:       connected" in output
    assert "Discord:      retrying" in output
    assert "Cron:         2 active / 3 total" in output
    assert "Cron next:    2026-04-19T07:00:00Z" in output
