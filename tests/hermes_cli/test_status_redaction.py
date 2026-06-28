from types import SimpleNamespace


def _stub_status_dependencies(monkeypatch, status, auth, *, env_values=None, hermes_home=None):
    env_values = env_values or {}
    monkeypatch.setattr(status, "get_env_value", lambda name, *args, **kwargs: env_values.get(name, ""))
    monkeypatch.setattr(status, "get_env_path", lambda: SimpleNamespace(exists=lambda: True))
    monkeypatch.setattr(status, "load_config", lambda: {"model": {"default": "test-model"}})
    monkeypatch.setattr(status, "_effective_provider_label", lambda: "Test Provider")
    monkeypatch.setattr(auth, "get_anthropic_key", lambda: env_values.get("ANTHROPIC_API_KEY", ""))
    monkeypatch.setattr(auth, "get_nous_auth_status", lambda: {})
    monkeypatch.setattr(auth, "get_codex_auth_status", lambda: {})
    monkeypatch.setattr(auth, "get_qwen_auth_status", lambda: {})
    monkeypatch.setattr(auth, "get_minimax_oauth_auth_status", lambda: {})
    monkeypatch.setattr(auth, "get_xai_oauth_auth_status", lambda: {})
    if hermes_home is not None:
        monkeypatch.setattr(status, "get_hermes_home", lambda: hermes_home)


def test_status_all_redacts_api_key_values(monkeypatch, capsys):
    from hermes_cli import auth, status

    secrets = {
        "OPENROUTER_API_KEY": "sk-or-...7890",
        "ANTHROPIC_API_KEY": "sk-ant...7890",
    }

    _stub_status_dependencies(monkeypatch, status, auth, env_values=secrets)

    status.show_status(SimpleNamespace(all=True, deep=False))

    out = capsys.readouterr().out
    assert secrets["OPENROUTER_API_KEY"] not in out
    assert secrets["ANTHROPIC_API_KEY"] not in out
    assert status.redact_key(secrets["OPENROUTER_API_KEY"]) in out
    assert status.redact_key(secrets["ANTHROPIC_API_KEY"]) in out


def test_status_accepts_legacy_bare_list_jobs_file(monkeypatch, capsys, tmp_path):
    from hermes_cli import auth, status

    cron_dir = tmp_path / "cron"
    cron_dir.mkdir()
    (cron_dir / "jobs.json").write_text("[]", encoding="utf-8")
    _stub_status_dependencies(monkeypatch, status, auth, hermes_home=tmp_path)

    status.show_status(SimpleNamespace(all=True, deep=False))

    out = capsys.readouterr().out
    assert "Jobs:         0 active, 0 total" in out
    assert "error reading jobs file" not in out
