"""UTF-8 BOM tolerance for independent jobs.json readers (dump/status).

cron/jobs.load_jobs is covered in tests/cron/test_jobs.py. dump and status
each open jobs.json themselves — keep them on the same utf-8-sig dialect.
"""

from types import SimpleNamespace


def test_dump_cron_summary_accepts_utf8_bom(tmp_path):
    from hermes_cli.dump import _cron_summary

    cron = tmp_path / "cron"
    cron.mkdir()
    (cron / "jobs.json").write_bytes(
        b'\xef\xbb\xbf{"jobs": [{"id": "j1", "enabled": true},'
        b' {"id": "j2", "enabled": false}]}'
    )

    assert _cron_summary(tmp_path) == "1 active / 2 total"


def test_dump_cron_summary_bomless_regression(tmp_path):
    from hermes_cli.dump import _cron_summary

    cron = tmp_path / "cron"
    cron.mkdir()
    (cron / "jobs.json").write_text(
        '{"jobs": [{"id": "j1", "enabled": true}]}',
        encoding="utf-8",
    )

    assert _cron_summary(tmp_path) == "1 active / 1 total"


def test_status_scheduled_jobs_accepts_utf8_bom(monkeypatch, capsys, tmp_path):
    """hermes status must not print '(error reading jobs file)' under BOM."""
    from hermes_cli import status as status_mod
    import hermes_cli.auth as auth_mod
    import hermes_cli.gateway as gateway_mod

    cron = tmp_path / "cron"
    cron.mkdir()
    (cron / "jobs.json").write_bytes(
        b'\xef\xbb\xbf{"jobs": [{"id": "j1", "enabled": true},'
        b' {"id": "j2", "enabled": true}]}'
    )

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(status_mod, "get_env_path", lambda: tmp_path / ".env", raising=False)
    monkeypatch.setattr(status_mod, "get_hermes_home", lambda: tmp_path, raising=False)
    monkeypatch.setattr(status_mod, "load_config", lambda: {"model": "gpt-5.4"}, raising=False)
    monkeypatch.setattr(
        status_mod, "resolve_requested_provider", lambda requested=None: "openai-codex", raising=False
    )
    monkeypatch.setattr(
        status_mod, "resolve_provider", lambda requested=None, **kwargs: "openai-codex", raising=False
    )
    monkeypatch.setattr(status_mod, "provider_label", lambda provider: "OpenAI Codex", raising=False)
    monkeypatch.setattr(auth_mod, "get_nous_auth_status", lambda: {}, raising=False)
    monkeypatch.setattr(auth_mod, "get_codex_auth_status", lambda: {}, raising=False)
    monkeypatch.setattr(auth_mod, "get_xai_oauth_auth_status", lambda: {}, raising=False)
    monkeypatch.setattr(gateway_mod, "find_gateway_pids", lambda exclude_pids=None: [], raising=False)

    status_mod.show_status(SimpleNamespace(all=False, deep=False))
    out = capsys.readouterr().out
    assert "(error reading jobs file)" not in out
    assert "2 active, 2 total" in out
