from __future__ import annotations

from cli import HermesCLI
from hermes_cli import fetch


def test_cli_fetch_slash_command_prints_overview(monkeypatch, capsys):
    cli_obj = HermesCLI.__new__(HermesCLI)
    cli_obj._pending_resume_sessions = None

    monkeypatch.setattr(
        fetch,
        "collect_fetch_info",
        lambda: {
            "generated_at": "2026-06-02T00:00:00+00:00",
            "version": "0.test",
            "release_date": "2026.6.2",
            "repo": {"branch": "main", "commit": "abc123", "dirty": False, "path": "~/hermes-agent"},
            "profile": "default",
            "persona": "Atsuko",
            "hermes_home": "~/.hermes",
            "model": {"provider": "openai-codex", "name": "gpt-5.5"},
            "gateway": "running (manual, pid 123)",
            "platforms": ["telegram ✓"],
            "tools": "57 available",
            "skills": 42,
            "cron": {"active": 2, "paused": 1, "total": 3},
            "memory": "enabled (built-in)",
            "mcp_servers": 1,
            "host": {"system": "Android", "release": "16", "machine": "aarch64", "termux": True},
            "runtime": {"python": "3.13.0", "node": "v26.0.0", "npm": "11.0.0"},
            "update": "up to date",
        },
    )

    assert cli_obj.process_command("/fetch compact") is True
    out = capsys.readouterr().out
    assert "Hermes Agent" in out
    assert "openai-codex / gpt-5.5" in out
    assert "Tools" not in out
