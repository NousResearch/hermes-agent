from __future__ import annotations

import argparse
import json

from hermes_cli import fetch
from hermes_cli.commands import ACTIVE_SESSION_BYPASS_COMMANDS, resolve_command


def _info() -> dict:
    return {
        "generated_at": "2026-06-02T00:00:00+00:00",
        "version": "0.test",
        "release_date": "2026.6.2",
        "repo": {"branch": "main", "commit": "abc123", "dirty": False, "path": "~/hermes-agent"},
        "profile": "default",
        "persona": "Atsuko",
        "hermes_home": "~/.hermes",
        "model": {"provider": "openai-codex", "name": "gpt-5.5"},
        "gateway": "running (manual, pid 123)",
        "platforms": ["discord ✓", "telegram ✓"],
        "tools": "57 available",
        "skills": 42,
        "cron": {"active": 2, "paused": 1, "total": 3},
        "memory": "enabled (built-in)",
        "mcp_servers": 1,
        "host": {"system": "Android", "release": "16", "machine": "aarch64", "termux": True},
        "runtime": {"python": "3.13.0", "node": "v26.0.0", "npm": "11.0.0"},
        "update": "up to date",
    }


def test_render_compact_persona_overview():
    output = fetch.render_fetch_info(_info(), compact=True)

    assert "Hermes Agent" in output
    assert "persona-aware · Atsuko" in output
    assert "openai-codex / gpt-5.5" in output
    assert "discord ✓, telegram ✓" in output
    assert "2 active / 3 total" in output
    assert "Tools" not in output  # compact intentionally trims lower-priority rows


def test_render_plain_no_persona_is_neutral():
    output = fetch.render_fetch_info(_info(), plain=True, no_persona=True)

    assert "Persona:    disabled" in output
    assert "Hermes:     v0.test (2026.6.2)" in output
    assert "Host:       Android 16 aarch64 · Termux" in output
    assert "persona-aware" not in output
    assert "TOKEN" not in output.upper()


def test_run_fetch_json_uses_collector(monkeypatch, capsys):
    monkeypatch.setattr(fetch, "collect_fetch_info", _info)

    rc = fetch.run_fetch(argparse.Namespace(json=True, plain=False, no_persona=False, compact=False))

    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert data["profile"] == "default"
    assert data["model"] == {"provider": "openai-codex", "name": "gpt-5.5"}
    assert data["platforms"] == ["discord ✓", "telegram ✓"]


def test_cron_summary_reads_jobs_file(tmp_path):
    jobs_dir = tmp_path / "cron"
    jobs_dir.mkdir()
    (jobs_dir / "jobs.json").write_text(
        json.dumps({"jobs": [{"enabled": True}, {"enabled": False}, {}]}),
        encoding="utf-8",
    )

    assert fetch._cron_summary(tmp_path) == {"active": 2, "paused": 1, "total": 3}


def test_render_fetch_slash_args_supports_concise_formats(monkeypatch):
    monkeypatch.setattr(fetch, "collect_fetch_info", _info)

    compact = fetch.render_fetch_slash_args("compact")
    assert "Hermes Agent" in compact
    assert "Tools" not in compact

    plain = fetch.render_fetch_slash_args("--plain --no-persona")
    assert "Persona:    disabled" in plain

    data = json.loads(fetch.render_fetch_slash_args("json"))
    assert data["version"] == "0.test"


def test_fetch_is_registered_as_slash_command_and_midrun_bypass():
    cmd = resolve_command("fetch")

    assert cmd is not None
    assert cmd.name == "fetch"
    assert "fetch" in ACTIVE_SESSION_BYPASS_COMMANDS
