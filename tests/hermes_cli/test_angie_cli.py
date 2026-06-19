"""Tests for the Angie production Hermes helper CLI."""

from __future__ import annotations

import json
from argparse import Namespace

import pytest

from hermes_cli import angie


def _quiet_readonly(command, *, timeout=5.0):
    return None, f"{command[0]} unavailable in test"


def test_redact_removes_secret_values_and_non_prod_home():
    text = (
        "token=xoxb-1...cret "
        "Authorization: Bearer *** "
        "DATABASE_URL=postgresql://user:SECRET_PLACEHOLDER@example.invalid:5432/app "
        "path=/home/joesu/.hermes"
    )

    redacted = angie.redact(text)

    assert "xoxb-1234567890-secret" not in redacted
    assert "abc.def.ghi" not in redacted
    assert "postgresql://user:SECRET_PLACEHOLDER@example.invalid:5432/app" not in redacted
    assert "/home/joesu" not in redacted
    assert "<NON_PROD_HOME>" in redacted


def test_doctor_relative_hermes_home_is_usage_error(capsys):
    code = angie.run_hermes_doctor(Namespace(hermes_home="relative/path", json=True))

    captured = capsys.readouterr().out
    payload = json.loads(captured)
    assert code == 2
    assert payload["status"] == "blocker"
    assert payload["checks"][0]["id"] == "input.hermes_home"


def test_doctor_reports_missing_config_as_blocker(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(angie, "_run_readonly", _quiet_readonly)
    monkeypatch.setattr(angie.shutil, "which", lambda _cmd: "/usr/bin/hermes")
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir(mode=0o700)

    code = angie.run_hermes_doctor(Namespace(hermes_home=str(hermes_home), json=True))

    payload = json.loads(capsys.readouterr().out)
    assert code == 1
    assert payload["status"] == "blocker"
    by_id = {check["id"]: check for check in payload["checks"]}
    assert by_id["config.parse"]["status"] == "fail"
    assert by_id["config.parse"]["severity"] == "blocker"


def test_doctor_json_contains_key_names_not_secret_values(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(angie, "_run_readonly", _quiet_readonly)
    monkeypatch.setattr(angie.shutil, "which", lambda _cmd: "/usr/bin/hermes")
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir(mode=0o700)
    (hermes_home / "config.yaml").write_text(
        "gateway:\n  slack:\n    enabled: true\n    app_id: A123\n"
        "memory:\n  provider: builtin\n",
        encoding="utf-8",
    )
    env_path = hermes_home / ".env"
    env_path.write_text(
        "SLACK_BOT_TOKEN=xoxb-raw-secret\n"
        "OPENAI_API_KEY=sk-raw-secret\n"
        "HERMES_PUBLIC_SETTING=ok\n",
        encoding="utf-8",
    )
    env_path.chmod(0o600)

    code = angie.run_hermes_doctor(Namespace(hermes_home=str(hermes_home), json=True))

    raw = capsys.readouterr().out
    payload = json.loads(raw)
    assert code == 0
    assert payload["redactions_applied"] is True
    assert "xoxb-raw-secret" not in raw
    assert "sk-raw-secret" not in raw
    assert "SLACK_BOT_TOKEN" in raw
    assert "OPENAI_API_KEY" in raw


def test_doctor_flags_non_prod_home_literal_under_plugins(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(angie, "_run_readonly", _quiet_readonly)
    monkeypatch.setattr(angie.shutil, "which", lambda _cmd: "/usr/bin/hermes")
    hermes_home = tmp_path / ".hermes"
    plugin = hermes_home / "plugins" / "example"
    plugin.mkdir(parents=True)
    (hermes_home / "config.yaml").write_text("memory:\n  provider: builtin\n", encoding="utf-8")
    env_path = hermes_home / ".env"
    env_path.write_text("OPENAI_API_KEY=sk-secret\n", encoding="utf-8")
    env_path.chmod(0o600)
    (plugin / "plugin.py").write_text("CACHE='/home/joesu/tmp/cache'\n", encoding="utf-8")

    code = angie.run_hermes_doctor(Namespace(hermes_home=str(hermes_home), json=True))

    raw = capsys.readouterr().out
    payload = json.loads(raw)
    by_id = {check["id"]: check for check in payload["checks"]}
    assert code == 1
    assert by_id["plugins.non_prod_home_scan"]["severity"] == "blocker"
    assert "/home/joesu" not in raw
    assert "<NON_PROD_HOME>" in raw


def test_plugin_sync_dry_run_inventory_excludes_interactive_cli(tmp_path, capsys):
    source = tmp_path / "plugins" / "hermes"
    (source / "alpha").mkdir(parents=True)
    (source / "interactive-cli").mkdir()
    (source / "alpha" / "plugin.py").write_text("SLACK_BOT_TOKEN = ''\n", encoding="utf-8")
    hermes_home = tmp_path / "home" / ".hermes"
    hermes_home.mkdir(parents=True)

    code = angie.run_plugins_sync(
        Namespace(hermes_home=str(hermes_home), source=str(source), dry_run=True, json=True)
    )

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    by_plugin = {item["plugin"]: item for item in payload["inventory"]}
    assert by_plugin["alpha"]["decision"] == "sync_candidate"
    assert "SLACK_BOT_TOKEN" in by_plugin["alpha"]["env_key_names"]
    assert by_plugin["interactive-cli"]["decision"] == "exclude"


def test_plugin_sync_refuses_non_dry_run(tmp_path, capsys):
    source = tmp_path / "plugins" / "hermes"
    (source / "alpha").mkdir(parents=True)
    hermes_home = tmp_path / "home" / ".hermes"
    hermes_home.mkdir(parents=True)

    code = angie.run_plugins_sync(
        Namespace(hermes_home=str(hermes_home), source=str(source), dry_run=False, json=False)
    )

    captured = capsys.readouterr()
    assert code == 2
    assert "Refusing to mutate plugins" in captured.err
