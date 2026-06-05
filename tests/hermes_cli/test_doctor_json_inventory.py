"""Tests for hermes doctor JSON inventory mode."""

import contextlib
import io
import json
from argparse import Namespace
from types import SimpleNamespace

from hermes_cli import doctor as doctor_mod


def test_doctor_json_inventory_redacts_env_contents(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir(parents=True)
    (home / "config.yaml").write_text(
        "model:\n  provider: openrouter\n  default: openai/gpt-4.1-mini\n",
        encoding="utf-8",
    )
    (home / ".env").write_text(
        "OPENROUTER_API_KEY=sk-test-value-that-must-not-leak\n",
        encoding="utf-8",
    )
    skills_dir = home / "skills" / "demo"
    skills_dir.mkdir(parents=True)
    (skills_dir / "SKILL.md").write_text("---\nname: demo\n---\n", encoding="utf-8")

    monkeypatch.setattr(doctor_mod, "HERMES_HOME", home)
    monkeypatch.setattr(doctor_mod, "PROJECT_ROOT", tmp_path / "project")
    monkeypatch.setattr(doctor_mod, "_DHH", str(home))

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        doctor_mod.run_doctor(Namespace(fix=False, ack=None, json=True, all_profiles=False))

    output = buf.getvalue()
    assert "sk-test-value-that-must-not-leak" not in output
    payload = json.loads(output)
    assert payload["schema_version"] == 1
    assert payload["mode"] == "inventory"
    assert payload["profiles"][0]["name"] == "current"
    assert payload["profiles"][0]["has_env"] is True
    assert payload["profiles"][0]["env_keys_present"] == ["OPENROUTER_API_KEY"]
    assert payload["profiles"][0]["skill_count"] == 1
    assert payload["profiles"][0]["issues"] == []
    assert payload["repair_plan"] == []


def test_doctor_json_all_profiles_reports_safe_repair_plan(monkeypatch, tmp_path):
    default_home = tmp_path / ".hermes"
    profiles_root = default_home / "profiles"
    broken = profiles_root / "broken"
    ok = profiles_root / "ok"
    broken.mkdir(parents=True)
    ok.mkdir(parents=True)

    (default_home / "config.yaml").write_text("model:\n  provider: custom\n", encoding="utf-8")
    (ok / "config.yaml").write_text(
        "model:\n  provider: anthropic\n  default: claude-sonnet-4\n",
        encoding="utf-8",
    )
    (ok / ".env").write_text("ANTHROPIC_API_KEY=redacted-in-json\n", encoding="utf-8")
    (broken / ".env").write_text("OPENAI_API_KEY=also-redacted\n", encoding="utf-8")

    monkeypatch.setattr(doctor_mod, "HERMES_HOME", default_home)
    monkeypatch.setattr(doctor_mod, "PROJECT_ROOT", tmp_path / "project")
    monkeypatch.setattr(doctor_mod, "_DHH", str(default_home))

    fake_profiles = [
        SimpleNamespace(
            name="default",
            path=default_home,
            is_default=True,
            gateway_running=False,
            model=None,
            provider="custom",
            has_env=False,
            skill_count=0,
            alias_path=None,
        ),
        SimpleNamespace(
            name="broken",
            path=broken,
            is_default=False,
            gateway_running=False,
            model=None,
            provider=None,
            has_env=True,
            skill_count=0,
            alias_path=None,
        ),
        SimpleNamespace(
            name="ok",
            path=ok,
            is_default=False,
            gateway_running=True,
            model="claude-sonnet-4",
            provider="anthropic",
            has_env=True,
            skill_count=0,
            alias_path=tmp_path / "ok",
        ),
    ]
    monkeypatch.setattr(doctor_mod, "_list_doctor_profiles", lambda: fake_profiles)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        doctor_mod.run_doctor(Namespace(fix=False, ack=None, json=True, all_profiles=True))

    output = buf.getvalue()
    assert "redacted-in-json" not in output
    assert "also-redacted" not in output
    payload = json.loads(output)
    assert payload["profile_count"] == 3
    by_name = {profile["name"]: profile for profile in payload["profiles"]}
    assert by_name["ok"]["gateway_running"] is True
    assert by_name["ok"]["env_keys_present"] == ["ANTHROPIC_API_KEY"]
    assert "Missing config.yaml" in by_name["broken"]["issues"]
    assert any("broken: create config.yaml" in item for item in payload["repair_plan"])
    assert "redacted-in-json" not in json.dumps(payload)
