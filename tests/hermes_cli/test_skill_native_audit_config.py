"""Native skill-call audit configuration in an isolated profile."""

import json
from pathlib import Path

import pytest

import hermes_yaml as yaml
from hermes_cli.config import atomic_config_write, load_config, save_config


@pytest.fixture
def audit_profile(tmp_path, monkeypatch):
    profile = tmp_path / "profile"
    profile.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(profile))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return profile


def test_new_profile_loads_native_call_audit_as_exact_false(audit_profile):
    config = load_config()

    assert config["skills"].get("native_call_audit") is False


def test_generated_defaults_persist_native_call_audit_as_yaml_false(audit_profile):
    save_config(load_config(), strip_defaults=False)

    generated = yaml.safe_load((audit_profile / "config.yaml").read_text(encoding="utf-8"))
    assert generated["skills"]["native_call_audit"] is False
    assert load_config()["skills"]["native_call_audit"] is False


def test_existing_skills_config_without_audit_setting_remains_disabled(audit_profile):
    atomic_config_write(audit_profile / "config.yaml", {"skills": {"ledger": False}})

    config = load_config()

    assert config["skills"]["native_call_audit"] is False
    assert config["skills"]["ledger"] is False


@pytest.mark.parametrize("enabled", [True, False])
def test_explicit_native_call_audit_boolean_survives_load_and_save(audit_profile, enabled):
    atomic_config_write(
        audit_profile / "config.yaml", {"skills": {"native_call_audit": enabled}}
    )

    config = load_config()
    assert config["skills"]["native_call_audit"] is enabled
    save_config(config)

    saved = yaml.safe_load((audit_profile / "config.yaml").read_text(encoding="utf-8"))
    assert saved["skills"]["native_call_audit"] is enabled
    assert load_config()["skills"]["native_call_audit"] is enabled


@pytest.mark.parametrize("explicit_false", [False, True])
def test_disabled_native_dispatch_creates_no_audit_files(audit_profile, explicit_false):
    # Isolate every writable skill destination without mocking native dispatch.
    skills = {"external_dirs": [], "create_dir": "", "project_discovery": False}
    if explicit_false:
        skills["native_call_audit"] = False
    atomic_config_write(audit_profile / "config.yaml", {"skills": skills})

    from agent.skill_utils import get_all_skills_dirs
    from hermes_constants import get_hermes_home
    from model_tools import handle_function_call
    from tools import skill_manager_tool

    assert get_hermes_home().resolve() == audit_profile.resolve()
    assert skill_manager_tool._skills_dir().resolve() == (audit_profile / "skills").resolve()
    assert all(
        path.resolve().is_relative_to(audit_profile.resolve())
        for path in get_all_skills_dirs()
    )
    assert load_config()["skills"]["native_call_audit"] is False

    content = (
        "---\nname: config-audit-fixture\n"
        "description: Use when testing an isolated native audit configuration.\n"
        "---\n\nOwned default-disabled fixture.\n"
    )
    result = handle_function_call(
        "skill_manage",
        {"operations": [{"action": "create", "name": "config-audit-fixture", "content": content}]},
        session_id="config-audit-fixture-session",
    )

    assert json.loads(result)["success"] is True
    assert (audit_profile / "skills" / "config-audit-fixture" / "SKILL.md").is_file()
    assert not (audit_profile / "skill_native_audit.key").exists()
    assert not (audit_profile / "skill_native_audit.jsonl").exists()
