"""Tests for channel-skill administration and CLI output."""

from types import SimpleNamespace

from gateway.channel_skill_admin import ChannelSkillAdmin
from gateway.channel_skill_generation import ChannelSkillRegistry
from hermes_cli.channel_skills_command import channel_skills_command


def _write_skill(tmp_path, name="discord-channel-123", channel_id="123"):
    skill_dir = tmp_path / "skills" / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        f"""---
name: {name}
description: Use when handling this channel.
version: 0.1.0
author: Test
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    generated: true
    platform: discord
    channel_id: \"{channel_id}\"
---

# Rules

- Keep the channel rules.
""",
        encoding="utf-8",
    )
    return skill_dir


def _registry(tmp_path, name="discord-channel-123", status="ready"):
    registry = ChannelSkillRegistry(tmp_path / "channel-skills.yaml")
    registry.put("123", {
        "platform": "discord",
        "channel_id": "123",
        "skill": name,
        "profile": name,
        "resolved_skills": [name],
        "status": status,
        "generated": True,
    })
    return registry


def test_admin_reports_ready_and_file_integrity(tmp_path, monkeypatch):
    skill_dir = _write_skill(tmp_path)
    registry = _registry(tmp_path)
    monkeypatch.setattr("gateway.channel_skill_admin._find_skill", lambda name: {"path": skill_dir})

    info = ChannelSkillAdmin(registry).show("123")

    assert info["status"] == "ready"
    assert info["file"]["exists"] is True
    assert info["file"]["frontmatter"] == "valid"
    assert info["file"]["name"] == "matched"
    assert info["file"]["channel_id"] == "matched"


def test_admin_detects_missing_skill_file(tmp_path, monkeypatch):
    registry = _registry(tmp_path)
    monkeypatch.setattr("gateway.channel_skill_admin._find_skill", lambda name: None)

    info = ChannelSkillAdmin(registry).show("123")

    assert info["status"] == "missing"
    assert info["file"]["exists"] is False


def test_remove_detaches_by_default_and_keeps_skill(tmp_path, monkeypatch):
    skill_dir = _write_skill(tmp_path)
    registry = _registry(tmp_path)
    monkeypatch.setattr("gateway.channel_skill_admin._find_skill", lambda name: {"path": skill_dir})

    result = ChannelSkillAdmin(registry).remove("123")

    assert result["detached"] is True
    assert (skill_dir / "SKILL.md").exists()
    assert registry.get("123") is None


def test_cli_list_json_is_machine_readable(tmp_path, monkeypatch, capsys):
    registry = _registry(tmp_path)
    monkeypatch.setattr("hermes_cli.channel_skills_command._registry", lambda: registry)

    rc = channel_skills_command(SimpleNamespace(action="list", json=True))
    output = capsys.readouterr().out

    assert rc == 0
    assert '"channel_id": "123"' in output


def test_admin_list_all_includes_static_binding(tmp_path, monkeypatch):
    skill_dir = _write_skill(tmp_path, name="static-skill", channel_id="456")
    registry = _registry(tmp_path)
    monkeypatch.setattr("gateway.channel_skill_admin._find_skill", lambda name: {"path": skill_dir} if name == "static-skill" else None)

    admin = ChannelSkillAdmin(registry, config_extra={
        "channel_skill_bindings": [{"id": "456", "skills": ["static-skill"]}],
    })

    items = admin.list(include_static=True)

    assert {item["channel_id"] for item in items} == {"123", "456"}
    static = next(item for item in items if item["channel_id"] == "456")
    assert static["source"] == "static"
    assert static["binding_type"] == "exact"
    assert static["effective_skills"] == ["static-skill"]


def test_admin_effective_resolves_static_profile(tmp_path):
    admin = ChannelSkillAdmin(ChannelSkillRegistry(tmp_path / "channel-skills.yaml"), config_extra={
        "channel_skill_profiles": {
            "base": {"skills": ["base-skill"]},
            "child": {"extends": "base", "skills": ["child-skill"]},
        },
        "channel_skill_bindings": [{"id": "456", "profile": "child"}],
    })

    result = admin.effective("456")

    assert result["effective_skills"] == ["base-skill", "child-skill"]
    assert result["profile_chain"] == ["base", "child"]


def test_cli_effective_json_is_machine_readable(tmp_path, monkeypatch, capsys):
    registry = ChannelSkillRegistry(tmp_path / "channel-skills.yaml")
    monkeypatch.setattr("hermes_cli.channel_skills_command._registry", lambda: registry)
    monkeypatch.setattr("hermes_cli.channel_skills_command._config_extra", lambda: {
        "channel_skill_bindings": [{"id": "456", "skills": ["static-skill"]}],
    })

    rc = channel_skills_command(SimpleNamespace(action="effective", channel_id="456", json=True))
    output = capsys.readouterr().out

    assert rc == 0
    assert '"static-skill"' in output


def test_admin_validate_reports_missing_static_skill(tmp_path, monkeypatch):
    monkeypatch.setattr("gateway.channel_skill_admin._find_skill", lambda name: None)
    admin = ChannelSkillAdmin(ChannelSkillRegistry(tmp_path / "channel-skills.yaml"), config_extra={
        "channel_skill_bindings": [{"id": "456", "skills": ["missing-skill"]}],
    })

    result = admin.validate()

    assert result["valid"] is False
    assert result["issues"][0]["type"] == "missing_skill"


def test_regenerate_with_backup_preserves_previous_skill(tmp_path, monkeypatch):
    skill_dir = _write_skill(tmp_path)
    registry = _registry(tmp_path)
    monkeypatch.setattr("gateway.channel_skill_admin._find_skill", lambda name: {"path": skill_dir})
    monkeypatch.setattr("gateway.channel_skill_admin._generate_with_auxiliary_llm", lambda **kwargs: "---\nname: discord-channel-123\ndescription: Use in Discord channel 123.\n---\n\n# Updated\n")
    monkeypatch.setattr("gateway.channel_skill_admin.get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr("tools.skill_manager_tool._edit_skill", lambda name, content: {"success": True, "skill_md": str(skill_dir / "SKILL.md")})

    result = ChannelSkillAdmin(registry).regenerate("123", backup=True)

    assert result["backup_path"]
    assert (tmp_path / "channel-skills" / "backups" / "123").is_dir()


def test_cli_list_all_json_reports_source(tmp_path, monkeypatch, capsys):
    registry = _registry(tmp_path)
    monkeypatch.setattr("hermes_cli.channel_skills_command._registry", lambda: registry)
    monkeypatch.setattr("hermes_cli.channel_skills_command._config_extra", lambda: {
        "channel_skill_bindings": [{"id": "456", "skills": ["static-skill"]}],
    })

    rc = channel_skills_command(SimpleNamespace(action="list", all=True, json=True))
    output = capsys.readouterr().out

    assert rc == 0
    assert '"source": "generated"' in output
    assert '"source": "static"' in output


def test_static_status_filter_applies_to_static_bindings(tmp_path, monkeypatch):
    skill_dir = _write_skill(tmp_path, name="static-skill", channel_id="456")
    monkeypatch.setattr("gateway.channel_skill_admin._find_skill", lambda name: {"path": skill_dir})
    admin = ChannelSkillAdmin(ChannelSkillRegistry(tmp_path / "channel-skills.yaml"), config_extra={
        "channel_skill_bindings": [{"id": "456", "skills": ["static-skill"]}],
    })

    assert admin.list(include_static=True, status="ready") == []
    assert [item["channel_id"] for item in admin.list(include_static=True, status="configured")] == ["456"]


def test_static_binding_overrides_duplicate_generated_record(tmp_path, monkeypatch):
    skill_dir = _write_skill(tmp_path, name="static-skill", channel_id="456")
    registry = ChannelSkillRegistry(tmp_path / "channel-skills.yaml")
    registry.put("456", {
        "platform": "discord", "channel_id": "456", "skill": "discord-channel-456",
        "resolved_skills": ["discord-channel-456"], "status": "ready", "generated": True,
    })
    monkeypatch.setattr("gateway.channel_skill_admin._find_skill", lambda name: {"path": skill_dir} if name == "static-skill" else None)
    admin = ChannelSkillAdmin(registry, config_extra={
        "channel_skill_bindings": [{"id": "456", "skills": ["static-skill"]}],
    })

    items = admin.list(include_static=True)

    item = next(item for item in items if item["channel_id"] == "456")
    assert item["source"] == "static"
    assert item["effective_skills"] == ["static-skill"]


def test_malformed_skill_finder_is_reported_as_missing(tmp_path, monkeypatch):
    monkeypatch.setattr("gateway.channel_skill_admin._find_skill", lambda name: {"path": object()})
    admin = ChannelSkillAdmin(ChannelSkillRegistry(tmp_path / "channel-skills.yaml"), config_extra={
        "channel_skill_bindings": [{"id": "456", "skills": ["bad-skill"]}],
    })

    result = admin.validate()

    assert result["valid"] is False
    assert result["issues"][0]["type"] == "missing_skill"


def test_regenerate_rejects_path_unsafe_channel_id(tmp_path):
    registry = ChannelSkillRegistry(tmp_path / "channel-skills.yaml")
    registry.put("../escape", {
        "platform": "discord", "channel_id": "../escape", "skill": "discord-channel-../escape",
        "resolved_skills": ["discord-channel-../escape"], "status": "ready", "generated": True,
    })

    try:
        ChannelSkillAdmin(registry).regenerate("../escape", dry_run=True)
    except (KeyError, ValueError):
        pass
    else:
        raise AssertionError("unsafe channel id was accepted")


def test_effective_does_not_fall_back_to_generated_for_invalid_static_binding(tmp_path):
    registry = ChannelSkillRegistry(tmp_path / "channel-skills.yaml")
    registry.put("456", {
        "platform": "discord", "channel_id": "456", "skill": "discord-channel-456",
        "resolved_skills": ["discord-channel-456"], "status": "ready", "generated": True,
    })
    admin = ChannelSkillAdmin(registry, config_extra={
        "channel_skill_profiles": {},
        "channel_skill_bindings": [{"id": "456", "profile": "missing-profile"}],
    })

    result = admin.effective("456")

    assert result["source"] == "static"
    assert result["effective_skills"] == []
    assert result["status"] == "invalid"


def test_cli_parent_json_flag_is_preserved(tmp_path, monkeypatch, capsys):
    registry = ChannelSkillRegistry(tmp_path / "channel-skills.yaml")
    monkeypatch.setattr("hermes_cli.channel_skills_command._registry", lambda: registry)
    monkeypatch.setattr("hermes_cli.channel_skills_command._config_extra", lambda: {})
    args = SimpleNamespace(action="list", root_json=True, json=False)

    rc = channel_skills_command(args)
    output = capsys.readouterr().out

    assert rc == 0
    assert output.strip() == "[]"
