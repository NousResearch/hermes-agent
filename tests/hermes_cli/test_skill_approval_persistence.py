"""Real CLI scope commands persist atomically into the active profile."""

from pathlib import Path

import pytest
import hermes_yaml as yaml

from hermes_constants import set_hermes_home_override, reset_hermes_home_override
from tools import write_approval as wa


@pytest.fixture
def cli_profile(tmp_path, monkeypatch):
    from cli import HermesCLI
    launch = tmp_path / "launch"
    profile = tmp_path / "profiles" / "beta"
    launch.mkdir()
    profile.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(launch))
    (launch / "config.yaml").write_text("skills:\n  write_approval: false\n")
    (profile / "config.yaml").write_text(
        "# operator policy\nskills:\n  write_approval: false # gate\n"
        "  write_approval_mode: create # scope\n  creation_nudge_interval: 7\n"
        "model:\n  default: 'kept/model' # model\n")
    cli = object.__new__(HermesCLI)
    cli._slash_metrics_surface = None
    cli.agent = None
    token = set_hermes_home_override(profile)
    try:
        yield cli, launch, profile
    finally:
        reset_hermes_home_override(token)


def test_cli_mixed_roundtrips_preserve_queue_profile_comments_and_fresh_reads(cli_profile, capsys):
    cli, launch, home = cli_profile
    launch_before = (launch / "config.yaml").read_bytes()
    record = wa.stage_write("skills", {"action": "create", "name": "old-queued"},
                            summary="old queued request", origin="foreground")
    pending = home / "pending" / "skills" / (record["id"] + ".json")
    pending_before = pending.read_bytes()
    for command, enabled, scope in [
        ("approval all", True, "all"), ("approval off", False, "all"),
        ("approval on", True, "all"), ("mode create", True, "create"),
        ("mode off", False, "create"), ("mode on", True, "create"),
    ]:
        assert cli.process_command("/skills " + command)
        assert "set to" in capsys.readouterr().out
        config = yaml.safe_load((home / "config.yaml").read_text())
        assert config["skills"]["write_approval"] is enabled
        assert config["skills"]["write_approval_mode"] == scope
        assert wa.write_approval_enabled("skills") is enabled
        assert wa.skill_write_approval_mode() == scope
        assert pending.read_bytes() == pending_before
        assert not (home / "skills" / "old-queued").exists()
    text = (home / "config.yaml").read_text()
    for fragment in ["# operator policy", "# gate", "# scope", "'kept/model' # model",
                     "creation_nudge_interval: 7"]:
        assert fragment in text
    assert (launch / "config.yaml").read_bytes() == launch_before
    assert config["model"]["default"] == "kept/model"


@pytest.mark.parametrize("command", ["approval create", "approval all", "approval on", "approval off"])
def test_cli_failed_replace_leaves_entire_config_unchanged(cli_profile, monkeypatch, capsys, command):
    import utils
    cli, _launch, home = cli_profile
    before = (home / "config.yaml").read_bytes()
    def fail_replace(*args, **kwargs):
        raise OSError("simulated replace failure")
    monkeypatch.setattr(utils.os, "replace", fail_replace)
    cli.process_command("/skills " + command)
    assert "Failed to set" in capsys.readouterr().out
    assert (home / "config.yaml").read_bytes() == before


def test_cli_memory_stays_boolean_and_invalid_commands_do_not_write(cli_profile, capsys):
    cli, _launch, home = cli_profile
    for command in ["/memory approval create", "/memory mode all", "/skills approval typo",
                    "/skills approval create extra"]:
        before = (home / "config.yaml").read_bytes()
        cli.process_command(command)
        assert "Invalid value" in capsys.readouterr().out
        assert (home / "config.yaml").read_bytes() == before
    for mode, enabled in [("on", True), ("off", False)]:
        cli.process_command("/memory approval " + mode)
        assert "set to '" + mode + "'" in capsys.readouterr().out
        config = yaml.safe_load((home / "config.yaml").read_text())
        assert config["memory"]["write_approval"] is enabled
        assert "write_approval_mode" not in config["memory"]


def test_cli_scope_selection_controls_real_skill_writes_without_approving_old_queue(cli_profile, capsys):
    import json
    from tools import skill_manager_tool as smt
    cli, _launch, home = cli_profile
    body = "---\nname: existing\ndescription: Use when testing skill scopes.\n---\n\nOld body.\n"
    path = home / "skills" / "existing" / "SKILL.md"
    path.parent.mkdir(parents=True)
    path.write_text(body)
    cli.process_command("/skills approval create")
    capsys.readouterr()
    edited = json.loads(smt.skill_manage(action="patch", name=str(path.parent),
                                        old_string="Old body.", new_string="New body."))
    assert edited["success"] and not edited.get("staged"), edited
    new = json.loads(smt.skill_manage(action="create", name="new-skill",
                                     content=body.replace("existing", "new-skill")))
    assert new["staged"], new
    cli.process_command("/skills approval all")
    capsys.readouterr()
    staged = json.loads(smt.skill_manage(action="patch", name=str(path.parent),
                                        old_string="New body.", new_string="All body."))
    assert staged["staged"], staged
    assert "New body." in path.read_text()
    cli.process_command("/skills approval off")
    capsys.readouterr()
    edited = json.loads(smt.skill_manage(action="patch", name="existing",
                                        old_string="New body.", new_string="Off body."))
    assert edited["success"] and not edited.get("staged"), edited
    assert {r["id"] for r in wa.list_pending("skills")} == {new["pending_id"], staged["pending_id"]}
    assert not (home / "skills" / "new-skill").exists()


@pytest.mark.parametrize("content", ["skills: [broken\n", "skills: not-a-mapping\n"])
def test_cli_unreadable_settings_fail_closed(cli_profile, content, capsys):
    cli, _launch, home = cli_profile
    path = home / "config.yaml"
    path.write_text(content)
    before = path.read_bytes()
    cli.process_command("/skills approval create")
    assert "Failed to set" in capsys.readouterr().out
    assert path.read_bytes() == before


def test_cli_hub_help_and_unknown_guidance_explain_approval_choices(cli_profile, capsys):
    cli, _launch, home = cli_profile
    before = (home / "config.yaml").read_bytes()
    for command in ["/skills help", "/skills unknown-verb"]:
        cli.process_command(command)
        output = capsys.readouterr().out
        assert "on|off|create|all" in output
        assert "/skills approval help" in output
        assert "pending writes stay pending" in output
        assert (home / "config.yaml").read_bytes() == before


def test_cli_legacy_unset_scope_remains_default_all(cli_profile, capsys):
    cli, _launch, home = cli_profile
    path = home / "config.yaml"
    path.write_text("skills:\n  write_approval: false\n")
    for mode, enabled in [("on", True), ("off", False)]:
        cli.process_command("/skills approval " + mode)
        assert "scope: all" in capsys.readouterr().out
        assert yaml.safe_load(path.read_text())["skills"] == {"write_approval": enabled}
    cli.process_command("/skills approval create")
    assert "scope: create" in capsys.readouterr().out
    assert yaml.safe_load(path.read_text())["skills"] == {"write_approval": True, "write_approval_mode": "create"}
