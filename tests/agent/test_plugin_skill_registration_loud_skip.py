"""sys-gn02e9: loud plugin-skill registration and scan skips.

register_skill must enforce the SKILL.md file contract (directories normalize
to their SKILL.md; anything else raises naming SKILL.md), and
_scan_plugin_skill_commands must log one warning per skipped qualified name so
a stale in-memory catalog is discoverable instead of silently shrinking the
command palette.
"""

import logging

import pytest


@pytest.fixture
def ctx(tmp_path, monkeypatch):
    from hermes_cli import plugins as plugins_mod
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest

    pm = PluginManager()
    monkeypatch.setattr(plugins_mod, "_plugin_manager", pm)
    manifest = PluginManifest(
        name="testplugin", version="1.0.0", description="test", source="user",
    )
    return PluginContext(manifest, pm)


class TestRegisterSkillFileContract:
    def test_directory_normalizes_to_skill_md(self, ctx, tmp_path):
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\nname: my-skill\n---\nContent.\n")

        ctx.register_skill("my-skill", skill_dir)

        assert ctx._manager.find_plugin_skill("testplugin:my-skill") == skill_dir / "SKILL.md"

    def test_directory_without_skill_md_raises_naming_skill_md(self, ctx, tmp_path):
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()

        with pytest.raises(FileNotFoundError, match="SKILL.md not found"):
            ctx.register_skill("my-skill", skill_dir)

    def test_missing_file_raises_naming_skill_md(self, ctx, tmp_path):
        with pytest.raises(FileNotFoundError, match="SKILL.md not found"):
            ctx.register_skill("my-skill", tmp_path / "gone" / "SKILL.md")


class TestScanLoudSkip:
    def _scan(self, manager, monkeypatch, tmp_path):
        from agent import skill_commands

        monkeypatch.setattr(skill_commands, "_plugin_skill_commands", {})
        monkeypatch.setattr(skill_commands, "_plugin_skill_commands_tag", None)
        metadata = manager.list_plugin_skill_metadata()
        return skill_commands._scan_plugin_skill_commands(manager, metadata, ("probe",))

    def test_missing_file_skips_log_one_warning_each(self, ctx, tmp_path, monkeypatch, caplog):
        for name in ("gone", "vanished"):
            d = tmp_path / name
            d.mkdir()
            (d / "SKILL.md").write_text("test")
            ctx.register_skill(name, d / "SKILL.md")
        # Stale in-memory catalog: files disappear after registration.
        (tmp_path / "gone" / "SKILL.md").unlink()
        (tmp_path / "vanished" / "SKILL.md").unlink()
        # A ghost entry present in the scan metadata but not in the registry.
        metadata = ctx._manager.list_plugin_skill_metadata() + [
            {"name": "testplugin:ghost"},
        ]
        ctx._manager._plugin_skills.pop("testplugin:vanished")

        from agent import skill_commands

        monkeypatch.setattr(skill_commands, "_plugin_skill_commands", {})
        monkeypatch.setattr(skill_commands, "_plugin_skill_commands_tag", None)
        with caplog.at_level(logging.WARNING, logger="agent.skill_commands"):
            commands = skill_commands._scan_plugin_skill_commands(ctx._manager, metadata, ("probe",))

        assert commands == {}
        warnings = [r for r in caplog.records if r.name == "agent.skill_commands"
                    and r.levelno == logging.WARNING]
        assert len(warnings) == 3
        messages = " ".join(r.getMessage() for r in warnings)
        for qualified in ("testplugin:gone", "testplugin:vanished", "testplugin:ghost"):
            assert qualified in messages

    def test_platform_mismatch_skip_logs(self, ctx, tmp_path, monkeypatch, caplog):
        md = tmp_path / "SKILL.md"
        md.write_text('---\nname: p\nplatforms: ["nonexistent-platform-xyz"]\n---\n')
        ctx.register_skill("p", md)

        with caplog.at_level(logging.WARNING, logger="agent.skill_commands"):
            commands = self._scan(ctx._manager, monkeypatch, tmp_path)

        assert commands == {}
        assert any("platform/environment/app mismatch" in r.getMessage()
                   for r in caplog.records if r.name == "agent.skill_commands")
