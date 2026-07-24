"""Regression: symlinked profile skills dir must not drop skills from the menu.

Local patch ``gateway-skill-menu-symlink-resolve`` (hermes tooling/patches,
issue NousResearch/hermes-agent#8108). Extends the #8110 external-dir fix.

``_collect_gateway_skill_entries`` builds its allowed prefixes from
``SKILLS_DIR.resolve()`` (symlinks followed) but ``get_skill_commands()`` stores
``skill_md_path`` unresolved. When a profile's skills dir is a symlink to a
shared source (``~/.hermes/profiles/<p>/skills -> vault``), the raw
``startswith()`` check fails for every skill and the whole skill tier is
silently dropped from the Telegram/Discord menu — while ``/skill-name`` dispatch
still works. The fix normalizes ``skill_path`` with ``os.path.realpath`` so both
sides live in the same symlink-resolved namespace.
"""

from unittest.mock import patch

import pytest

from hermes_cli.commands import telegram_menu_commands


def _symlink_dir_or_skip(link, target):
    """Create a directory symlink, skipping on platforms without symlink
    support (e.g. Windows without Developer Mode). Mirrors the
    ``_symlink_file_or_skip`` helper in test_profile_distribution.py."""
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable in test environment: {exc}")


def _write_min_config(tmp_path):
    """HERMES_HOME config with no disabled skills, so the platform-disabled
    filter does not interfere with what we are actually asserting."""
    (tmp_path / "config.yaml").write_text("skills:\n  external_dirs: []\n")


class TestSymlinkedSkillsDirMenu:
    def test_skill_under_symlinked_skills_dir_appears_in_menu(self, tmp_path, monkeypatch):
        # Real source the symlink points at, plus the symlinked skills dir
        # (mirrors ~/.hermes/profiles/<p>/skills -> vault).
        real_src = tmp_path / "vault-skills"
        real_src.mkdir()
        link_dir = tmp_path / "skills"  # the symlink == patched SKILLS_DIR
        _symlink_dir_or_skip(link_dir, real_src)

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        _write_min_config(tmp_path)

        # skill_md_path stored UNRESOLVED (via the symlink), as get_skill_commands does.
        fake_cmds = {
            "/frontend-design": {
                "name": "frontend-design",
                "description": "Distinctive frontend UI",
                "skill_md_path": f"{link_dir}/creative/frontend-design/SKILL.md",
                "skill_dir": f"{link_dir}/creative/frontend-design",
            },
        }

        with (
            patch("agent.skill_commands.get_skill_commands", return_value=fake_cmds),
            patch("tools.skills_tool.SKILLS_DIR", link_dir),
            patch("agent.skill_utils.get_external_skills_dirs", return_value=[]),
        ):
            menu, _ = telegram_menu_commands(max_commands=100)

        assert "frontend_design" in {n for n, _ in menu}, (
            "skill under a symlinked SKILLS_DIR must appear in the menu "
            "(realpath normalization of skill_md_path)"
        )

    def test_hub_skill_under_symlinked_dir_still_excluded(self, tmp_path, monkeypatch):
        """The .hub exclusion must survive realpath normalization."""
        real_src = tmp_path / "vault-skills"
        real_src.mkdir()
        link_dir = tmp_path / "skills"
        _symlink_dir_or_skip(link_dir, real_src)

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        _write_min_config(tmp_path)

        fake_cmds = {
            "/normal-skill": {
                "name": "normal-skill",
                "description": "Regular skill",
                "skill_md_path": f"{link_dir}/creative/normal-skill/SKILL.md",
                "skill_dir": f"{link_dir}/creative/normal-skill",
            },
            "/hub-skill": {
                "name": "hub-skill",
                "description": "Hub-installed, lives under .hub",
                "skill_md_path": f"{link_dir}/.hub/hub-skill/SKILL.md",
                "skill_dir": f"{link_dir}/.hub/hub-skill",
            },
        }

        with (
            patch("agent.skill_commands.get_skill_commands", return_value=fake_cmds),
            patch("tools.skills_tool.SKILLS_DIR", link_dir),
            patch("agent.skill_utils.get_external_skills_dirs", return_value=[]),
        ):
            menu, _ = telegram_menu_commands(max_commands=100)

        names = {n for n, _ in menu}
        assert "normal_skill" in names, "non-hub skill must appear"
        assert "hub_skill" not in names, ".hub skill must stay excluded after realpath"
