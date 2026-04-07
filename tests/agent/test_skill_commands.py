"""Tests for agent/skill_commands.py — skill slash command scanning and platform filtering."""

import os
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import tools.skills_tool as skills_tool_module
from agent.skill_commands import (
    build_plan_path,
    build_preloaded_skills_prompt,
    build_skill_invocation_message,
    resolve_skill_command_key,
    scan_skill_commands,
)


def _make_skill(
    skills_dir, name, frontmatter_extra="", body="Do the thing.", category=None
):
    """Helper to create a minimal skill directory with SKILL.md."""
    if category:
        skill_dir = skills_dir / category / name
    else:
        skill_dir = skills_dir / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    content = f"""\
---
name: {name}
description: Description for {name}.
{frontmatter_extra}---

# {name}

{body}
"""
    (skill_dir / "SKILL.md").write_text(content)
    return skill_dir


class TestScanSkillCommands:
    def test_finds_skills(self, tmp_path):
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            _make_skill(tmp_path, "my-skill")
            result = scan_skill_commands()
        assert "/my-skill" in result
        assert result["/my-skill"]["name"] == "my-skill"

    def test_empty_dir(self, tmp_path):
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            result = scan_skill_commands()
        assert result == {}

    def test_excludes_incompatible_platform(self, tmp_path):
        """macOS-only skills should not register slash commands on Linux."""
        with (
            patch("tools.skills_tool.SKILLS_DIR", tmp_path),
            patch("agent.skill_utils.sys") as mock_sys,
        ):
            mock_sys.platform = "linux"
            _make_skill(tmp_path, "imessage", frontmatter_extra="platforms: [macos]\n")
            _make_skill(tmp_path, "web-search")
            result = scan_skill_commands()
        assert "/web-search" in result
        assert "/imessage" not in result

    def test_includes_matching_platform(self, tmp_path):
        """macOS-only skills should register slash commands on macOS."""
        with (
            patch("tools.skills_tool.SKILLS_DIR", tmp_path),
            patch("agent.skill_utils.sys") as mock_sys,
        ):
            mock_sys.platform = "darwin"
            _make_skill(tmp_path, "imessage", frontmatter_extra="platforms: [macos]\n")
            result = scan_skill_commands()
        assert "/imessage" in result

    def test_universal_skill_on_any_platform(self, tmp_path):
        """Skills without platforms field should register on any platform."""
        with (
            patch("tools.skills_tool.SKILLS_DIR", tmp_path),
            patch("agent.skill_utils.sys") as mock_sys,
        ):
            mock_sys.platform = "win32"
            _make_skill(tmp_path, "generic-tool")
            result = scan_skill_commands()
        assert "/generic-tool" in result

    def test_excludes_disabled_skills(self, tmp_path):
        """Disabled skills should not register slash commands."""
        with (
            patch("tools.skills_tool.SKILLS_DIR", tmp_path),
            patch(
                "tools.skills_tool._get_disabled_skill_names",
                return_value={"disabled-skill"},
            ),
        ):
            _make_skill(tmp_path, "enabled-skill")
            _make_skill(tmp_path, "disabled-skill")
            result = scan_skill_commands()
        assert "/enabled-skill" in result
        assert "/disabled-skill" not in result


    def test_special_chars_stripped_from_cmd_key(self, tmp_path):
        """Skill names with +, /, or other special chars produce clean cmd keys."""
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            # Simulate a skill named "Jellyfin + Jellystat 24h Summary"
            skill_dir = tmp_path / "jellyfin-plus"
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text(
                "---\nname: Jellyfin + Jellystat 24h Summary\n"
                "description: Test skill\n---\n\nBody.\n"
            )
            result = scan_skill_commands()
        # The + should be stripped, not left as a literal character
        assert "/jellyfin-jellystat-24h-summary" in result
        # The old buggy key should NOT exist
        assert "/jellyfin-+-jellystat-24h-summary" not in result

    def test_allspecial_name_skipped(self, tmp_path):
        """Skill with name consisting only of special chars is silently skipped."""
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            skill_dir = tmp_path / "bad-name"
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text(
                "---\nname: +++\ndescription: Bad skill\n---\n\nBody.\n"
            )
            result = scan_skill_commands()
        # Should not create a "/" key or any entry
        assert "/" not in result
        assert result == {}

    def test_slash_in_name_stripped_from_cmd_key(self, tmp_path):
        """Skill names with / chars produce clean cmd keys."""
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            skill_dir = tmp_path / "sonarr-api"
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text(
                "---\nname: Sonarr v3/v4 API\n"
                "description: Test skill\n---\n\nBody.\n"
            )
            result = scan_skill_commands()
        assert "/sonarr-v3v4-api" in result
        assert any("/" in k[1:] for k in result) is False  # no unescaped /


class TestResolveSkillCommandKey:
    """Telegram bot-command names disallow hyphens, so the menu registers
    skills with hyphens swapped for underscores. When Telegram autocomplete
    sends the underscored form back, we need to find the hyphenated key.
    """

    def test_hyphenated_form_matches_directly(self, tmp_path):
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            _make_skill(tmp_path, "claude-code")
            scan_skill_commands()
            assert resolve_skill_command_key("claude-code") == "/claude-code"

    def test_underscore_form_resolves_to_hyphenated_skill(self, tmp_path):
        """/claude_code from Telegram autocomplete must resolve to /claude-code."""
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            _make_skill(tmp_path, "claude-code")
            scan_skill_commands()
            assert resolve_skill_command_key("claude_code") == "/claude-code"

    def test_single_word_command_resolves(self, tmp_path):
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            _make_skill(tmp_path, "investigate")
            scan_skill_commands()
            assert resolve_skill_command_key("investigate") == "/investigate"

    def test_unknown_command_returns_none(self, tmp_path):
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            _make_skill(tmp_path, "claude-code")
            scan_skill_commands()
            assert resolve_skill_command_key("does_not_exist") is None
            assert resolve_skill_command_key("does-not-exist") is None

    def test_empty_command_returns_none(self, tmp_path):
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            scan_skill_commands()
            assert resolve_skill_command_key("") is None

    def test_hyphenated_command_is_not_mangled(self, tmp_path):
        """A user-typed /foo-bar (hyphen) must not trigger the underscore fallback."""
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            _make_skill(tmp_path, "foo-bar")
            scan_skill_commands()
            assert resolve_skill_command_key("foo-bar") == "/foo-bar"
            # Underscore form also works (Telegram round-trip)
            assert resolve_skill_command_key("foo_bar") == "/foo-bar"


class TestBuildPreloadedSkillsPrompt:
    def test_builds_prompt_for_multiple_named_skills(self, tmp_path):
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            _make_skill(tmp_path, "first-skill")
            _make_skill(tmp_path, "second-skill")
            prompt, loaded, missing = build_preloaded_skills_prompt(
                ["first-skill", "second-skill"]
            )

        assert missing == []
        assert loaded == ["first-skill", "second-skill"]
        assert "first-skill" in prompt
        assert "second-skill" in prompt
        assert "preloaded" in prompt.lower()

    def test_reports_missing_named_skills(self, tmp_path):
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            _make_skill(tmp_path, "present-skill")
            prompt, loaded, missing = build_preloaded_skills_prompt(
                ["present-skill", "missing-skill"]
            )

        assert "present-skill" in prompt
        assert loaded == ["present-skill"]
        assert missing == ["missing-skill"]


class TestBuildSkillInvocationMessage:
    def test_loads_skill_by_stored_path_when_frontmatter_name_differs(self, tmp_path):
        skill_dir = tmp_path / "mlops" / "audiocraft"
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "SKILL.md").write_text(
            """\
---
name: audiocraft-audio-generation
description: Generate audio with AudioCraft.
---

# AudioCraft

Generate some audio.
"""
        )

        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            scan_skill_commands()
            msg = build_skill_invocation_message("/audiocraft-audio-generation", "compose")

        assert msg is not None
        assert "AudioCraft" in msg
        assert "compose" in msg

    def test_builds_message(self, tmp_path):
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            _make_skill(tmp_path, "test-skill")
            scan_skill_commands()
            msg = build_skill_invocation_message("/test-skill", "do stuff")
        assert msg is not None
        assert "test-skill" in msg
        assert "do stuff" in msg

    def test_returns_none_for_unknown(self, tmp_path):
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            scan_skill_commands()
            msg = build_skill_invocation_message("/nonexistent")
        assert msg is None

    def test_uses_shared_skill_loader_for_secure_setup(self, tmp_path, monkeypatch):
        monkeypatch.delenv("TENOR_API_KEY", raising=False)
        calls = []

        def fake_secret_callback(var_name, prompt, metadata=None):
            calls.append((var_name, prompt, metadata))
            os.environ[var_name] = "stored-in-test"
            return {
                "success": True,
                "stored_as": var_name,
                "validated": False,
                "skipped": False,
            }

        monkeypatch.setattr(
            skills_tool_module,
            "_secret_capture_callback",
            fake_secret_callback,
            raising=False,
        )

        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            _make_skill(
                tmp_path,
                "test-skill",
                frontmatter_extra=(
                    "required_environment_variables:\n"
                    "  - name: TENOR_API_KEY\n"
                    "    prompt: Tenor API key\n"
                ),
            )
            scan_skill_commands()
            msg = build_skill_invocation_message("/test-skill", "do stuff")

        assert msg is not None
        assert "test-skill" in msg
        assert len(calls) == 1
        assert calls[0][0] == "TENOR_API_KEY"

    def test_gateway_still_loads_skill_but_returns_setup_guidance(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.delenv("TENOR_API_KEY", raising=False)

        def fail_if_called(var_name, prompt, metadata=None):
            raise AssertionError(
                "gateway flow should not try secure in-band secret capture"
            )

        monkeypatch.setattr(
            skills_tool_module,
            "_secret_capture_callback",
            fail_if_called,
            raising=False,
        )

        with patch.dict(
            os.environ, {"HERMES_SESSION_PLATFORM": "telegram"}, clear=False
        ):
            with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
                _make_skill(
                    tmp_path,
                    "test-skill",
                    frontmatter_extra=(
                        "required_environment_variables:\n"
                        "  - name: TENOR_API_KEY\n"
                        "    prompt: Tenor API key\n"
                    ),
                )
                scan_skill_commands()
                msg = build_skill_invocation_message("/test-skill", "do stuff")

        assert msg is not None
        assert "local cli" in msg.lower()

    def test_preserves_remaining_remote_setup_warning(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TERMINAL_ENV", "ssh")
        monkeypatch.delenv("TENOR_API_KEY", raising=False)
        monkeypatch.setattr(
            skills_tool_module,
            "_secret_capture_callback",
            None,
            raising=False,
        )

        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            _make_skill(
                tmp_path,
                "test-skill",
                frontmatter_extra=(
                    "required_environment_variables:\n"
                    "  - name: TENOR_API_KEY\n"
                    "    prompt: Tenor API key\n"
                ),
            )
            scan_skill_commands()
            msg = build_skill_invocation_message("/test-skill", "do stuff")

        assert msg is not None
        assert "remote environment" in msg.lower()

    def test_supporting_file_hint_uses_file_path_argument(self, tmp_path):
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            skill_dir = _make_skill(tmp_path, "test-skill")
            references = skill_dir / "references"
            references.mkdir()
            (references / "api.md").write_text("reference")
            scan_skill_commands()
            msg = build_skill_invocation_message("/test-skill", "do stuff")

        assert msg is not None
        assert 'file_path="<path>"' in msg


class TestPlanSkillHelpers:
    def test_build_plan_path_uses_workspace_relative_dir_and_slugifies_request(self):
        path = build_plan_path(
            "Implement OAuth login + refresh tokens!",
            now=datetime(2026, 3, 15, 9, 30, 45),
        )

        assert path == Path(".hermes") / "plans" / "2026-03-15_093045-implement-oauth-login-refresh-tokens.md"

    def test_plan_skill_message_can_include_runtime_save_path_note(self, tmp_path):
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            _make_skill(
                tmp_path,
                "plan",
                body="Save plans under .hermes/plans in the active workspace and do not execute the work.",
            )
            scan_skill_commands()
            msg = build_skill_invocation_message(
                "/plan",
                "Add a /plan command",
                runtime_note=(
                    "Save the markdown plan with write_file to this exact relative path inside "
                    "the active workspace/backend cwd: .hermes/plans/plan.md"
                ),
            )

        assert msg is not None
        assert "Save plans under $HERMES_HOME/plans" not in msg
        assert ".hermes/plans" in msg
        assert "Add a /plan command" in msg
        assert ".hermes/plans/plan.md" in msg
        assert "Runtime note:" in msg
