"""Tests for agent/skill_commands.py — skill slash command scanning and platform filtering."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

import tools.skills_tool as skills_tool_module
from agent.skill_commands import (
    build_preloaded_skills_prompt,
    build_skill_invocation_message,
    find_triggered_skill_command,
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


def _symlink_category(skills_dir: Path, linked_root: Path, category: str) -> Path:
    """Create a category symlink under skills_dir pointing outside the tree."""
    external_category = linked_root / category
    external_category.mkdir(parents=True, exist_ok=True)
    symlink_path = skills_dir / category
    try:
        symlink_path.symlink_to(external_category, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlinks unavailable in test environment: {exc}")
    return external_category


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

    def test_reads_frontmatter_triggers(self, tmp_path):
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            _make_skill(
                tmp_path,
                "market-watch",
                frontmatter_extra=(
                    "metadata:\n"
                    "  hermes:\n"
                    "    triggers: [market, EUR/USD, market]\n"
                ),
            )
            result = scan_skill_commands()

        assert result["/market-watch"]["triggers"] == ["market", "EUR/USD"]

    def test_finds_triggered_skill_by_word_boundary(self, tmp_path):
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            _make_skill(
                tmp_path,
                "market-watch",
                frontmatter_extra=(
                    "metadata:\n"
                    "  hermes:\n"
                    "    triggers: [market]\n"
                ),
            )
            scan_skill_commands()
            matched = find_triggered_skill_command("What is the market doing today?")
            not_matched = find_triggered_skill_command("This supermarket is busy.")

        assert matched is not None
        assert matched[0] == "/market-watch"
        assert matched[2] == "market"
        assert not_matched is None

    def test_finds_triggered_skill_with_punctuation_phrase(self, tmp_path):
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            _make_skill(
                tmp_path,
                "fx-watch",
                frontmatter_extra=(
                    "metadata:\n"
                    "  hermes:\n"
                    "    triggers: [EUR/USD]\n"
                ),
            )
            scan_skill_commands()
            matched = find_triggered_skill_command("What is happening with EUR/USD today?")

        assert matched is not None
        assert matched[0] == "/fx-watch"
        assert matched[2] == "EUR/USD"

    def test_finds_skills_in_symlinked_category_dir(self, tmp_path):
        external_root = tmp_path / "repo"
        skills_root = tmp_path / "skills"
        skills_root.mkdir()

        external_category = _symlink_category(skills_root, external_root, "linked")
        _make_skill(external_category.parent, "knowledge-brain", category="linked")

        with patch("tools.skills_tool.SKILLS_DIR", skills_root):
            result = scan_skill_commands()

        assert "/knowledge-brain" in result
        assert result["/knowledge-brain"]["name"] == "knowledge-brain"

    def test_loads_skill_invocation_from_symlinked_skill_dir(self, tmp_path):
        """Slash commands should load skills symlinked under the local skills dir."""
        external_root = tmp_path / "external"
        skills_root = tmp_path / "skills"
        skills_root.mkdir()
        real_skill_dir = _make_skill(
            external_root,
            "impeccable",
            body="Apply impeccable design craft.",
        )
        symlink_path = skills_root / "impeccable"
        try:
            symlink_path.symlink_to(real_skill_dir, target_is_directory=True)
        except (OSError, NotImplementedError) as exc:
            pytest.skip(f"symlinks unavailable in test environment: {exc}")

        with patch("tools.skills_tool.SKILLS_DIR", skills_root):
            result = scan_skill_commands()
            message = build_skill_invocation_message("/impeccable")

        assert "/impeccable" in result
        assert message is not None
        assert "Apply impeccable design craft." in message

    def test_get_skill_commands_rescans_when_platform_scope_changes(self, tmp_path):
        """Platform-specific disabled-skill caches must not leak across platforms.

        Regression test for #14536: a gateway process serving Telegram
        and Discord concurrently would seed the process-global cache
        with whichever platform scanned first, and subsequent
        ``get_skill_commands()`` calls from the other platform silently
        inherited that filter.
        """
        import agent.skill_commands as sc_mod
        from agent.skill_commands import get_skill_commands

        def _disabled_skills():
            platform = os.getenv("HERMES_PLATFORM")
            if platform == "telegram":
                return {"telegram-only"}
            if platform == "discord":
                return {"discord-only"}
            return set()

        with (
            patch("tools.skills_tool.SKILLS_DIR", tmp_path),
            patch("tools.skills_tool._get_disabled_skill_names", side_effect=_disabled_skills),
            patch.object(sc_mod, "_skill_commands", {}),
            patch.object(sc_mod, "_skill_commands_platform", None),
        ):
            _make_skill(tmp_path, "shared")
            _make_skill(tmp_path, "telegram-only")
            _make_skill(tmp_path, "discord-only")

            with patch.dict(os.environ, {"HERMES_PLATFORM": "telegram"}):
                telegram_commands = dict(get_skill_commands())

            assert "/shared" in telegram_commands
            assert "/discord-only" in telegram_commands
            assert "/telegram-only" not in telegram_commands

            with patch.dict(os.environ, {"HERMES_PLATFORM": "discord"}):
                discord_commands = dict(get_skill_commands())

            assert "/shared" in discord_commands
            assert "/telegram-only" in discord_commands
            assert "/discord-only" not in discord_commands

            # Switching back to telegram must also rescan — not re-serve
            # the discord view that was just cached.
            with patch.dict(os.environ, {"HERMES_PLATFORM": "telegram"}):
                telegram_again = dict(get_skill_commands())

            assert "/telegram-only" not in telegram_again
            assert "/discord-only" in telegram_again

    def test_get_skill_commands_rescans_when_session_platform_changes(self, tmp_path):
        """``HERMES_SESSION_PLATFORM`` from the gateway session context must
        also trigger a rescan, not just ``HERMES_PLATFORM`` (#14536).

        Exercises the real ContextVar path: the gateway sets the active
        adapter via ``set_session_vars(platform=...)`` and the resolver
        reads it via ``get_session_env``. Setting ``HERMES_SESSION_PLATFORM``
        in ``os.environ`` would only test ``get_session_env``'s legacy
        env-var fallback — a regression that swapped ``get_session_env``
        for plain ``os.getenv`` would still pass while breaking concurrent
        gateway sessions, which is the bug the ContextVar plumbing exists
        to prevent in the first place.
        """
        import agent.skill_commands as sc_mod
        from agent.skill_commands import get_skill_commands
        from gateway.session_context import (
            clear_session_vars,
            get_session_env,
            set_session_vars,
        )

        def _disabled_skills():
            platform = (
                os.getenv("HERMES_PLATFORM")
                or get_session_env("HERMES_SESSION_PLATFORM")
            )
            if platform == "telegram":
                return {"telegram-only"}
            if platform == "discord":
                return {"discord-only"}
            return set()

        with (
            patch("tools.skills_tool.SKILLS_DIR", tmp_path),
            patch("tools.skills_tool._get_disabled_skill_names", side_effect=_disabled_skills),
            patch.object(sc_mod, "_skill_commands", {}),
            patch.object(sc_mod, "_skill_commands_platform", None),
        ):
            _make_skill(tmp_path, "shared")
            _make_skill(tmp_path, "telegram-only")
            _make_skill(tmp_path, "discord-only")

            # First simulated gateway request: telegram handler.
            tokens = set_session_vars(platform="telegram")
            try:
                telegram_commands = dict(get_skill_commands())
            finally:
                clear_session_vars(tokens)

            assert "/shared" in telegram_commands
            assert "/discord-only" in telegram_commands
            assert "/telegram-only" not in telegram_commands

            # Second simulated gateway request: discord handler. The cache
            # was just populated for telegram; the rescan trigger must fire
            # off the ContextVar change, not just an env-var change.
            tokens = set_session_vars(platform="discord")
            try:
                discord_commands = dict(get_skill_commands())
            finally:
                clear_session_vars(tokens)

            assert "/shared" in discord_commands
            assert "/telegram-only" in discord_commands
            assert "/discord-only" not in discord_commands

    def test_get_skill_commands_rescans_when_leaving_platform_scope(self, tmp_path, monkeypatch):
        """Returning to no-platform-scope (CLI / cron / RL) after a gateway
        session must rescan so the unfiltered view is repopulated (#14536).

        A long-lived process running both gateway sessions and bare CLI
        invocations would otherwise stay stuck on whichever platform's
        filter was last applied.
        """
        import agent.skill_commands as sc_mod
        from agent.skill_commands import get_skill_commands

        def _disabled_skills():
            if os.getenv("HERMES_PLATFORM") == "telegram":
                return {"telegram-only"}
            return set()

        with (
            patch("tools.skills_tool.SKILLS_DIR", tmp_path),
            patch("tools.skills_tool._get_disabled_skill_names", side_effect=_disabled_skills),
            patch.object(sc_mod, "_skill_commands", {}),
            patch.object(sc_mod, "_skill_commands_platform", None),
        ):
            _make_skill(tmp_path, "shared")
            _make_skill(tmp_path, "telegram-only")

            monkeypatch.setenv("HERMES_PLATFORM", "telegram")
            telegram_commands = dict(get_skill_commands())
            assert "/telegram-only" not in telegram_commands

            # Drop back to no platform scope — bare CLI / cron / RL rollouts.
            monkeypatch.delenv("HERMES_PLATFORM", raising=False)
            bare_commands = dict(get_skill_commands())

            assert "/telegram-only" in bare_commands
            assert sc_mod._skill_commands_platform is None

    def test_get_skill_commands_does_not_rescan_when_platform_unchanged(self, tmp_path):
        """Same-platform back-to-back calls must hit the cache, not rescan.

        The rescan trigger is *change* in platform scope, not "always
        re-resolve." A gateway serving consecutive telegram requests must
        not pay the scan cost for each one.
        """
        import agent.skill_commands as sc_mod
        from agent.skill_commands import get_skill_commands

        with (
            patch("tools.skills_tool.SKILLS_DIR", tmp_path),
            patch.object(sc_mod, "_skill_commands", {}),
            patch.object(sc_mod, "_skill_commands_platform", None),
            patch.dict(os.environ, {"HERMES_PLATFORM": "telegram"}),
        ):
            _make_skill(tmp_path, "shared")
            # Prime the cache.
            get_skill_commands()
            # Spy on rescans during the subsequent same-platform calls.
            with patch(
                "agent.skill_commands.scan_skill_commands",
                wraps=sc_mod.scan_skill_commands,
            ) as scan_spy:
                get_skill_commands()
                get_skill_commands()
                get_skill_commands()
            assert scan_spy.call_count == 0


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

    # -- core-command collision guard (#31204 / #53450) ---------------------

    def test_skill_collides_with_core_command_is_skipped(self, tmp_path):
        """A skill whose auto-generated /command collides with a core Hermes
        command (e.g. 'skills') should be excluded from the slash-command map.
        The skill remains loadable via /skill <name>."""
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            _make_skill(tmp_path, "skills")
            result = scan_skill_commands()
        assert "/skills" not in result

    def test_skill_collides_with_core_alias_is_skipped(self, tmp_path):
        """A skill whose slug matches a core command *alias* is also skipped."""
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            _make_skill(tmp_path, "bg")
            result = scan_skill_commands()
        # "bg" is an alias of the "background" command
        assert "/bg" not in result

    def test_core_command_collision_does_not_block_others(self, tmp_path):
        """A colliding skill is skipped, but non-colliding skills in the same
        scan pass are still registered."""
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            _make_skill(tmp_path, "skills")
            _make_skill(tmp_path, "my-other-skill")
            result = scan_skill_commands()
        assert "/skills" not in result
        assert "/my-other-skill" in result

    # -- inter-skill slug collision dedup (#50304 / #63305) ------------------

    def test_slug_collision_keeps_first_skill(self, tmp_path):
        """Two skills whose names normalize to the same slug do not clobber.

        ``git_helper`` and ``git-helper`` are distinct frontmatter names but
        both reduce to the ``/git-helper`` command. The first one scanned must
        keep the command rather than being silently overwritten by the second.
        """
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            # ``a-first`` sorts before ``z-second`` so the index walk visits the
            # underscore-named skill first; that one must win the slash command.
            first = tmp_path / "a-first"
            first.mkdir()
            (first / "SKILL.md").write_text(
                "---\nname: git_helper\ndescription: First skill.\n---\n\nBody.\n"
            )
            second = tmp_path / "z-second"
            second.mkdir()
            (second / "SKILL.md").write_text(
                "---\nname: git-helper\ndescription: Second skill.\n---\n\nBody.\n"
            )
            result = scan_skill_commands()
        assert "/git-helper" in result
        # First-wins: the entry resolves to the first skill, not the shadowing one.
        assert result["/git-helper"]["name"] == "git_helper"
        assert result["/git-helper"]["skill_dir"] == str(first)

    def test_slug_collision_warns(self, tmp_path, caplog):
        """A slug collision emits a warning so the user can diagnose the
        shadowed skill."""
        import logging as _logging
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            first = tmp_path / "a-first"
            first.mkdir()
            (first / "SKILL.md").write_text(
                "---\nname: my-skill\ndescription: First.\n---\n\nBody.\n"
            )
            second = tmp_path / "z-second"
            second.mkdir()
            (second / "SKILL.md").write_text(
                "---\nname: my_skill\ndescription: Second.\n---\n\nBody.\n"
            )
            with caplog.at_level(_logging.WARNING, logger="agent.skill_commands"):
                scan_skill_commands()
        assert any("already claimed" in r.message for r in caplog.records)


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

    def test_skips_disabled_skill(self, tmp_path, monkeypatch):
        """A globally-disabled skill must not be force-loaded via -s /
        HERMES_TUI_SKILLS preloading (mirrors the bundle gate, #59156)."""
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            _make_skill(tmp_path, "enabled-skill", body="Enabled content.")
            _make_skill(tmp_path, "disabled-skill", body="SECRET DISABLED CONTENT.")

            import agent.skill_utils as su_module
            monkeypatch.setattr(
                su_module, "get_disabled_skill_names", lambda platform=None: {"disabled-skill"}
            )

            prompt, loaded, missing = build_preloaded_skills_prompt(
                ["enabled-skill", "disabled-skill"]
            )

        assert loaded == ["enabled-skill"]
        assert missing == ["disabled-skill"]
        assert "SECRET DISABLED CONTENT." not in prompt
        assert "enabled-skill" in prompt

    def test_loads_normally_when_nothing_disabled(self, tmp_path, monkeypatch):
        """Positive control: without a disabled-skills config, both load."""
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            _make_skill(tmp_path, "first-skill")
            _make_skill(tmp_path, "second-skill")

            import agent.skill_utils as su_module
            monkeypatch.setattr(su_module, "get_disabled_skill_names", lambda platform=None: set())

            prompt, loaded, missing = build_preloaded_skills_prompt(
                ["first-skill", "second-skill"]
            )

        assert missing == []
        assert loaded == ["first-skill", "second-skill"]


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

    def test_returns_none_when_skill_load_fails(self, tmp_path):
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            _make_skill(tmp_path, "broken-skill")
            scan_skill_commands()
            with patch("agent.skill_commands._load_skill_payload", return_value=None):
                msg = build_skill_invocation_message("/broken-skill", "do stuff")
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

        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            from gateway.session_context import clear_session_vars, set_session_vars

            tokens = set_session_vars(platform="telegram")
            try:
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
            finally:
                clear_session_vars(tokens)

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


class TestSkillDirectoryHeader:
    """The activation message must expose the absolute skill directory and
    explain how to resolve relative paths, so skills with bundled scripts
    don't force the agent into a second ``skill_view()`` round-trip."""

    def test_header_contains_absolute_skill_dir(self, tmp_path):
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            skill_dir = _make_skill(tmp_path, "abs-dir-skill")
            scan_skill_commands()
            msg = build_skill_invocation_message("/abs-dir-skill", "go")

        assert msg is not None
        assert f"[Skill directory: {skill_dir}]" in msg
        assert "Resolve any relative paths" in msg

    def test_supporting_files_shown_with_absolute_paths(self, tmp_path):
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            skill_dir = _make_skill(tmp_path, "scripted-skill")
            (skill_dir / "scripts").mkdir()
            (skill_dir / "scripts" / "run.js").write_text("console.log('hi')")
            scan_skill_commands()
            msg = build_skill_invocation_message("/scripted-skill")

        assert msg is not None
        # The supporting-files block must emit both the relative form (so the
        # agent can call skill_view on it) and the absolute form (so it can
        # run the script directly via terminal).
        assert "scripts/run.js" in msg
        assert str(skill_dir / "scripts" / "run.js") in msg
        assert f"node {skill_dir}/scripts/foo.js" in msg


class TestTemplateVarSubstitution:
    """``${HERMES_SKILL_DIR}`` and ``${HERMES_SESSION_ID}`` in SKILL.md body
    are replaced before the agent sees the content."""

    def test_substitutes_skill_dir(self, tmp_path):
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            skill_dir = _make_skill(
                tmp_path,
                "templated",
                body="Run: node ${HERMES_SKILL_DIR}/scripts/foo.js",
            )
            scan_skill_commands()
            msg = build_skill_invocation_message("/templated")

        assert msg is not None
        assert f"node {skill_dir}/scripts/foo.js" in msg
        # The literal template token must not leak through.
        assert "${HERMES_SKILL_DIR}" not in msg.split("[Skill directory:")[0]

    def test_substitutes_session_id_when_available(self, tmp_path):
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            _make_skill(
                tmp_path,
                "sess-templated",
                body="Session: ${HERMES_SESSION_ID}",
            )
            scan_skill_commands()
            msg = build_skill_invocation_message(
                "/sess-templated", task_id="abc-123"
            )

        assert msg is not None
        assert "Session: abc-123" in msg

    def test_leaves_session_id_token_when_missing(self, tmp_path):
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            _make_skill(
                tmp_path,
                "sess-missing",
                body="Session: ${HERMES_SESSION_ID}",
            )
            scan_skill_commands()
            msg = build_skill_invocation_message("/sess-missing", task_id=None)

        assert msg is not None
        # No session — token left intact so the author can spot it.
        assert "Session: ${HERMES_SESSION_ID}" in msg

    def test_disable_template_vars_via_config(self, tmp_path):
        with (
            patch("tools.skills_tool.SKILLS_DIR", tmp_path),
            patch(
                "agent.skill_commands._load_skills_config",
                return_value={"template_vars": False},
            ),
        ):
            _make_skill(
                tmp_path,
                "no-sub",
                body="Run: node ${HERMES_SKILL_DIR}/scripts/foo.js",
            )
            scan_skill_commands()
            msg = build_skill_invocation_message("/no-sub")

        assert msg is not None
        # Template token must survive when substitution is disabled.
        assert "${HERMES_SKILL_DIR}/scripts/foo.js" in msg


class TestInlineShellExpansion:
    """Inline ``!`cmd`` snippets in SKILL.md run before the agent sees the
    content — but only when the user has opted in via config."""

    def test_inline_shell_is_off_by_default(self, tmp_path):
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            _make_skill(
                tmp_path,
                "dyn-default-off",
                body="Today is !`echo INLINE_RAN`.",
            )
            scan_skill_commands()
            msg = build_skill_invocation_message("/dyn-default-off")

        assert msg is not None
        # Default config has inline_shell=False — snippet must stay literal.
        assert "!`echo INLINE_RAN`" in msg
        assert "Today is INLINE_RAN." not in msg

    def test_inline_shell_runs_when_enabled(self, tmp_path):
        with (
            patch("tools.skills_tool.SKILLS_DIR", tmp_path),
            patch(
                "agent.skill_commands._load_skills_config",
                return_value={"template_vars": True, "inline_shell": True,
                              "inline_shell_timeout": 5},
            ),
        ):
            _make_skill(
                tmp_path,
                "dyn-on",
                body="Marker: !`echo INLINE_RAN`.",
            )
            scan_skill_commands()
            msg = build_skill_invocation_message("/dyn-on")

        assert msg is not None
        assert "Marker: INLINE_RAN." in msg
        assert "!`echo INLINE_RAN`" not in msg

    def test_inline_shell_runs_in_skill_directory(self, tmp_path):
        """Inline snippets get the skill dir as CWD so relative paths work."""
        with (
            patch("tools.skills_tool.SKILLS_DIR", tmp_path),
            patch(
                "agent.skill_commands._load_skills_config",
                return_value={"template_vars": True, "inline_shell": True,
                              "inline_shell_timeout": 5},
            ),
        ):
            skill_dir = _make_skill(
                tmp_path,
                "dyn-cwd",
                body="Here: !`pwd`",
            )
            scan_skill_commands()
            msg = build_skill_invocation_message("/dyn-cwd")

        assert msg is not None
        assert f"Here: {skill_dir}" in msg

    def test_inline_shell_timeout_does_not_break_message(self, tmp_path):
        with (
            patch("tools.skills_tool.SKILLS_DIR", tmp_path),
            patch(
                "agent.skill_commands._load_skills_config",
                return_value={"template_vars": True, "inline_shell": True,
                              "inline_shell_timeout": 1},
            ),
        ):
            _make_skill(
                tmp_path,
                "dyn-slow",
                body="Slow: !`sleep 5 && printf DYN_MARKER`",
            )
            scan_skill_commands()
            msg = build_skill_invocation_message("/dyn-slow")

        assert msg is not None
        # Timeout is surfaced as a marker instead of propagating as an error,
        # and the rest of the skill message still renders.
        assert "inline-shell timeout" in msg
        # The command's intended stdout never made it through — only the
        # timeout marker (which echoes the command text) survives.
        assert "DYN_MARKER" not in msg.replace("sleep 5 && printf DYN_MARKER", "")


class TestStackedSkillCommands:
    """Stacked slash-skill invocations — inspired by Claude Code v2.1.199."""

    def _setup_three_skills(self, tmp_path):
        _make_skill(tmp_path, "skill-a", body="Body A.")
        _make_skill(tmp_path, "skill-b", body="Body B.")
        _make_skill(tmp_path, "skill-c", body="Body C.")

    def test_split_consumes_leading_skill_tokens(self, tmp_path):
        from agent.skill_commands import split_stacked_skill_commands
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            self._setup_three_skills(tmp_path)
            scan_skill_commands()
            keys, instruction = split_stacked_skill_commands(
                "/skill-b /skill-c do the thing"
            )
        assert keys == ["/skill-b", "/skill-c"]
        assert instruction == "do the thing"

    def test_split_stops_at_non_skill_token(self, tmp_path):
        from agent.skill_commands import split_stacked_skill_commands
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            self._setup_three_skills(tmp_path)
            scan_skill_commands()
            keys, instruction = split_stacked_skill_commands(
                "/skill-b /not-a-skill /skill-c hello"
            )
        assert keys == ["/skill-b"]
        # Parsing stops at the first unresolvable token; everything from
        # there on is the user instruction (slash included).
        assert instruction == "/not-a-skill /skill-c hello"

    def test_split_plain_instruction_passthrough(self, tmp_path):
        from agent.skill_commands import split_stacked_skill_commands
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            self._setup_three_skills(tmp_path)
            scan_skill_commands()
            keys, instruction = split_stacked_skill_commands("just do the thing")
        assert keys == []
        assert instruction == "just do the thing"

    def test_split_underscore_form_resolves(self, tmp_path):
        """Telegram autocomplete sends /skill_b — must resolve like /skill-b."""
        from agent.skill_commands import split_stacked_skill_commands
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            self._setup_three_skills(tmp_path)
            scan_skill_commands()
            keys, instruction = split_stacked_skill_commands("/skill_b go")
        assert keys == ["/skill-b"]
        assert instruction == "go"

    def test_split_caps_at_five_total(self, tmp_path):
        from agent.skill_commands import split_stacked_skill_commands
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            for i in range(7):
                _make_skill(tmp_path, f"stk-{i}")
            scan_skill_commands()
            rest = " ".join(f"/stk-{i}" for i in range(1, 7)) + " run"
            keys, instruction = split_stacked_skill_commands(rest)
        # First skill was already consumed by the caller — split returns at
        # most 4 extras so the total stays at 5.
        assert len(keys) == 4
        assert instruction.startswith("/stk-5")

    def test_split_dedupes_repeated_skill(self, tmp_path):
        from agent.skill_commands import split_stacked_skill_commands
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            self._setup_three_skills(tmp_path)
            scan_skill_commands()
            keys, instruction = split_stacked_skill_commands(
                "/skill-b /skill-b go"
            )
        # The duplicate stops parsing (treated as instruction text).
        assert keys == ["/skill-b"]
        assert instruction == "/skill-b go"

    def test_stacked_message_contains_all_bodies_and_instruction(self, tmp_path):
        from agent.skill_commands import build_stacked_skill_invocation_message
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            self._setup_three_skills(tmp_path)
            scan_skill_commands()
            result = build_stacked_skill_invocation_message(
                ["/skill-a", "/skill-b"], "do the thing"
            )
        assert result is not None
        msg, loaded, missing = result
        assert loaded == ["skill-a", "skill-b"]
        assert missing == []
        assert "Body A." in msg
        assert "Body B." in msg
        assert "User instruction: do the thing" in msg

    def test_stacked_message_skips_missing_skills(self, tmp_path):
        from agent.skill_commands import build_stacked_skill_invocation_message
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            self._setup_three_skills(tmp_path)
            scan_skill_commands()
            result = build_stacked_skill_invocation_message(
                ["/skill-a", "/gone"], "go"
            )
        assert result is not None
        msg, loaded, missing = result
        assert loaded == ["skill-a"]
        assert missing == ["gone"]
        assert "Skills missing (skipped): gone" in msg

    def test_stacked_message_none_when_nothing_loads(self, tmp_path):
        from agent.skill_commands import build_stacked_skill_invocation_message
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            scan_skill_commands()
            result = build_stacked_skill_invocation_message(["/gone"], "go")
        assert result is None

    def test_memory_extractor_recovers_instruction_from_stacked_turn(self, tmp_path):
        """The stacked scaffolding reuses bundle markers so memory providers
        recover the user's instruction, not N skill bodies."""
        from agent.skill_commands import (
            build_stacked_skill_invocation_message,
            extract_user_instruction_from_skill_message,
        )
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            self._setup_three_skills(tmp_path)
            scan_skill_commands()
            result = build_stacked_skill_invocation_message(
                ["/skill-a", "/skill-b"], "summarize the repo"
            )
        assert result is not None
        msg, _, _ = result
        assert extract_user_instruction_from_skill_message(msg) == "summarize the repo"

    def test_memory_extractor_returns_none_for_bare_stacked_turn(self, tmp_path):
        from agent.skill_commands import (
            build_stacked_skill_invocation_message,
            extract_user_instruction_from_skill_message,
        )
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            self._setup_three_skills(tmp_path)
            scan_skill_commands()
            result = build_stacked_skill_invocation_message(
                ["/skill-a", "/skill-b"], ""
            )
        assert result is not None
        msg, _, _ = result
        assert extract_user_instruction_from_skill_message(msg) is None
