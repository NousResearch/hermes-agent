"""Tests for agent/skill_commands.py — skill slash command scanning and platform filtering."""

import dis
import os
import sys
import threading
from pathlib import Path
from unittest.mock import patch

import pytest

import tools.skills_tool as skills_tool_module
from agent.skill_commands import (
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

    def test_get_skill_commands_rescans_when_profile_home_changes(self, tmp_path):
        """Switching profiles must rescan even when the platform is unchanged
        (#88023): a Desktop session that switches profiles mid-session keeps
        the same platform scope, so only ``HERMES_HOME`` moves. Each profile
        declares its own ``skills.external_dirs``, and the previous profile's
        skill list must not leak into the new one.
        """
        import agent.skill_commands as sc_mod
        from agent.skill_commands import get_skill_commands
        from hermes_constants import reset_hermes_home_override, set_hermes_home_override

        empty_local_dir = tmp_path / "no-local-skills"
        empty_local_dir.mkdir()

        profile_a = tmp_path / "profile_a"
        profile_b = tmp_path / "profile_b"
        external_a = tmp_path / "external_a"
        external_b = tmp_path / "external_b"
        profile_a.mkdir()
        profile_b.mkdir()
        _make_skill(external_a, "a-only")
        _make_skill(external_b, "b-only")
        (profile_a / "config.yaml").write_text(
            f"skills:\n  external_dirs:\n    - {external_a}\n"
        )
        (profile_b / "config.yaml").write_text(
            f"skills:\n  external_dirs:\n    - {external_b}\n"
        )

        with (
            patch("tools.skills_tool.SKILLS_DIR", empty_local_dir),
            patch.object(sc_mod, "_skill_commands", {}),
            patch.object(sc_mod, "_skill_commands_platform", None),
            patch.object(sc_mod, "_skill_commands_home", None),
        ):
            token = set_hermes_home_override(profile_a)
            try:
                profile_a_commands = dict(get_skill_commands())
            finally:
                reset_hermes_home_override(token)

            assert "/a-only" in profile_a_commands
            assert "/b-only" not in profile_a_commands

            # Switching profiles without touching the cache directly must
            # rescan — not keep serving profile_a's stale view.
            token = set_hermes_home_override(profile_b)
            try:
                profile_b_commands = dict(get_skill_commands())
            finally:
                reset_hermes_home_override(token)

            assert "/b-only" in profile_b_commands
            assert "/a-only" not in profile_b_commands

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






    # -- core-command collision guard (#31204 / #53450) ---------------------




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

    def test_public_discovery_returns_mutable_copies_not_frozen_snapshot(
        self, tmp_path, monkeypatch
    ):
        """Keep the historical dict API without exposing authoritative state."""
        import agent.skill_commands as sc_mod
        from hermes_constants import hermes_home_key

        monkeypatch.setattr(sc_mod, "_skill_command_snapshots", {})
        monkeypatch.setattr(sc_mod, "_skill_commands", {})
        monkeypatch.setattr("tools.skills_tool.SKILLS_DIR", tmp_path)
        monkeypatch.setattr("agent.skill_utils.get_project_skills_dirs", lambda: [])
        monkeypatch.setattr("agent.skill_utils.get_external_skills_dirs", lambda: [])
        _make_skill(tmp_path, "mutable-public")

        public_scan = sc_mod.scan_skill_commands()
        public_scan["/injected"] = {"name": "injected"}
        public_scan["/mutable-public"]["name"] = "mutated"

        home_key = hermes_home_key(sc_mod._resolve_skill_commands_home())
        platform = sc_mod._resolve_skill_commands_platform()
        cached = sc_mod.get_cached_skill_commands(home_key, platform)
        assert cached is not None
        assert "/injected" not in cached
        assert cached["/mutable-public"]["name"] == "mutable-public"

        public_get = sc_mod.get_skill_commands()
        assert isinstance(public_get, dict)
        assert isinstance(public_get["/mutable-public"], dict)
        assert public_get["/mutable-public"]["name"] == "mutable-public"
        public_get.pop("/mutable-public")
        assert "/mutable-public" in cached


class TestResolveCachedSkillCommandKey:
    def test_rejects_cache_from_another_profile(self, monkeypatch):
        import agent.skill_commands as sc_mod

        assert sc_mod.resolve_cached_skill_command_key(
            "only-a", "/profiles/b", "telegram"
        ) is None

    def test_prepared_identity_lookup_is_filesystem_free(self, monkeypatch):
        import agent.skill_commands as sc_mod
        from hermes_constants import hermes_home_key

        home = Path("/profiles") / "telegram" / ".." / "telegram"
        home_key = hermes_home_key(home)
        monkeypatch.setattr(
            sc_mod,
            "_skill_command_snapshots",
            {
                (home_key, "telegram"): {
                    "/only-a": {"name": "only-a"},
                }
            },
            raising=False,
        )
        original_exists = Path.exists
        original_read_text = Path.read_text
        monkeypatch.setattr(
            Path,
            "exists",
            lambda _path: (_ for _ in ()).throw(
                AssertionError("cache-only lookup touched the filesystem")
            ),
        )
        monkeypatch.setattr(
            Path,
            "read_text",
            lambda _path, *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("cache-only lookup read the filesystem")
            ),
        )
        try:
            assert sc_mod.resolve_cached_skill_command_key(
                "only-a", home_key, " TELEGRAM "
            ) == "/only-a"
        finally:
            monkeypatch.setattr(Path, "exists", original_exists)
            monkeypatch.setattr(Path, "read_text", original_read_text)

    def test_scan_and_lookup_share_canonical_home_and_platform_identity(
        self, tmp_path, monkeypatch
    ):
        import agent.skill_commands as sc_mod
        from hermes_constants import hermes_home_key

        skills_dir = tmp_path / "skills"
        profile_home = tmp_path / "profile"
        (profile_home / "nested").mkdir(parents=True)
        _make_skill(skills_dir, "canonical-skill")
        equivalent_home = profile_home / "nested" / ".."

        monkeypatch.setattr(sc_mod, "_skill_command_snapshots", {})
        monkeypatch.setattr(sc_mod, "_skill_commands", {})
        monkeypatch.setattr(
            sc_mod, "_resolve_skill_commands_home", lambda: str(equivalent_home)
        )
        monkeypatch.setattr(
            sc_mod, "_resolve_skill_commands_platform", lambda: " Telegram "
        )
        monkeypatch.setattr("tools.skills_tool.SKILLS_DIR", skills_dir)
        monkeypatch.setattr("agent.skill_utils.get_project_skills_dirs", lambda: [])
        monkeypatch.setattr("agent.skill_utils.get_external_skills_dirs", lambda: [])

        sc_mod.scan_skill_commands()

        assert sc_mod.resolve_cached_skill_command_key(
            "canonical-skill", hermes_home_key(profile_home), "telegram"
        ) == "/canonical-skill"

    def test_interleaved_scans_publish_complete_profile_snapshots(self, tmp_path, monkeypatch):
        import agent.skill_commands as sc_mod

        first_ready = threading.Event()
        allow_first = threading.Event()
        identity = threading.local()
        skill_a = tmp_path / "a" / "SKILL.md"
        skill_b = tmp_path / "b" / "SKILL.md"
        skill_a.parent.mkdir()
        skill_b.parent.mkdir()
        skill_a.write_text("skill-a")
        skill_b.write_text("skill-b")

        def iter_files(_scan_dir, _filename):
            if identity.name == "a":
                first_ready.set()
                assert allow_first.wait(2)
                return [skill_a]
            return [skill_b]

        def parse_frontmatter(_content):
            name = "skill-a" if identity.name == "a" else "skill-b"
            return {"name": name, "description": name}, "body"

        monkeypatch.setattr(sc_mod, "_skill_command_snapshots", {}, raising=False)
        monkeypatch.setattr(sc_mod, "_skill_commands", {})
        monkeypatch.setattr(sc_mod, "_skill_commands_home", None)
        monkeypatch.setattr(sc_mod, "_skill_commands_platform", None)
        monkeypatch.setattr(sc_mod, "_resolve_skill_commands_home", lambda: "/profiles/" + identity.name)
        monkeypatch.setattr(sc_mod, "_resolve_skill_commands_platform", lambda: "telegram")
        monkeypatch.setattr("agent.skill_utils.get_project_skills_dirs", lambda: [])
        monkeypatch.setattr("agent.skill_utils.get_external_skills_dirs", lambda: [])
        monkeypatch.setattr("agent.skill_utils.iter_skill_index_files", iter_files)
        monkeypatch.setattr("tools.skills_tool.SKILLS_DIR", tmp_path)
        monkeypatch.setattr("tools.skills_tool._parse_frontmatter", parse_frontmatter)
        monkeypatch.setattr("tools.skills_tool.skill_matches_platform", lambda _fm: True)
        monkeypatch.setattr("tools.skills_tool.skill_matches_environment", lambda _fm: True)
        monkeypatch.setattr("tools.skills_tool._get_disabled_skill_names", lambda: set())
        monkeypatch.setattr("hermes_cli.commands.resolve_command", lambda _name: None)

        failures = []

        def run(name):
            identity.name = name
            try:
                sc_mod.scan_skill_commands()
            except BaseException as exc:  # pragma: no cover - surfaced below
                failures.append(exc)

        thread_a = threading.Thread(target=run, args=("a",))
        thread_b = threading.Thread(target=run, args=("b",))
        thread_a.start()
        assert first_ready.wait(2)
        thread_b.start()
        thread_b.join(2)
        allow_first.set()
        thread_a.join(2)

        assert not failures
        assert not thread_a.is_alive() and not thread_b.is_alive()
        assert set(sc_mod._skill_commands) in ({"/skill-a"}, {"/skill-b"})
        assert not ({"/skill-a", "/skill-b"} <= set(sc_mod._skill_commands))

    def test_get_returns_its_selected_snapshot_when_compat_alias_changes(
        self, monkeypatch
    ):
        """A concurrent caller must not replace the mapping selected for this call.

        The module-level alias is retained for legacy observability, but it is
        process-global.  Pause caller A on its final return line, let caller B
        publish its own profile view, then prove A still returns profile A's
        local snapshot rather than re-reading B's alias.
        """
        import agent.skill_commands as sc_mod
        from hermes_constants import hermes_home_key

        identity = threading.local()
        home_a = hermes_home_key("/profiles/a")
        home_b = hermes_home_key("/profiles/b")
        commands_a = {"/skill-a": {"name": "skill-a"}}
        commands_b = {"/skill-b": {"name": "skill-b"}}
        monkeypatch.setattr(
            sc_mod,
            "_skill_command_snapshots",
            {
                (home_a, "telegram"): commands_a,
                (home_b, "telegram"): commands_b,
            },
        )
        monkeypatch.setattr(sc_mod, "_skill_commands", {"/seed": {}})
        monkeypatch.setattr(
            sc_mod,
            "_resolve_skill_commands_home",
            lambda: f"/profiles/{identity.name}",
        )
        monkeypatch.setattr(
            sc_mod, "_resolve_skill_commands_platform", lambda: "telegram"
        )

        final_line = max(line for _, line in dis.findlinestarts(sc_mod.get_skill_commands.__code__))
        a_at_return = threading.Event()
        release_a = threading.Event()
        blocked = False
        failures = []
        results = {}

        def trace(frame, event, _arg):
            nonlocal blocked
            if (
                not blocked
                and frame.f_code is sc_mod.get_skill_commands.__code__
                and event == "line"
                and frame.f_lineno == final_line
            ):
                blocked = True
                a_at_return.set()
                release_a.wait(2)
            return trace

        def run_a():
            identity.name = "a"
            sys.settrace(trace)
            try:
                results["a"] = sc_mod.get_skill_commands()
            except BaseException as exc:  # pragma: no cover - surfaced below
                failures.append(exc)
            finally:
                sys.settrace(None)

        thread_a = threading.Thread(target=run_a)
        thread_a.start()
        assert a_at_return.wait(2)

        identity.name = "b"
        results["b"] = sc_mod.get_skill_commands()
        release_a.set()
        thread_a.join(2)

        assert not failures
        assert not thread_a.is_alive()
        assert set(results["a"]) == {"/skill-a"}
        assert set(results["b"]) == {"/skill-b"}

    def test_rejects_cache_from_another_platform(self, monkeypatch):
        import agent.skill_commands as sc_mod

        assert sc_mod.resolve_cached_skill_command_key(
            "only-a", "/profiles/a", "telegram"
        ) is None

    def test_telegram_and_discord_use_distinct_cache_identities(self, monkeypatch):
        import agent.skill_commands as sc_mod

        monkeypatch.setattr(
            sc_mod,
            "_skill_command_snapshots",
            {
                ("/profiles/a", "telegram"): {
                    "/only-telegram": {"name": "only-telegram"},
                }
            },
            raising=False,
        )
        assert sc_mod.resolve_cached_skill_command_key(
            "only-telegram", "/profiles/a", " Telegram "
        ) == "/only-telegram"
        assert sc_mod.resolve_cached_skill_command_key(
            "only-telegram", "/profiles/a", "discord"
        ) is None


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



    def test_unknown_command_returns_none(self, tmp_path):
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            _make_skill(tmp_path, "claude-code")
            scan_skill_commands()
            assert resolve_skill_command_key("does_not_exist") is None
            assert resolve_skill_command_key("does-not-exist") is None




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

    def test_forwards_task_id_to_skill_usage(self, tmp_path):
        with (
            patch("tools.skills_tool.SKILLS_DIR", tmp_path),
            patch("tools.skill_usage.bump_use") as bump_use,
        ):
            _make_skill(tmp_path, "preloaded-skill")
            _prompt, loaded, missing = build_preloaded_skills_prompt(
                ["preloaded-skill"],
                task_id="task-preloaded",
            )

        assert loaded == ["preloaded-skill"]
        assert missing == []
        bump_use.assert_called_once_with(
            "preloaded-skill",
            task_id="task-preloaded",
        )


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



class TestBuildSkillInvocationMessage:



    def test_forwards_task_id_to_skill_usage(self, tmp_path):
        with (
            patch("tools.skills_tool.SKILLS_DIR", tmp_path),
            patch("tools.skill_usage.bump_use") as bump_use,
        ):
            _make_skill(tmp_path, "test-skill")
            scan_skill_commands()
            msg = build_skill_invocation_message(
                "/test-skill",
                task_id="task-slash",
            )

        assert msg is not None
        bump_use.assert_called_once_with("test-skill", task_id="task-slash")


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



    def test_stacked_message_forwards_task_id_to_each_skill(self, tmp_path):
        from agent.skill_commands import build_stacked_skill_invocation_message

        with (
            patch("tools.skills_tool.SKILLS_DIR", tmp_path),
            patch("tools.skill_usage.bump_use") as bump_use,
        ):
            self._setup_three_skills(tmp_path)
            scan_skill_commands()
            result = build_stacked_skill_invocation_message(
                ["/skill-a", "/skill-b"],
                task_id="task-stacked",
            )

        assert result is not None
        assert [call.args[0] for call in bump_use.call_args_list] == [
            "skill-a",
            "skill-b",
        ]
        assert all(
            call.kwargs == {"task_id": "task-stacked"}
            for call in bump_use.call_args_list
        )

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

    def test_explicit_snapshot_drives_stack_parse_and_build_without_ambient_cache(
        self, tmp_path
    ):
        from agent.skill_commands import (
            build_stacked_skill_invocation_message,
            split_stacked_skill_commands,
        )

        first_dir = _make_skill(tmp_path, "first", body="Explicit first body.")
        second_dir = _make_skill(tmp_path, "second", body="Explicit second body.")
        commands = {
            "/first": {"name": "first", "skill_dir": str(first_dir)},
            "/second": {"name": "second", "skill_dir": str(second_dir)},
        }

        with (
            patch("tools.skills_tool.SKILLS_DIR", tmp_path),
            patch(
                "agent.skill_commands.get_skill_commands",
                side_effect=AssertionError("stacked path touched ambient cache"),
            ),
        ):
            extra_keys, instruction = split_stacked_skill_commands(
                "/second run it",
                commands=commands,
            )
            result = build_stacked_skill_invocation_message(
                ["/first", *extra_keys],
                instruction,
                commands=commands,
            )

        assert extra_keys == ["/second"]
        assert instruction == "run it"
        assert result is not None
        message, loaded, missing = result
        assert loaded == ["first", "second"]
        assert missing == []
        assert "Explicit first body." in message
        assert "Explicit second body." in message
