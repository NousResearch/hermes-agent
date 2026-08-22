"""Tests for the cross-Hermes-profile write guard in agent/file_safety.

The guard fires when a tool tries to write into another Hermes profile's
skills/plugins/cron/memories directory. It's a soft guard — defense in
depth, NOT a security boundary — but it prevents the agent from silently
corrupting a profile that belongs to a different session.

Reference: May 2026 incident — a hermes-security profile session
accidentally edited skills under both ~/.hermes/profiles/hermes-security/skills/
AND ~/.hermes/skills/ (the default profile's skills), realizing only
afterwards that the second path belonged to a different profile.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers — set up a fake Hermes root with two profiles, monkeypatch the
# resolver helpers so the classifier sees the test layout.
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_hermes(tmp_path, monkeypatch):
    """Build a fake Hermes layout:

        <tmp>/
          skills/foo/SKILL.md           # default profile
          plugins/foo/__init__.py
          cron/<state>
          memories/MEMORY.md
          profiles/
            hermes-security/
              skills/foo/SKILL.md       # named profile
              plugins/...
            coder/
              skills/foo/SKILL.md       # another named profile
    """
    root = tmp_path / "fake-hermes"
    (root / "skills" / "foo").mkdir(parents=True)
    (root / "skills" / "foo" / "SKILL.md").write_text("# default skill\n", encoding="utf-8")
    (root / "plugins" / "foo").mkdir(parents=True)
    (root / "memories").mkdir(parents=True)
    (root / "cron").mkdir(parents=True)

    sec_home = root / "profiles" / "hermes-security"
    (sec_home / "skills" / "foo").mkdir(parents=True)
    (sec_home / "skills" / "foo" / "SKILL.md").write_text("# sec skill\n", encoding="utf-8")
    (sec_home / "plugins").mkdir(parents=True)

    coder_home = root / "profiles" / "coder"
    (coder_home / "skills" / "foo").mkdir(parents=True)
    (coder_home / "skills" / "foo" / "SKILL.md").write_text("# coder skill\n", encoding="utf-8")

    # Monkeypatch the resolver functions used by file_safety so each test
    # can choose which profile is "active".
    import hermes_constants
    monkeypatch.setattr(hermes_constants, "get_default_hermes_root", lambda: root)

    # The reloads below ensure get_cross_profile_warning/classify see the patched root.
    import agent.file_safety as fs
    monkeypatch.setattr(fs, "_hermes_root_path", lambda: root)

    return {
        "root": root,
        "default_home": root,
        "security_home": sec_home,
        "coder_home": coder_home,
    }


def _set_active_home(monkeypatch, hermes_home: Path):
    """Point file_safety._hermes_home_path at a specific profile dir."""
    import agent.file_safety as fs
    monkeypatch.setattr(fs, "_hermes_home_path", lambda: hermes_home)


def _set_strict_scope(monkeypatch, *allowed_paths: Path):
    """Enable strict profile scope without loading the test process config."""
    import agent.file_safety as fs

    monkeypatch.setattr(
        fs,
        "_profile_scope_settings",
        lambda: ("strict", tuple(allowed_paths)),
    )


class _DivergentHostFileOps:
    """Host backend whose cwd deliberately differs from the task workspace."""

    def __init__(self, cwd: Path):
        self.cwd = cwd

    def _sink_path(self, path: str) -> Path:
        candidate = Path(path)
        return candidate if candidate.is_absolute() else self.cwd / candidate

    def read_file(self, path: str, offset: int, limit: int):
        from tools.file_operations import ReadResult

        target = self._sink_path(path)
        content = target.read_text(encoding="utf-8")
        return ReadResult(
            content=f"1|{content.rstrip()}",
            total_lines=1,
            file_size=target.stat().st_size,
        )

    def search(self, pattern: str, path: str, **_kwargs):
        from tools.file_operations import SearchMatch, SearchResult

        root = self._sink_path(path)
        matches = []
        for candidate in root.glob("*"):
            if not candidate.is_file():
                continue
            content = candidate.read_text(encoding="utf-8")
            if pattern in content:
                # A relative backend search result is interpreted against the
                # task cwd by the result filter, which is the second half of
                # the raw-path bypass this fixture models.
                reported_path = (
                    str(candidate) if Path(path).is_absolute() else candidate.name
                )
                matches.append(SearchMatch(reported_path, 1, content.rstrip()))
        return SearchResult(matches=matches, total_count=len(matches))


# ---------------------------------------------------------------------------
# _resolve_active_profile_name
# ---------------------------------------------------------------------------


class TestResolveActiveProfileName:
    def test_default_when_home_is_root(self, fake_hermes, monkeypatch):
        _set_active_home(monkeypatch, fake_hermes["default_home"])
        from agent.file_safety import _resolve_active_profile_name

        assert _resolve_active_profile_name() == "default"

    def test_falls_back_to_default_on_resolution_failure(
        self, fake_hermes, monkeypatch
    ):
        """If HERMES_HOME resolution raises, return 'default' rather than crashing the tool."""
        import agent.file_safety as fs

        def _boom():
            raise RuntimeError("simulated")

        monkeypatch.setattr(fs, "_hermes_home_path", _boom)
        # Should not raise — falls back to "default"
        assert fs._resolve_active_profile_name() == "default"


# ---------------------------------------------------------------------------
# classify_cross_profile_target
# ---------------------------------------------------------------------------


class TestClassifyCrossProfileTarget:
    def test_security_writing_default_skill(self, fake_hermes, monkeypatch):
        """The exact incident from May 2026."""
        _set_active_home(monkeypatch, fake_hermes["security_home"])
        from agent.file_safety import classify_cross_profile_target

        result = classify_cross_profile_target(
            str(fake_hermes["default_home"] / "skills" / "foo" / "SKILL.md")
        )
        assert result is not None
        assert result["active_profile"] == "hermes-security"
        assert result["target_profile"] == "default"
        assert result["area"] == "skills"

    def test_default_writing_security_skill(self, fake_hermes, monkeypatch):
        """Inverse direction — default-profile session reaching into a named profile."""
        _set_active_home(monkeypatch, fake_hermes["default_home"])
        from agent.file_safety import classify_cross_profile_target

        result = classify_cross_profile_target(
            str(fake_hermes["security_home"] / "skills" / "foo" / "SKILL.md")
        )
        assert result is not None
        assert result["active_profile"] == "default"
        assert result["target_profile"] == "hermes-security"

    @pytest.mark.parametrize("area", ["skills", "plugins", "cron", "memories"])
    def test_all_profile_scoped_areas_classified(self, fake_hermes, monkeypatch, area):
        _set_active_home(monkeypatch, fake_hermes["security_home"])
        from agent.file_safety import classify_cross_profile_target

        target = fake_hermes["default_home"] / area / "foo.txt"
        result = classify_cross_profile_target(str(target))
        assert result is not None
        assert result["area"] == area


# ---------------------------------------------------------------------------
# get_cross_profile_warning
# ---------------------------------------------------------------------------


class TestGetCrossProfileWarning:
    def test_in_profile_returns_none(self, fake_hermes, monkeypatch):
        _set_active_home(monkeypatch, fake_hermes["security_home"])
        from agent.file_safety import get_cross_profile_warning

        assert (
            get_cross_profile_warning(
                str(fake_hermes["security_home"] / "skills" / "foo" / "SKILL.md")
            )
            is None
        )

    def test_cross_profile_warning_names_both_profiles(self, fake_hermes, monkeypatch):
        _set_active_home(monkeypatch, fake_hermes["security_home"])
        from agent.file_safety import get_cross_profile_warning

        warn = get_cross_profile_warning(
            str(fake_hermes["default_home"] / "skills" / "foo" / "SKILL.md")
        )
        assert warn is not None
        # Must name BOTH profiles so the model knows which is which.
        assert "default" in warn
        assert "hermes-security" in warn
        # Must name the bypass kwarg.
        assert "cross_profile=True" in warn
        # Must reference the area.
        assert "skills" in warn

    def test_warning_is_defense_in_depth_not_boundary(self, fake_hermes, monkeypatch):
        _set_active_home(monkeypatch, fake_hermes["security_home"])
        from agent.file_safety import get_cross_profile_warning

        warn = get_cross_profile_warning(
            str(fake_hermes["default_home"] / "skills" / "foo" / "SKILL.md")
        )
        # Must self-document as defense-in-depth so future reviewers
        # don't promote it to a hard block.
        assert "not a security boundary" in warn.lower()


# ---------------------------------------------------------------------------
# Strict profile file-tool scope
# ---------------------------------------------------------------------------


class TestStrictProfileScope:
    def test_settings_read_strict_mode_and_absolute_allow_paths(
        self, monkeypatch, tmp_path
    ):
        import agent.file_safety as fs
        import hermes_cli.config as config

        shared_cache = tmp_path / "shared-cache"
        monkeypatch.setattr(
            config,
            "load_config_readonly",
            lambda: {
                "agent": {
                    "profile_scope": "strict",
                    "profile_scope_allow": [str(shared_cache), "relative-path", 42],
                }
            },
        )

        assert fs._profile_scope_settings() == ("strict", (shared_cache.resolve(),))

    def test_named_profile_denies_default_and_sibling_reads_and_writes(
        self, fake_hermes, monkeypatch
    ):
        _set_active_home(monkeypatch, fake_hermes["security_home"])
        _set_strict_scope(monkeypatch)
        from agent.file_safety import get_read_block_error, get_write_denied_error

        default_memory = fake_hermes["default_home"] / "memories" / "MEMORY.md"
        sibling_skill = fake_hermes["coder_home"] / "skills" / "foo" / "SKILL.md"

        for target in (default_memory, sibling_skill):
            read_error = get_read_block_error(str(target))
            write_error = get_write_denied_error(str(target))
            assert read_error is not None
            assert write_error is not None
            assert "agent.profile_scope: strict" in read_error
            assert "agent.profile_scope: strict" in write_error
            assert "hermes-security" in read_error

    def test_named_profile_keeps_own_tree_and_non_hermes_paths(
        self, fake_hermes, monkeypatch, tmp_path
    ):
        _set_active_home(monkeypatch, fake_hermes["security_home"])
        _set_strict_scope(monkeypatch)
        from agent.file_safety import get_read_block_error, get_write_denied_error

        own_file = fake_hermes["security_home"] / "notes.md"
        external_file = tmp_path / "project" / "notes.md"
        assert get_read_block_error(str(own_file)) is None
        assert get_write_denied_error(str(own_file)) is None
        assert get_read_block_error(str(external_file)) is None
        assert get_write_denied_error(str(external_file)) is None

    def test_explicit_shared_root_is_allowed(self, fake_hermes, monkeypatch):
        _set_active_home(monkeypatch, fake_hermes["security_home"])
        shared_cache = fake_hermes["root"] / "cache"
        _set_strict_scope(monkeypatch, shared_cache)
        from agent.file_safety import get_read_block_error, get_write_denied_error

        target = shared_cache / "index.json"
        assert get_read_block_error(str(target)) is None
        assert get_write_denied_error(str(target)) is None

    def test_default_profile_strict_scope_denies_named_profiles(
        self, fake_hermes, monkeypatch
    ):
        _set_active_home(monkeypatch, fake_hermes["default_home"])
        _set_strict_scope(monkeypatch)
        from agent.file_safety import get_read_block_error

        assert (
            get_read_block_error(
                str(fake_hermes["security_home"] / "skills" / "foo" / "SKILL.md")
            )
            is not None
        )
        assert (
            get_read_block_error(str(fake_hermes["default_home"] / "config.yaml"))
            is None
        )

    def test_default_mode_preserves_existing_cross_profile_behavior(
        self, fake_hermes, monkeypatch
    ):
        _set_active_home(monkeypatch, fake_hermes["security_home"])
        import agent.file_safety as fs

        monkeypatch.setattr(fs, "_profile_scope_settings", lambda: ("none", ()))
        target = fake_hermes["default_home"] / "memories" / "MEMORY.md"
        assert fs.get_read_block_error(str(target)) is None

    def test_fresh_malformed_config_fails_closed(self, fake_hermes, monkeypatch):
        """A fresh process must not turn an invalid strict config into no scope."""
        _set_active_home(monkeypatch, fake_hermes["security_home"])
        import agent.file_safety as fs
        import hermes_cli.config as config

        config_path = fake_hermes["security_home"] / "config.yaml"
        config_path.write_text(
            "agent:\n  profile_scope: strict\n  profile_scope_allow: [\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(config, "get_config_path", lambda: config_path)
        config_key = str(config_path)
        fs._PROFILE_SCOPE_CONFIG_HEALTH.pop(config_key, None)
        config._LOAD_CONFIG_CACHE.pop(config_key, None)
        config._LAST_EXPANDED_CONFIG_BY_PATH.pop(config_key, None)

        assert fs._profile_scope_settings() == ("strict", ())

    def test_unreadable_config_fails_closed(self, fake_hermes, monkeypatch):
        """An unreadable config cannot retain a last-known-good allow path."""
        _set_active_home(monkeypatch, fake_hermes["security_home"])
        import builtins

        import agent.file_safety as fs
        import hermes_cli.config as config

        config_path = fake_hermes["security_home"] / "config.yaml"
        shared_cache = fake_hermes["root"] / "cache"
        config_path.write_text(
            f"agent:\n  profile_scope: strict\n  profile_scope_allow: [{shared_cache}]\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(config, "get_config_path", lambda: config_path)
        config_key = str(config_path)
        fs._PROFILE_SCOPE_CONFIG_HEALTH.pop(config_key, None)
        config._LOAD_CONFIG_CACHE.pop(config_key, None)
        config._LAST_EXPANDED_CONFIG_BY_PATH.pop(config_key, None)
        assert fs._profile_scope_settings() == ("strict", (shared_cache.resolve(),))
        assert fs.get_read_block_error(str(shared_cache)) is None
        assert fs.get_write_denied_error(str(shared_cache)) is None

        real_open = builtins.open

        def deny_profile_config(file, *args, **kwargs):
            if Path(file) == config_path:
                raise PermissionError("simulated unreadable config")
            return real_open(file, *args, **kwargs)

        monkeypatch.setattr(builtins, "open", deny_profile_config)
        # The permission change invalidates a real file's ctime. Clear the
        # synthetic test cache to model that source-state transition.
        fs._PROFILE_SCOPE_CONFIG_HEALTH.pop(config_key, None)

        assert fs._profile_scope_settings() == ("strict", ())
        assert fs.get_read_block_error(str(shared_cache)) is not None
        assert fs.get_write_denied_error(str(shared_cache)) is not None

    def test_last_known_good_strict_scope_drops_allow_paths_during_config_corruption(
        self, fake_hermes, monkeypatch
    ):
        _set_active_home(monkeypatch, fake_hermes["security_home"])
        import agent.file_safety as fs
        import hermes_cli.config as config

        config_path = fake_hermes["security_home"] / "config.yaml"
        shared_cache = fake_hermes["root"] / "cache"
        config_path.write_text(
            f"agent:\n  profile_scope: strict\n  profile_scope_allow: [{shared_cache}]\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(config, "get_config_path", lambda: config_path)
        config_key = str(config_path)
        fs._PROFILE_SCOPE_CONFIG_HEALTH.pop(config_key, None)
        config._LOAD_CONFIG_CACHE.pop(config_key, None)
        config._LAST_EXPANDED_CONFIG_BY_PATH.pop(config_key, None)
        assert fs._profile_scope_settings() == ("strict", (shared_cache.resolve(),))
        assert fs.get_read_block_error(str(shared_cache)) is None
        assert fs.get_write_denied_error(str(shared_cache)) is None

        config_path.write_text("agent: [\n", encoding="utf-8")

        assert fs._profile_scope_settings() == ("strict", ())
        assert fs.get_read_block_error(str(shared_cache)) is not None
        assert fs.get_write_denied_error(str(shared_cache)) is not None

        config_path.write_text(
            f"agent:\n  profile_scope: strict\n  profile_scope_allow: [{shared_cache}]\n",
            encoding="utf-8",
        )

        assert fs._profile_scope_settings() == ("strict", (shared_cache.resolve(),))
        assert fs.get_read_block_error(str(shared_cache)) is None
        assert fs.get_write_denied_error(str(shared_cache)) is None

    def test_host_read_uses_the_guarded_task_path_when_backend_cwd_differs(
        self, fake_hermes, monkeypatch
    ):
        """A host backend must not re-resolve a safe relative path elsewhere."""
        _set_active_home(monkeypatch, fake_hermes["security_home"])
        _set_strict_scope(monkeypatch)
        import tools.file_tools as file_tools
        import tools.terminal_tool as terminal_tool

        task_id = "strict-read-cwd"
        workspace = fake_hermes["security_home"] / "workspace"
        workspace.mkdir()
        (workspace / "report.txt").write_text("OWN_REPORT\n", encoding="utf-8")
        (fake_hermes["coder_home"] / "report.txt").write_text(
            "SIBLING_SECRET\n", encoding="utf-8"
        )
        monkeypatch.setattr(terminal_tool, "_task_env_overrides", {})
        monkeypatch.setattr(terminal_tool, "_session_cwd", {})
        terminal_tool.register_task_env_overrides(task_id, {"cwd": str(workspace)})
        backend = _DivergentHostFileOps(fake_hermes["coder_home"])
        monkeypatch.setattr(file_tools, "_get_file_ops", lambda _task_id: backend)

        result = json.loads(file_tools.read_file_tool("report.txt", task_id=task_id))

        assert "OWN_REPORT" in result["content"]
        assert "SIBLING_SECRET" not in result["content"]

    def test_host_search_uses_the_guarded_task_path_when_backend_cwd_differs(
        self, fake_hermes, monkeypatch
    ):
        """Search must share read_file's canonical host-path sink invariant."""
        _set_active_home(monkeypatch, fake_hermes["security_home"])
        _set_strict_scope(monkeypatch)
        import tools.file_tools as file_tools
        import tools.terminal_tool as terminal_tool

        task_id = "strict-search-cwd"
        workspace = fake_hermes["security_home"] / "workspace"
        workspace.mkdir()
        (workspace / "report.txt").write_text("OWN_REPORT\n", encoding="utf-8")
        (fake_hermes["coder_home"] / "report.txt").write_text(
            "SIBLING_SEARCH_SECRET\n", encoding="utf-8"
        )
        monkeypatch.setattr(terminal_tool, "_task_env_overrides", {})
        monkeypatch.setattr(terminal_tool, "_session_cwd", {})
        terminal_tool.register_task_env_overrides(task_id, {"cwd": str(workspace)})
        backend = _DivergentHostFileOps(fake_hermes["coder_home"])
        monkeypatch.setattr(file_tools, "_get_file_ops", lambda _task_id: backend)

        result = file_tools.search_tool(
            "SIBLING_SEARCH_SECRET", path=".", task_id=task_id
        )

        assert "SIBLING_SEARCH_SECRET" not in result

    def test_container_backend_keeps_its_original_path_mapping(self):
        """Only host backends receive the canonical host path."""
        from pathlib import PurePosixPath

        from tools.file_tools import _file_ops_path

        class ContainerFileOps:
            env = object()

        assert (
            _file_ops_path(
                "report.txt", PurePosixPath("/workspace/report.txt"), ContainerFileOps()
            )
            == "report.txt"
        )

    def test_file_tools_enforce_strict_scope_even_with_write_bypass(
        self, fake_hermes, monkeypatch
    ):
        _set_active_home(monkeypatch, fake_hermes["security_home"])
        _set_strict_scope(monkeypatch)
        from tools.file_tools import read_file_tool, search_tool, write_file_tool

        target = fake_hermes["default_home"] / "memories" / "MEMORY.md"
        target.write_text("default profile memory\n", encoding="utf-8")
        original = target.read_text(encoding="utf-8")

        read_result = json.loads(read_file_tool(str(target), task_id="strict-read"))
        search_result = json.loads(
            search_tool("default", path=str(target), task_id="strict-search")
        )
        write_result = json.loads(
            write_file_tool(
                str(target),
                "overwritten",
                task_id="strict-write",
                cross_profile=True,
            )
        )

        for result in (read_result, search_result, write_result):
            assert "agent.profile_scope: strict" in result["error"]
        assert target.read_text(encoding="utf-8") == original
