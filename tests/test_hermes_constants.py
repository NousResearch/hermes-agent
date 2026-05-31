"""Tests for hermes_constants module."""

import os
from pathlib import Path

import pytest

import hermes_constants
from hermes_constants import (
    VALID_REASONING_EFFORTS,
    extract_reasoning_keyword,
    get_default_hermes_root,
    get_hermes_home,
    is_container,
    parse_reasoning_effort,
    reasoning_effort_rank,
    secure_parent_dir,
)


class TestGetDefaultHermesRoot:
    """Tests for get_default_hermes_root() — Docker/custom deployment awareness."""

    def test_no_hermes_home_returns_native(self, tmp_path, monkeypatch):
        """When HERMES_HOME is not set, returns ~/.hermes."""
        monkeypatch.delenv("HERMES_HOME", raising=False)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        assert get_default_hermes_root() == tmp_path / ".hermes"

    def test_hermes_home_is_native(self, tmp_path, monkeypatch):
        """When HERMES_HOME = ~/.hermes, returns ~/.hermes."""
        native = tmp_path / ".hermes"
        native.mkdir()
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setenv("HERMES_HOME", str(native))
        assert get_default_hermes_root() == native

    def test_hermes_home_is_profile(self, tmp_path, monkeypatch):
        """When HERMES_HOME is a profile under ~/.hermes, returns ~/.hermes."""
        native = tmp_path / ".hermes"
        profile = native / "profiles" / "coder"
        profile.mkdir(parents=True)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setenv("HERMES_HOME", str(profile))
        assert get_default_hermes_root() == native

    def test_hermes_home_is_docker(self, tmp_path, monkeypatch):
        """When HERMES_HOME points outside ~/.hermes (Docker), returns HERMES_HOME."""
        docker_home = tmp_path / "opt" / "data"
        docker_home.mkdir(parents=True)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setenv("HERMES_HOME", str(docker_home))
        assert get_default_hermes_root() == docker_home

    def test_hermes_home_is_custom_path(self, tmp_path, monkeypatch):
        """Any HERMES_HOME outside ~/.hermes is treated as the root."""
        custom = tmp_path / "my-hermes-data"
        custom.mkdir()
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setenv("HERMES_HOME", str(custom))
        assert get_default_hermes_root() == custom

    def test_docker_profile_active(self, tmp_path, monkeypatch):
        """When a Docker profile is active (HERMES_HOME=<root>/profiles/<name>),
        returns the Docker root, not the profile dir."""
        docker_root = tmp_path / "opt" / "data"
        profile = docker_root / "profiles" / "coder"
        profile.mkdir(parents=True)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setenv("HERMES_HOME", str(profile))
        assert get_default_hermes_root() == docker_root

    def test_no_hermes_home_returns_localappdata_root_on_windows(self, tmp_path, monkeypatch):
        """Native Windows falls back to %LOCALAPPDATA%\\hermes, not ~/.hermes."""
        local_appdata = tmp_path / "LocalAppData"
        monkeypatch.delenv("HERMES_HOME", raising=False)
        monkeypatch.setenv("LOCALAPPDATA", str(local_appdata))
        monkeypatch.setattr(Path, "home", lambda: tmp_path / "Home")
        monkeypatch.setattr(hermes_constants.sys, "platform", "win32")

        assert get_default_hermes_root() == local_appdata / "hermes"

    def test_no_hermes_home_uses_windows_path_when_localappdata_missing(self, tmp_path, monkeypatch):
        """Windows fallback still uses AppData/Local/hermes without LOCALAPPDATA."""
        home = tmp_path / "Home"
        monkeypatch.delenv("HERMES_HOME", raising=False)
        monkeypatch.delenv("LOCALAPPDATA", raising=False)
        monkeypatch.setattr(Path, "home", lambda: home)
        monkeypatch.setattr(hermes_constants.sys, "platform", "win32")

        assert get_default_hermes_root() == home / "AppData" / "Local" / "hermes"


class TestGetHermesHome:
    """Tests for get_hermes_home() platform-aware fallback."""

    def test_windows_fallback_uses_localappdata(self, tmp_path, monkeypatch):
        """When HERMES_HOME is unset on Windows, use %LOCALAPPDATA%\\hermes."""
        local_appdata = tmp_path / "LocalAppData"
        monkeypatch.delenv("HERMES_HOME", raising=False)
        monkeypatch.setenv("LOCALAPPDATA", str(local_appdata))
        monkeypatch.setattr(Path, "home", lambda: tmp_path / "Home")
        monkeypatch.setattr(hermes_constants.sys, "platform", "win32")
        monkeypatch.setattr(hermes_constants, "_profile_fallback_warned", False)

        assert get_hermes_home() == local_appdata / "hermes"


class TestIsContainer:
    """Tests for is_container() — Docker/Podman detection."""

    def _reset_cache(self, monkeypatch):
        """Reset the cached detection result before each test."""
        monkeypatch.setattr(hermes_constants, "_container_detected", None)

    def test_detects_dockerenv(self, monkeypatch, tmp_path):
        """/.dockerenv triggers container detection."""
        self._reset_cache(monkeypatch)
        monkeypatch.setattr(os.path, "exists", lambda p: p == "/.dockerenv")
        assert is_container() is True

    def test_detects_containerenv(self, monkeypatch, tmp_path):
        """/run/.containerenv triggers container detection (Podman)."""
        self._reset_cache(monkeypatch)
        monkeypatch.setattr(os.path, "exists", lambda p: p == "/run/.containerenv")
        assert is_container() is True

    def test_detects_cgroup_docker(self, monkeypatch, tmp_path):
        """/proc/1/cgroup containing 'docker' triggers detection."""
        import builtins
        self._reset_cache(monkeypatch)
        monkeypatch.setattr(os.path, "exists", lambda p: False)
        cgroup_file = tmp_path / "cgroup"
        cgroup_file.write_text("12:memory:/docker/abc123\n")
        _real_open = builtins.open
        monkeypatch.setattr("builtins.open", lambda p, *a, **kw: _real_open(str(cgroup_file), *a, **kw) if p == "/proc/1/cgroup" else _real_open(p, *a, **kw))
        assert is_container() is True

    def test_negative_case(self, monkeypatch, tmp_path):
        """Returns False on a regular Linux host."""
        import builtins
        self._reset_cache(monkeypatch)
        monkeypatch.setattr(os.path, "exists", lambda p: False)
        cgroup_file = tmp_path / "cgroup"
        cgroup_file.write_text("12:memory:/\n")
        _real_open = builtins.open
        monkeypatch.setattr("builtins.open", lambda p, *a, **kw: _real_open(str(cgroup_file), *a, **kw) if p == "/proc/1/cgroup" else _real_open(p, *a, **kw))
        assert is_container() is False

    def test_caches_result(self, monkeypatch):
        """Second call uses cached value without re-probing."""
        monkeypatch.setattr(hermes_constants, "_container_detected", True)
        assert is_container() is True
        # Even if we make os.path.exists return False, cached value wins
        monkeypatch.setattr(os.path, "exists", lambda p: False)
        assert is_container() is True


class TestParseReasoningEffort:
    """Tests for parse_reasoning_effort() — string → reasoning config dict."""

    @pytest.mark.parametrize("value", ["", "   ", "\t", "\n"])
    def test_empty_or_whitespace_returns_none(self, value):
        """Empty / whitespace-only input falls back to caller default (None)."""
        assert parse_reasoning_effort(value) is None

    def test_none_disables_reasoning(self):
        """The literal "none" disables reasoning explicitly."""
        assert parse_reasoning_effort("none") == {"enabled": False}

    @pytest.mark.parametrize("level", list(VALID_REASONING_EFFORTS))
    def test_each_valid_level(self, level):
        """Every level listed in VALID_REASONING_EFFORTS is accepted as-is."""
        assert parse_reasoning_effort(level) == {"enabled": True, "effort": level}

    @pytest.mark.parametrize(
        "raw, expected_effort",
        [
            ("MEDIUM", "medium"),
            ("High", "high"),
            ("  low  ", "low"),
            ("\tXHIGH\n", "xhigh"),
            ("None", False),
        ],
    )
    def test_case_and_whitespace_normalized(self, raw, expected_effort):
        """Mixed case and surrounding whitespace are normalized before lookup."""
        result = parse_reasoning_effort(raw)
        if expected_effort is False:
            assert result == {"enabled": False}
        else:
            assert result == {"enabled": True, "effort": expected_effort}

    @pytest.mark.parametrize(
        "value",
        ["bogus", "very-high", "max", "0", "off", "true", "default"],
    )
    def test_unknown_levels_return_none(self, value):
        """Unrecognized strings fall back to the caller default (None)."""
        assert parse_reasoning_effort(value) is None

    def test_known_supported_levels_are_documented(self):
        """Guard against silently dropping a documented level.

        The docstring promises "minimal", "low", "medium", "high", "xhigh".
        If someone removes one from VALID_REASONING_EFFORTS without updating
        the docstring, this test will fail and force the call out.
        """
        documented = {"minimal", "low", "medium", "high", "xhigh"}
        assert documented.issubset(set(VALID_REASONING_EFFORTS))


class TestSecureParentDir:
    """Tests for secure_parent_dir() — prevents chmod on / or top-level dirs."""

    def test_safe_path_calls_chmod(self, tmp_path, monkeypatch):
        """Normal nested path (depth >= 3) should call os.chmod."""
        safe_dir = tmp_path / "home" / "user" / ".hermes"
        safe_dir.mkdir(parents=True)
        target = safe_dir / "auth.json"
        target.touch()

        called_with = []
        monkeypatch.setattr(os, "chmod", lambda p, m: called_with.append((str(p), m)))

        secure_parent_dir(target)
        assert len(called_with) == 1
        assert called_with[0] == (str(safe_dir), 0o700)

    def test_root_dir_skipped(self, monkeypatch):
        """Parent resolving to / must NOT be chmod'd."""
        called_with = []
        monkeypatch.setattr(os, "chmod", lambda p, m: called_with.append((str(p), m)))

        # Path("/foo").parent == Path("/")
        secure_parent_dir(Path("/foo"))
        assert called_with == []

    def test_top_level_dir_skipped(self, monkeypatch):
        """Parent resolving to a top-level dir (depth 2) must NOT be chmod'd."""
        called_with = []
        monkeypatch.setattr(os, "chmod", lambda p, m: called_with.append((str(p), m)))

        # Path("/usr/foo").parent == Path("/usr") — depth 2
        secure_parent_dir(Path("/usr/foo"))
        assert called_with == []

    def test_two_component_path_skipped(self, monkeypatch):
        """Parent with < 3 resolved parts must NOT be chmod'd.

        Uses monkeypatch to avoid macOS firmlink resolution of /home.
        """
        called_with = []
        monkeypatch.setattr(os, "chmod", lambda p, m: called_with.append((str(p), m)))

        # Mock Path.resolve to return a short path regardless of OS quirks
        original_resolve = Path.resolve
        def mock_resolve(self):
            if str(self) == "/x/y":
                return Path("/x")
            return original_resolve(self)
        monkeypatch.setattr(Path, "resolve", mock_resolve)

        secure_parent_dir(Path("/x/y"))
        assert called_with == []

    def test_oserror_suppressed(self, tmp_path, monkeypatch):
        """OSError from chmod should be silently caught."""
        safe_dir = tmp_path / "a" / "b" / "c"
        safe_dir.mkdir(parents=True)
        target = safe_dir / "file.json"
        target.touch()

        def raise_oserror(p, m):
            raise OSError("permission denied")

        monkeypatch.setattr(os, "chmod", raise_oserror)
        # Should not raise
        secure_parent_dir(target)

    def test_symlink_resolved(self, tmp_path, monkeypatch):
        """Symlinks should be resolved before checking depth."""
        real_dir = tmp_path / "a" / "b"
        real_dir.mkdir(parents=True)
        target = real_dir / "file.json"
        target.touch()

        # Create a symlink with fewer path components
        link = tmp_path / "link"
        link.symlink_to(real_dir)
        link_target = link / "file.json"

        called_with = []
        monkeypatch.setattr(os, "chmod", lambda p, m: called_with.append((str(p), m)))

        # Even though /tmp/link has only 3 parts, the resolved path has 4
        # The resolved parent (real_dir) has depth 4, so it should be chmod'd
        secure_parent_dir(link_target)
        assert len(called_with) == 1
        assert called_with[0] == (str(real_dir), 0o700)


class TestReasoningEffortRank:
    """Tests for reasoning_effort_rank() — sortable effort levels."""

    def test_ordering_is_monotonic(self):
        """Ranks increase strictly from minimal up to xhigh."""
        order = ["minimal", "low", "medium", "high", "xhigh"]
        ranks = [reasoning_effort_rank(e) for e in order]
        assert ranks == sorted(ranks)
        assert len(set(ranks)) == len(ranks)  # all distinct

    def test_xhigh_beats_high_beats_medium(self):
        assert reasoning_effort_rank("xhigh") > reasoning_effort_rank("high")
        assert reasoning_effort_rank("high") > reasoning_effort_rank("medium")

    @pytest.mark.parametrize("value", ["", "   ", "bogus", "max", None])
    def test_unknown_falls_back_to_medium_rank(self, value):
        """Unknown/empty inputs rank as medium so a boost compares sanely."""
        assert reasoning_effort_rank(value) == reasoning_effort_rank("medium")

    @pytest.mark.parametrize("raw", ["XHIGH", "  High  ", "\tlow\n"])
    def test_case_and_whitespace_normalized(self, raw):
        assert reasoning_effort_rank(raw) == reasoning_effort_rank(raw.strip().lower())


class TestExtractReasoningKeyword:
    """Tests for extract_reasoning_keyword() — inline ultrathink-style boosts."""

    def test_ultrathink_prefix_strips_and_boosts_xhigh(self):
        cleaned, effort = extract_reasoning_keyword("ultrathink fix the auth bug")
        assert effort == "xhigh"
        assert cleaned == "fix the auth bug"
        assert "ultrathink" not in cleaned.lower()

    def test_keyword_anywhere_in_message(self):
        cleaned, effort = extract_reasoning_keyword(
            "fix the auth bug, ultrathink please"
        )
        assert effort == "xhigh"
        assert "ultrathink" not in cleaned.lower()
        assert "fix the auth bug" in cleaned

    def test_think_harder_boosts_high(self):
        cleaned, effort = extract_reasoning_keyword(
            "can you think harder about this race condition"
        )
        assert effort == "high"
        assert "think harder" not in cleaned.lower()
        assert "race condition" in cleaned

    @pytest.mark.parametrize("kw", ["ULTRATHINK", "UltraThink", "ultrathink"])
    def test_case_insensitive(self, kw):
        cleaned, effort = extract_reasoning_keyword(f"{kw} why is this flaky")
        assert effort == "xhigh"
        assert cleaned == "why is this flaky"

    def test_multiple_keywords_highest_effort_wins(self):
        # ultrathink (xhigh) + think hard (high) -> xhigh
        cleaned, effort = extract_reasoning_keyword("ultrathink and think hard")
        assert effort == "xhigh"

    def test_multiple_high_keywords_resolve_high(self):
        # megathink (high) + think hard (high) -> high, both stripped
        cleaned, effort = extract_reasoning_keyword(
            "megathink and think hard about the schema"
        )
        assert effort == "high"
        assert "megathink" not in cleaned.lower()
        assert "think hard" not in cleaned.lower()
        assert "schema" in cleaned

    @pytest.mark.parametrize("text", ["ultrathink", "   ultrathink   ", "think harder"])
    def test_lone_keyword_is_noop(self, text):
        """A message that is only a keyword must not bump effort or empty out."""
        cleaned, effort = extract_reasoning_keyword(text)
        assert effort is None
        assert cleaned == text  # original returned untouched

    def test_no_keyword_returns_unchanged(self):
        cleaned, effort = extract_reasoning_keyword("just a normal message")
        assert effort is None
        assert cleaned == "just a normal message"

    @pytest.mark.parametrize(
        "text",
        [
            "rethinking the design",     # 'think' is a substring, not a word
            "ultrathinking about it",    # ultrathink + suffix, not whole word
            "I love megathinking",       # megathink + suffix
        ],
    )
    def test_substring_does_not_match(self, text):
        """Whole-word boundaries — keywords embedded in longer words don't fire."""
        cleaned, effort = extract_reasoning_keyword(text)
        assert effort is None
        assert cleaned == text

    @pytest.mark.parametrize("bad", [None, "", 123, [], {}])
    def test_non_string_or_empty_is_safe(self, bad):
        cleaned, effort = extract_reasoning_keyword(bad)
        assert effort is None
        assert cleaned == bad

    def test_boost_effort_is_a_valid_level(self):
        """Every keyword maps to a level the rest of the system accepts."""
        for keyword, effort in hermes_constants.REASONING_KEYWORD_EFFORTS:
            assert effort in VALID_REASONING_EFFORTS

    def test_whitespace_collapsed_after_strip(self):
        """Removing a mid-sentence keyword shouldn't leave double spaces."""
        cleaned, effort = extract_reasoning_keyword(
            "please ultrathink and refactor this"
        )
        assert effort == "xhigh"
        assert "  " not in cleaned
        assert cleaned == "please and refactor this"


