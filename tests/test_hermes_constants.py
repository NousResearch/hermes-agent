"""Tests for hermes_constants module."""

import os
from pathlib import Path

import pytest

import hermes_constants
from hermes_constants import (
    VALID_REASONING_EFFORTS,
    get_default_hermes_root,
    get_hermes_dir,
    get_hermes_home,
    is_container,
    parse_reasoning_effort,
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


class TestGetHermesDir:
    """Tests for ``get_hermes_dir(new_subpath, old_name)``.

    Contract: prefer the legacy ``<old_name>/`` location, but only when
    it has content. An empty legacy stub must fall through to the new
    layout so dormant install scaffolds don't orphan populated data at
    ``<new_subpath>/``. Regression guard for #27602.
    """

    def _set_home(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    def test_neither_exists_returns_new(self, tmp_path, monkeypatch):
        self._set_home(tmp_path, monkeypatch)
        result = get_hermes_dir("platforms/pairing", "pairing")
        assert result == tmp_path / "platforms/pairing"

    def test_legacy_populated_returns_legacy(self, tmp_path, monkeypatch):
        self._set_home(tmp_path, monkeypatch)
        legacy = tmp_path / "image_cache"
        legacy.mkdir()
        (legacy / "cached.png").write_bytes(b"x")
        result = get_hermes_dir("cache/images", "image_cache")
        assert result == legacy

    def test_legacy_populated_with_subdir_returns_legacy(self, tmp_path, monkeypatch):
        """Sub-directories count as content (e.g. nested cache layout)."""
        self._set_home(tmp_path, monkeypatch)
        legacy = tmp_path / "matrix" / "store"
        legacy.mkdir(parents=True)
        (legacy / "session").mkdir()  # subdir, not a file
        result = get_hermes_dir("platforms/matrix/store", "matrix/store")
        assert result == legacy

    def test_legacy_empty_returns_new(self, tmp_path, monkeypatch):
        """The #27602 regression: empty legacy dir orphans populated new dir.

        Without the fix, the resolver returned the empty legacy path
        unconditionally, causing the pairing store to forget every
        previously-approved user when an empty ``pairing/`` stub had
        been pre-created at install time.
        """
        self._set_home(tmp_path, monkeypatch)
        legacy = tmp_path / "pairing"
        legacy.mkdir()
        # Populated new layout — this is the data that must not be orphaned.
        new = tmp_path / "platforms" / "pairing"
        new.mkdir(parents=True)
        (new / "telegram-approved.json").write_text("[]")
        result = get_hermes_dir("platforms/pairing", "pairing")
        assert result == new

    def test_legacy_empty_and_new_missing_returns_new(self, tmp_path, monkeypatch):
        """Empty legacy + no new yet — return the new path (will be created lazily).

        Slight behaviour change vs the old resolver (which would return the
        empty legacy dir): the new path is what every consumer mkdirs into
        when it doesn't exist, so the next write lands in the canonical
        location instead of perpetuating the empty stub.
        """
        self._set_home(tmp_path, monkeypatch)
        legacy = tmp_path / "audio_cache"
        legacy.mkdir()
        result = get_hermes_dir("cache/audio", "audio_cache")
        assert result == tmp_path / "cache/audio"

    def test_legacy_is_file_treated_as_content(self, tmp_path, monkeypatch):
        """A non-directory file at the legacy path counts as occupied.

        Defensive against odd installs where the caller previously wrote a
        single file instead of a directory. We honour whatever's there.
        """
        self._set_home(tmp_path, monkeypatch)
        legacy = tmp_path / "image_cache"
        legacy.write_bytes(b"sentinel")
        result = get_hermes_dir("cache/images", "image_cache")
        assert result == legacy

    def test_unreadable_legacy_dir_kept(self, tmp_path, monkeypatch):
        """If we can't enumerate the legacy dir, assume occupied — never
        accidentally orphan legacy data on a transient permission error.
        """
        self._set_home(tmp_path, monkeypatch)
        legacy = tmp_path / "whatsapp" / "session"
        legacy.mkdir(parents=True)
        # Populate the new path too. The point is to verify that an
        # OSError on iterdir does NOT fall through to the new layout.
        new = tmp_path / "platforms" / "whatsapp" / "session"
        new.mkdir(parents=True)
        (new / "creds.json").write_text("{}")

        real_iterdir = Path.iterdir

        def boom(self):
            if self == legacy:
                raise PermissionError("simulated")
            return real_iterdir(self)

        monkeypatch.setattr(Path, "iterdir", boom)
        result = get_hermes_dir(
            "platforms/whatsapp/session", "whatsapp/session"
        )
        assert result == legacy

    def test_unstatable_legacy_dir_kept(self, tmp_path, monkeypatch):
        """A ``PermissionError`` raised by the existence check itself (e.g.
        an unreadable parent) must NOT be read as "absent".

        The old ``Path.exists()``/``Path.is_dir()`` gate swallowed
        ``PermissionError`` and returned ``False``, so an unreadable legacy
        dir fell through to the new layout and orphaned legacy data —
        contradicting the docstring's "assume occupied on errors" intent.
        With the ``lstat()``-based gate this raises and is caught as
        occupied. Regression guard for the #27602 follow-up.
        """
        self._set_home(tmp_path, monkeypatch)
        legacy = tmp_path / "pairing"
        legacy.mkdir()
        # Populate the new path; it must NOT be selected.
        new = tmp_path / "platforms" / "pairing"
        new.mkdir(parents=True)
        (new / "telegram-approved.json").write_text("[]")

        real_lstat = Path.lstat

        def boom(self):
            if self == legacy:
                raise PermissionError("simulated unreadable parent")
            return real_lstat(self)

        monkeypatch.setattr(Path, "lstat", boom)
        result = get_hermes_dir("platforms/pairing", "pairing")
        assert result == legacy
