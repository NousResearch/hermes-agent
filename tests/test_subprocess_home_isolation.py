"""Tests for subprocess HOME handling in profile mode.

Hermes state stays profile-scoped through HERMES_HOME. Host subprocesses should
keep the user's real HOME by default so external CLIs find existing credentials.
Containers still use the profile home for persistence, and users can explicitly
opt into profile HOME isolation on the host.

See: https://github.com/NousResearch/hermes-agent/issues/25114
See: https://github.com/NousResearch/hermes-agent/issues/36144
See: https://github.com/NousResearch/hermes-agent/issues/29015
"""

import os
import threading
from pathlib import Path

import hermes_constants



# ---------------------------------------------------------------------------
# get_subprocess_home()
# ---------------------------------------------------------------------------

class TestGetSubprocessHome:
    """Unit tests for hermes_constants.get_subprocess_home()."""

    def _host_mode(self, monkeypatch):
        monkeypatch.setattr(hermes_constants, "is_container", lambda: False)
        monkeypatch.delenv("TERMINAL_HOME_MODE", raising=False)
        monkeypatch.delenv("HERMES_REAL_HOME", raising=False)

    def _container_mode(self, monkeypatch):
        monkeypatch.setattr(hermes_constants, "is_container", lambda: True)
        monkeypatch.delenv("TERMINAL_HOME_MODE", raising=False)
        monkeypatch.delenv("HERMES_REAL_HOME", raising=False)



    def test_host_auto_keeps_real_home_when_profile_home_exists(self, tmp_path, monkeypatch):
        """Host installs should not hide real ~/.ssh, ~/.gitconfig, ~/.azure, etc."""
        self._host_mode(monkeypatch)
        real_home = tmp_path / "real-home"
        hermes_home = real_home / ".hermes" / "profiles" / "coder"
        profile_home = hermes_home / "home"
        profile_home.mkdir(parents=True)
        monkeypatch.setenv("HOME", str(real_home))
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        from hermes_constants import get_subprocess_home
        assert get_subprocess_home() is None

    def test_container_auto_uses_profile_home_when_home_dir_exists(self, tmp_path, monkeypatch):
        self._container_mode(monkeypatch)
        hermes_home = tmp_path / ".hermes"
        profile_home = hermes_home / "home"
        profile_home.mkdir(parents=True)
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        from hermes_constants import get_subprocess_home
        assert get_subprocess_home() == str(profile_home)

    def test_returns_profile_specific_path(self, tmp_path, monkeypatch):
        """Explicit profile mode keeps the old per-profile HOME behavior."""
        self._host_mode(monkeypatch)
        profile_dir = tmp_path / ".hermes" / "profiles" / "coder"
        profile_dir.mkdir(parents=True)
        profile_home = profile_dir / "home"
        profile_home.mkdir()
        monkeypatch.setenv("TERMINAL_HOME_MODE", "profile")
        monkeypatch.setenv("HERMES_HOME", str(profile_dir))
        from hermes_constants import get_subprocess_home
        assert get_subprocess_home() == str(profile_home)

    def test_real_mode_repairs_parent_home_already_pointing_at_profile(self, tmp_path, monkeypatch):
        self._host_mode(monkeypatch)
        profile_dir = tmp_path / ".hermes" / "profiles" / "coder"
        profile_home = profile_dir / "home"
        profile_home.mkdir(parents=True)
        real_home = tmp_path / "real-home"
        real_home.mkdir()
        monkeypatch.setenv("TERMINAL_HOME_MODE", "real")
        monkeypatch.setenv("HERMES_HOME", str(profile_dir))
        monkeypatch.setenv("HOME", str(profile_home))
        monkeypatch.setenv("HERMES_REAL_HOME", str(real_home))

        from hermes_constants import get_subprocess_home, get_real_home

        assert get_real_home() == str(real_home)
        assert get_subprocess_home() == str(real_home)


    def test_two_profiles_get_different_homes(self, tmp_path, monkeypatch):
        self._container_mode(monkeypatch)
        base = tmp_path / ".hermes" / "profiles"
        for name in ("alpha", "beta"):
            p = base / name
            p.mkdir(parents=True)
            (p / "home").mkdir()

        from hermes_constants import get_subprocess_home

        monkeypatch.setenv("HERMES_HOME", str(base / "alpha"))
        home_a = get_subprocess_home()

        monkeypatch.setenv("HERMES_HOME", str(base / "beta"))
        home_b = get_subprocess_home()

        assert home_a is not None
        assert home_b is not None
        assert home_a != home_b
        assert home_a.endswith("alpha/home")
        assert home_b.endswith("beta/home")


# ---------------------------------------------------------------------------
# external_credential_home_candidates() / external_credential_path()
# ---------------------------------------------------------------------------

class TestExternalCredentialPath:
    """External CLI credential stores must stay reachable from a profile HOME.

    ``~/.claude/.credentials.json``, ``gh``'s config and ``~/.codex`` are written by
    third-party CLIs under the OS account's home. A Hermes process running with
    ``HOME={HERMES_HOME}/home`` (container installs, and hosts where ``is_container()``
    false-positives — #58135) previously read the empty profile home and reported the
    provider as unauthenticated, dropping its ``/model`` row.
    """

    def _profile_home_env(self, tmp_path, monkeypatch):
        """HOME points at the profile home; the real home holds the credential."""
        hermes_home = tmp_path / ".hermes" / "profiles" / "coder"
        profile_home = hermes_home / "home"
        profile_home.mkdir(parents=True)
        real_home = tmp_path / "real-home"
        real_home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setenv("HOME", str(profile_home))
        monkeypatch.setenv("HERMES_REAL_HOME", str(real_home))
        return profile_home, real_home

    def test_candidates_put_current_home_first_then_real_home(self, tmp_path, monkeypatch):
        profile_home, real_home = self._profile_home_env(tmp_path, monkeypatch)
        from hermes_constants import external_credential_home_candidates
        assert external_credential_home_candidates() == [Path(profile_home), Path(real_home)]

    def test_falls_back_to_real_home_when_profile_home_lacks_the_file(self, tmp_path, monkeypatch):
        _profile_home, real_home = self._profile_home_env(tmp_path, monkeypatch)
        creds = real_home / ".claude" / ".credentials.json"
        creds.parent.mkdir(parents=True)
        creds.write_text("{}", encoding="utf-8")

        from hermes_constants import external_credential_path
        assert external_credential_path(".claude", ".credentials.json") == creds

    def test_profile_home_wins_when_it_holds_the_file(self, tmp_path, monkeypatch):
        """A genuine container install logs in under the profile home — keep preferring it."""
        profile_home, real_home = self._profile_home_env(tmp_path, monkeypatch)
        for home in (profile_home, real_home):
            creds = home / ".claude" / ".credentials.json"
            creds.parent.mkdir(parents=True)
            creds.write_text("{}", encoding="utf-8")

        from hermes_constants import external_credential_path
        assert external_credential_path(".claude", ".credentials.json") == (
            profile_home / ".claude" / ".credentials.json")

    def test_missing_everywhere_returns_current_home_so_writers_keep_their_target(
            self, tmp_path, monkeypatch):
        profile_home, _real_home = self._profile_home_env(tmp_path, monkeypatch)
        from hermes_constants import external_credential_path
        assert external_credential_path(".claude", ".credentials.json") == (
            profile_home / ".claude" / ".credentials.json")

    def test_no_profile_home_yields_a_single_candidate(self, tmp_path, monkeypatch):
        real_home = tmp_path / "real-home"
        real_home.mkdir()
        monkeypatch.delenv("HERMES_HOME", raising=False)
        monkeypatch.delenv("HERMES_REAL_HOME", raising=False)
        monkeypatch.setenv("HOME", str(real_home))
        from hermes_constants import external_credential_home_candidates
        assert external_credential_home_candidates() == [Path(real_home)]

    def test_normal_home_is_never_widened_even_with_a_profile_home_present(
            self, tmp_path, monkeypatch):
        """Off a profile HOME the helper must behave exactly like ``Path.home()``."""
        hermes_home = tmp_path / ".hermes" / "profiles" / "coder"
        (hermes_home / "home").mkdir(parents=True)
        real_home = tmp_path / "real-home"
        real_home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setenv("HOME", str(real_home))
        monkeypatch.delenv("HERMES_REAL_HOME", raising=False)
        from hermes_constants import external_credential_home_candidates
        assert external_credential_home_candidates() == [Path(real_home)]

    def test_claude_code_credentials_path_follows_the_real_home(self, tmp_path, monkeypatch):
        """The concrete regression: Anthropic disappearing from the picker (#58135)."""
        _profile_home, real_home = self._profile_home_env(tmp_path, monkeypatch)
        creds = real_home / ".claude" / ".credentials.json"
        creds.parent.mkdir(parents=True)
        creds.write_text("{}", encoding="utf-8")

        from agent.anthropic_credentials import claude_code_credentials_path
        assert claude_code_credentials_path() == creds



# ---------------------------------------------------------------------------
# _make_run_env() injection
# ---------------------------------------------------------------------------

class TestMakeRunEnvHomeInjection:
    """Verify _make_run_env() applies the subprocess HOME policy."""

    def test_host_auto_preserves_real_home_when_profile_home_exists(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        (hermes_home / "home").mkdir()
        real_home = tmp_path / "real-home"
        real_home.mkdir()
        monkeypatch.setattr(hermes_constants, "is_container", lambda: False)
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setenv("HOME", str(real_home))
        monkeypatch.setenv("PATH", "/usr/bin:/bin")

        from tools.environments.local import _make_run_env
        result = _make_run_env({})

        assert result["HOME"] == str(real_home)
        assert result["HERMES_REAL_HOME"] == str(real_home)

    def test_profile_mode_injects_profile_home_when_profile_home_exists(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        (hermes_home / "home").mkdir()
        real_home = tmp_path / "real-home"
        real_home.mkdir()
        monkeypatch.setattr(hermes_constants, "is_container", lambda: False)
        monkeypatch.setenv("TERMINAL_HOME_MODE", "profile")
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setenv("HOME", str(real_home))
        monkeypatch.setenv("PATH", "/usr/bin:/bin")

        from tools.environments.local import _make_run_env
        result = _make_run_env({})

        assert result["HOME"] == str(hermes_home / "home")
        assert result["HERMES_REAL_HOME"] == str(real_home)

    def test_no_injection_when_home_dir_missing(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        # No home/ subdirectory
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setenv("HOME", "/root")
        monkeypatch.setenv("PATH", "/usr/bin:/bin")

        from tools.environments.local import _make_run_env
        result = _make_run_env({})

        assert result["HOME"] == "/root"

    def test_no_injection_when_hermes_home_unset(self, monkeypatch):
        monkeypatch.delenv("HERMES_HOME", raising=False)
        monkeypatch.setenv("HOME", "/home/user")
        monkeypatch.setenv("PATH", "/usr/bin:/bin")

        from tools.environments.local import _make_run_env
        result = _make_run_env({})

        assert result["HOME"] == "/home/user"



# ---------------------------------------------------------------------------
# _sanitize_subprocess_env() injection
# ---------------------------------------------------------------------------

class TestSanitizeSubprocessEnvHomeInjection:
    """Verify _sanitize_subprocess_env() applies the subprocess HOME policy."""

    def test_host_auto_preserves_real_home_when_profile_home_exists(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        (hermes_home / "home").mkdir()
        real_home = tmp_path / "real-home"
        real_home.mkdir()
        monkeypatch.setattr(hermes_constants, "is_container", lambda: False)
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        base_env = {"HOME": str(real_home), "PATH": "/usr/bin", "USER": "root"}
        from tools.environments.local import _sanitize_subprocess_env
        result = _sanitize_subprocess_env(base_env)

        assert result["HOME"] == str(real_home)
        assert result["HERMES_REAL_HOME"] == str(real_home)

    def test_profile_mode_injects_profile_home_when_profile_home_exists(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        (hermes_home / "home").mkdir()
        real_home = tmp_path / "real-home"
        real_home.mkdir()
        monkeypatch.setattr(hermes_constants, "is_container", lambda: False)
        monkeypatch.setenv("TERMINAL_HOME_MODE", "profile")
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        base_env = {"HOME": str(real_home), "PATH": "/usr/bin", "USER": "root"}
        from tools.environments.local import _sanitize_subprocess_env
        result = _sanitize_subprocess_env(base_env)

        assert result["HOME"] == str(hermes_home / "home")
        assert result["HERMES_REAL_HOME"] == str(real_home)

    def test_no_injection_when_home_dir_missing(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        base_env = {"HOME": "/root", "PATH": "/usr/bin"}
        from tools.environments.local import _sanitize_subprocess_env
        result = _sanitize_subprocess_env(base_env)

        assert result["HOME"] == "/root"



# ---------------------------------------------------------------------------
# Profile bootstrap
# ---------------------------------------------------------------------------

class TestProfileBootstrap:
    """Verify new profiles get a home/ subdirectory."""

    def test_profile_dirs_includes_home(self):
        from hermes_cli.profiles import _PROFILE_DIRS
        assert "home" in _PROFILE_DIRS

    def test_create_profile_bootstraps_home_dir(self, tmp_path, monkeypatch):
        """create_profile() should create home/ inside the profile dir."""
        home = tmp_path / ".hermes"
        home.mkdir()
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setenv("HERMES_HOME", str(home))

        from hermes_cli.profiles import create_profile
        profile_dir = create_profile("testbot", no_alias=True)
        assert (profile_dir / "home").is_dir()


# ---------------------------------------------------------------------------
# Python process HOME unchanged
# ---------------------------------------------------------------------------

