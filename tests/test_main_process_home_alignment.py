"""Regression tests for main-process HOME alignment with subprocess HOME (#27250).

The historical behaviour (introduced for #4426) intentionally kept the Python
process's own ``HOME`` set to the OS-level value (e.g. ``/opt/data`` in the
Docker image) while injecting ``{HERMES_HOME}/home/`` only into subprocess
environments.  Within a single profile that produces two competing ``HOME``
values, so paths like ``~/.ssh``, ``~/.gitconfig``, ``~/.config/gh`` and
``~/workspace`` resolve to *different* directories depending on whether the
Python process or a child tool does the expansion.

``align_main_process_home_with_subprocess`` pulls the main process's ``HOME``
(and ``USERPROFILE`` on Windows so ``Path.home()`` follows) over to the same
profile-local directory subprocesses already use, while preserving isolation
across profiles and exposing an ``HERMES_PRESERVE_HOST_HOME=1`` escape hatch.

See: https://github.com/NousResearch/hermes-agent/issues/27250
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Activation gates
# ---------------------------------------------------------------------------

class TestAlignmentActivationGates:
    """The alignment must only fire when a profile ``home/`` directory exists."""

    def test_returns_none_when_hermes_home_unset(self, monkeypatch):
        monkeypatch.delenv("HERMES_HOME", raising=False)
        monkeypatch.setenv("HOME", "/home/host-user")

        from hermes_constants import align_main_process_home_with_subprocess
        assert align_main_process_home_with_subprocess() is None
        assert os.environ["HOME"] == "/home/host-user"

    def test_returns_none_when_home_subdir_missing(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setenv("HOME", "/home/host-user")

        from hermes_constants import align_main_process_home_with_subprocess
        assert align_main_process_home_with_subprocess() is None
        assert os.environ["HOME"] == "/home/host-user"

    def test_aligns_when_profile_home_exists(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        profile_home = hermes_home / "home"
        profile_home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setenv("HOME", "/home/host-user")
        monkeypatch.delenv("HERMES_PRESERVE_HOST_HOME", raising=False)

        from hermes_constants import align_main_process_home_with_subprocess
        result = align_main_process_home_with_subprocess()

        assert result == str(profile_home)
        assert os.environ["HOME"] == str(profile_home)

    def test_idempotent_when_already_aligned(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        profile_home = hermes_home / "home"
        profile_home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setenv("HOME", str(profile_home))
        monkeypatch.delenv("HERMES_PRESERVE_HOST_HOME", raising=False)

        from hermes_constants import align_main_process_home_with_subprocess
        first = align_main_process_home_with_subprocess()
        second = align_main_process_home_with_subprocess()

        assert first == second == str(profile_home)
        assert os.environ["HOME"] == str(profile_home)


# ---------------------------------------------------------------------------
# HERMES_PRESERVE_HOST_HOME escape hatch
# ---------------------------------------------------------------------------

class TestPreserveHostHomeOptOut:
    """``HERMES_PRESERVE_HOST_HOME=1`` keeps the old split-HOME behaviour."""

    @pytest.mark.parametrize("truthy", ["1", "true", "TRUE", "yes", "on", "True"])
    def test_truthy_values_skip_alignment(self, tmp_path, monkeypatch, truthy):
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        (hermes_home / "home").mkdir()
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setenv("HOME", "/home/host-user")
        monkeypatch.setenv("HERMES_PRESERVE_HOST_HOME", truthy)

        from hermes_constants import align_main_process_home_with_subprocess
        assert align_main_process_home_with_subprocess() is None
        assert os.environ["HOME"] == "/home/host-user"

    @pytest.mark.parametrize("falsy", ["0", "false", "no", "off", "", "   "])
    def test_falsy_values_still_align(self, tmp_path, monkeypatch, falsy):
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        profile_home = hermes_home / "home"
        profile_home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setenv("HOME", "/home/host-user")
        monkeypatch.setenv("HERMES_PRESERVE_HOST_HOME", falsy)

        from hermes_constants import align_main_process_home_with_subprocess
        assert align_main_process_home_with_subprocess() == str(profile_home)
        assert os.environ["HOME"] == str(profile_home)


# ---------------------------------------------------------------------------
# config.yaml opt-out — the documented surface, honoured before .env loads
# ---------------------------------------------------------------------------

class TestPreserveHostHomeConfigOptOut:
    """``profiles.preserve_host_home: true`` in config.yaml skips alignment.

    The opt-out must be read straight from disk so it takes effect at startup
    *before* ``.env`` is loaded (the env var alone is too late for a value set
    in ``.env`` — see the hermes-sweeper review on #27260).
    """

    def test_config_true_skips_alignment(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        (hermes_home / "home").mkdir()
        (hermes_home / "config.yaml").write_text(
            "profiles:\n  preserve_host_home: true\n", encoding="utf-8"
        )
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setenv("HOME", "/home/host-user")
        # The env var is NOT set — the config.yaml value alone must opt out.
        monkeypatch.delenv("HERMES_PRESERVE_HOST_HOME", raising=False)

        from hermes_constants import align_main_process_home_with_subprocess
        assert align_main_process_home_with_subprocess() is None
        assert os.environ["HOME"] == "/home/host-user"

    def test_config_false_still_aligns(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        profile_home = hermes_home / "home"
        profile_home.mkdir()
        (hermes_home / "config.yaml").write_text(
            "profiles:\n  preserve_host_home: false\n", encoding="utf-8"
        )
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setenv("HOME", "/home/host-user")
        monkeypatch.delenv("HERMES_PRESERVE_HOST_HOME", raising=False)

        from hermes_constants import align_main_process_home_with_subprocess
        assert align_main_process_home_with_subprocess() == str(profile_home)
        assert os.environ["HOME"] == str(profile_home)

    def test_malformed_config_does_not_block_alignment(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        profile_home = hermes_home / "home"
        profile_home.mkdir()
        (hermes_home / "config.yaml").write_text("profiles: [oops\n", encoding="utf-8")
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setenv("HOME", "/home/host-user")
        monkeypatch.delenv("HERMES_PRESERVE_HOST_HOME", raising=False)

        from hermes_constants import align_main_process_home_with_subprocess
        # A broken config must not wedge startup — alignment proceeds.
        assert align_main_process_home_with_subprocess() == str(profile_home)
        assert os.environ["HOME"] == str(profile_home)


# ---------------------------------------------------------------------------
# Profile isolation invariant
# ---------------------------------------------------------------------------

class TestProfileIsolation:
    """Different profiles still resolve to different main-process HOMEs."""

    def test_two_profiles_get_different_main_homes(self, tmp_path, monkeypatch):
        root = tmp_path / ".hermes" / "profiles"
        for name in ("alpha", "beta"):
            (root / name / "home").mkdir(parents=True)

        from hermes_constants import align_main_process_home_with_subprocess

        monkeypatch.setenv("HERMES_HOME", str(root / "alpha"))
        monkeypatch.setenv("HOME", "/home/host-user")
        home_a = align_main_process_home_with_subprocess()

        monkeypatch.setenv("HERMES_HOME", str(root / "beta"))
        home_b = align_main_process_home_with_subprocess()

        assert home_a != home_b
        assert home_a.endswith(os.path.join("alpha", "home"))
        assert home_b.endswith(os.path.join("beta", "home"))

    def test_main_home_equals_subprocess_home(self, tmp_path, monkeypatch):
        """The core invariant from issue #27250."""
        hermes_home = tmp_path / "data"
        hermes_home.mkdir()
        (hermes_home / "home").mkdir()
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setenv("HOME", "/opt/data")  # docker useradd -d value
        monkeypatch.delenv("HERMES_PRESERVE_HOST_HOME", raising=False)

        from hermes_constants import (
            align_main_process_home_with_subprocess,
            get_subprocess_home,
        )

        aligned = align_main_process_home_with_subprocess()
        sub_home = get_subprocess_home()

        assert aligned == sub_home
        assert os.environ["HOME"] == sub_home


# ---------------------------------------------------------------------------
# Subprocess env still gets the same HOME after main-process alignment
# ---------------------------------------------------------------------------

class TestSubprocessEnvStillAligned:
    """``_make_run_env``/``_sanitize_subprocess_env`` should agree with the aligned main HOME."""

    def test_make_run_env_matches_aligned_main_home(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        profile_home = hermes_home / "home"
        profile_home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setenv("HOME", "/opt/data")
        monkeypatch.setenv("PATH", "/usr/bin:/bin")
        monkeypatch.delenv("HERMES_PRESERVE_HOST_HOME", raising=False)

        from hermes_constants import align_main_process_home_with_subprocess
        from tools.environments.local import _make_run_env

        align_main_process_home_with_subprocess()
        run_env = _make_run_env({})

        assert os.environ["HOME"] == str(profile_home)
        assert run_env["HOME"] == str(profile_home)
        # Same value in main and subprocess view — that's the invariant.
        assert run_env["HOME"] == os.environ["HOME"]

    def test_sanitize_subprocess_env_matches_aligned_main_home(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        profile_home = hermes_home / "home"
        profile_home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setenv("HOME", "/opt/data")
        monkeypatch.delenv("HERMES_PRESERVE_HOST_HOME", raising=False)

        from hermes_constants import align_main_process_home_with_subprocess
        from tools.environments.local import _sanitize_subprocess_env

        align_main_process_home_with_subprocess()
        sanitized = _sanitize_subprocess_env({"HOME": "/opt/data", "PATH": "/usr/bin"})

        assert os.environ["HOME"] == str(profile_home)
        assert sanitized["HOME"] == str(profile_home)
        assert sanitized["HOME"] == os.environ["HOME"]

    def test_preserve_host_home_keeps_split_behaviour(self, tmp_path, monkeypatch):
        """With opt-out: main HOME stays host-level, subprocesses still get profile HOME."""
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        profile_home = hermes_home / "home"
        profile_home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setenv("HOME", "/opt/data")
        monkeypatch.setenv("PATH", "/usr/bin:/bin")
        monkeypatch.setenv("HERMES_PRESERVE_HOST_HOME", "1")

        from hermes_constants import align_main_process_home_with_subprocess
        from tools.environments.local import _make_run_env

        align_main_process_home_with_subprocess()
        run_env = _make_run_env({})

        assert os.environ["HOME"] == "/opt/data"
        assert run_env["HOME"] == str(profile_home)


# ---------------------------------------------------------------------------
# Path.home() agreement after alignment
# ---------------------------------------------------------------------------

class TestPathHomeAgreement:
    """After alignment, ``Path.home()`` and ``os.path.expanduser('~')`` should follow."""

    def test_path_home_follows_aligned_home_on_posix(self, tmp_path, monkeypatch):
        if os.name == "nt":  # pragma: no cover - posix-only check
            pytest.skip("POSIX-only path expansion semantics")

        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        profile_home = hermes_home / "home"
        profile_home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setenv("HOME", "/opt/data")
        monkeypatch.delenv("HERMES_PRESERVE_HOST_HOME", raising=False)

        from hermes_constants import align_main_process_home_with_subprocess

        align_main_process_home_with_subprocess()

        assert str(Path.home()) == str(profile_home)
        assert os.path.expanduser("~") == str(profile_home)
        assert os.path.expanduser("~/.ssh") == str(profile_home / ".ssh")


# ---------------------------------------------------------------------------
# Backward compat: get_subprocess_home() is still a pure read.
# ---------------------------------------------------------------------------

class TestGetSubprocessHomeStillPureRead:
    """``get_subprocess_home()`` must not mutate the environment by itself."""

    def test_does_not_modify_env(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        (hermes_home / "home").mkdir()
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setenv("HOME", "/opt/data")

        from hermes_constants import get_subprocess_home

        sub_home = get_subprocess_home()
        assert sub_home is not None
        # Read-only: main-process HOME is untouched until alignment is called.
        assert os.environ["HOME"] == "/opt/data"
