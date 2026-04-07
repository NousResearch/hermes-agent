"""Tests for Matrix E2EE setup fallback in hermes_cli/setup.py."""

from __future__ import annotations

import subprocess
import sys
from unittest.mock import patch

import pytest


@pytest.fixture
def mock_env(monkeypatch, tmp_path):
    """Redirect env file writes to a temp dir so tests don't touch real config."""
    env_file = tmp_path / ".env"
    env_file.touch()
    monkeypatch.setattr("hermes_cli.setup.get_env_value", lambda key: None)
    saved = {}

    def _save(key, value):
        saved[key] = value

    monkeypatch.setattr("hermes_cli.setup.save_env_value", _save)
    return saved


def _make_result(returncode, stderr=""):
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout="", stderr=stderr)


def _nio_import_raises():
    """Context manager that makes __import__('nio') raise ImportError."""
    real_import = __import__

    def _selective_import(name, *args, **kwargs):
        if name == "nio":
            raise ImportError("No module named 'nio'")
        return real_import(name, *args, **kwargs)

    return patch("builtins.__import__", side_effect=_selective_import)


class TestMatrixE2EEFallback:
    """When matrix-nio[e2e] install fails, setup should offer a plain fallback."""

    def test_e2ee_install_fails_olm_offers_and_accepts_fallback(self, monkeypatch, mock_env):
        """If E2EE install fails with olm error, offer plain fallback; user accepts."""
        from hermes_cli import setup as setup_mod

        prompt_answers = iter([
            "https://matrix.example.org",  # homeserver URL
            "syt_fake_token",              # access token
            "",                            # user_id (optional, skip)
            "",                            # allowed users (skip)
            "",                            # home room (skip)
        ])
        yes_no_answers = iter([True, True])  # want E2EE=yes, accept fallback=yes

        monkeypatch.setattr(setup_mod, "prompt", lambda *a, **kw: next(prompt_answers))
        monkeypatch.setattr(setup_mod, "prompt_yes_no", lambda *a, **kw: next(yes_no_answers))

        call_log = []

        def _mock_run(cmd, **kwargs):
            call_log.append(cmd)
            if len(call_log) == 1:
                return _make_result(1, stderr="error: python-olm build failed: make static returned 2")
            return _make_result(0)

        monkeypatch.setattr(subprocess, "run", _mock_run)
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/uv" if name == "uv" else None)

        with _nio_import_raises():
            setup_mod._setup_matrix()

        # Should have flipped encryption to false after fallback succeeded
        assert mock_env.get("MATRIX_ENCRYPTION") == "false"
        assert len(call_log) == 2
        assert "matrix-nio[e2e]" in str(call_log[0])
        assert "matrix-nio" in str(call_log[1])
        assert "[e2e]" not in str(call_log[1])

    def test_e2ee_install_succeeds_no_fallback(self, monkeypatch, mock_env):
        """If E2EE install succeeds on first try, no fallback needed."""
        from hermes_cli import setup as setup_mod

        prompt_answers = iter([
            "https://matrix.example.org",
            "syt_fake_token",
            "",   # user_id
            "",   # allowed users
            "",   # home room
        ])
        yes_no_answers = iter([True])  # want E2EE=yes

        monkeypatch.setattr(setup_mod, "prompt", lambda *a, **kw: next(prompt_answers))
        monkeypatch.setattr(setup_mod, "prompt_yes_no", lambda *a, **kw: next(yes_no_answers))

        call_log = []

        def _mock_run(cmd, **kwargs):
            call_log.append(cmd)
            return _make_result(0)

        monkeypatch.setattr(subprocess, "run", _mock_run)
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/uv" if name == "uv" else None)

        with _nio_import_raises():
            setup_mod._setup_matrix()

        assert mock_env.get("MATRIX_ENCRYPTION") == "true"
        assert len(call_log) == 1

    def test_non_e2ee_install_fails_no_fallback_offered(self, monkeypatch, mock_env):
        """If user chose no E2EE and plain install fails, just show error."""
        from hermes_cli import setup as setup_mod

        prompt_answers = iter([
            "https://matrix.example.org",
            "syt_fake_token",
            "",   # user_id
            "",   # allowed users
            "",   # home room
        ])
        yes_no_answers = iter([False])  # want E2EE=no

        monkeypatch.setattr(setup_mod, "prompt", lambda *a, **kw: next(prompt_answers))
        monkeypatch.setattr(setup_mod, "prompt_yes_no", lambda *a, **kw: next(yes_no_answers))

        call_log = []

        def _mock_run(cmd, **kwargs):
            call_log.append(cmd)
            return _make_result(1, stderr="some pip error")

        monkeypatch.setattr(subprocess, "run", _mock_run)
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/uv" if name == "uv" else None)

        with _nio_import_raises():
            setup_mod._setup_matrix()

        # No MATRIX_ENCRYPTION saved (user said no to E2EE)
        assert "MATRIX_ENCRYPTION" not in mock_env
        assert len(call_log) == 1
        assert "[e2e]" not in str(call_log[0])

    def test_e2ee_fails_user_declines_fallback_disables_encryption(self, monkeypatch, mock_env):
        """If E2EE install fails and user declines fallback, MATRIX_ENCRYPTION must be false."""
        from hermes_cli import setup as setup_mod

        prompt_answers = iter([
            "https://matrix.example.org",
            "syt_fake_token",
            "",   # user_id
            "",   # allowed users
            "",   # home room
        ])
        yes_no_answers = iter([True, False])  # want E2EE=yes, decline fallback=no

        monkeypatch.setattr(setup_mod, "prompt", lambda *a, **kw: next(prompt_answers))
        monkeypatch.setattr(setup_mod, "prompt_yes_no", lambda *a, **kw: next(yes_no_answers))

        call_log = []

        def _mock_run(cmd, **kwargs):
            call_log.append(cmd)
            return _make_result(1, stderr="python-olm failed to build")

        monkeypatch.setattr(subprocess, "run", _mock_run)
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/uv" if name == "uv" else None)

        with _nio_import_raises():
            setup_mod._setup_matrix()

        # E2EE was initially set to true but must be flipped to false
        assert mock_env.get("MATRIX_ENCRYPTION") == "false"
        # Only 1 subprocess call (the failed e2e attempt)
        assert len(call_log) == 1

    def test_e2ee_fails_fallback_also_fails_disables_encryption(self, monkeypatch, mock_env):
        """If both E2EE and plain fallback install fail, MATRIX_ENCRYPTION must be false."""
        from hermes_cli import setup as setup_mod

        prompt_answers = iter([
            "https://matrix.example.org",
            "syt_fake_token",
            "",   # user_id
            "",   # allowed users
            "",   # home room
        ])
        yes_no_answers = iter([True, True])  # want E2EE=yes, accept fallback=yes

        monkeypatch.setattr(setup_mod, "prompt", lambda *a, **kw: next(prompt_answers))
        monkeypatch.setattr(setup_mod, "prompt_yes_no", lambda *a, **kw: next(yes_no_answers))

        call_log = []

        def _mock_run(cmd, **kwargs):
            call_log.append(cmd)
            # Both attempts fail
            if len(call_log) == 1:
                return _make_result(1, stderr="python-olm build failed")
            return _make_result(1, stderr="pip install matrix-nio also failed")

        monkeypatch.setattr(subprocess, "run", _mock_run)
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/uv" if name == "uv" else None)

        with _nio_import_raises():
            setup_mod._setup_matrix()

        # Encryption must be disabled even when everything fails
        assert mock_env.get("MATRIX_ENCRYPTION") == "false"
        assert len(call_log) == 2

    def test_e2ee_fails_empty_stderr(self, monkeypatch, mock_env):
        """If E2EE install fails with empty stderr, show generic message."""
        from hermes_cli import setup as setup_mod

        prompt_answers = iter([
            "https://matrix.example.org",
            "syt_fake_token",
            "",   # user_id
            "",   # allowed users
            "",   # home room
        ])
        yes_no_answers = iter([True, True])  # want E2EE=yes, accept fallback=yes

        monkeypatch.setattr(setup_mod, "prompt", lambda *a, **kw: next(prompt_answers))
        monkeypatch.setattr(setup_mod, "prompt_yes_no", lambda *a, **kw: next(yes_no_answers))

        call_log = []

        def _mock_run(cmd, **kwargs):
            call_log.append(cmd)
            if len(call_log) == 1:
                return _make_result(1, stderr="")  # empty stderr
            return _make_result(0)

        monkeypatch.setattr(subprocess, "run", _mock_run)
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/uv" if name == "uv" else None)

        with _nio_import_raises():
            setup_mod._setup_matrix()

        # Fallback should still work even with empty stderr
        assert mock_env.get("MATRIX_ENCRYPTION") == "false"
        assert len(call_log) == 2
