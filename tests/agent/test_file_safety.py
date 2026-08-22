"""Tests for agent/file_safety.py read guards — env file blocking.

Run with:  python -m pytest tests/agent/test_file_safety.py -v
"""

import os
from unittest.mock import patch

import pytest

from agent.file_safety import (
    _BLOCKED_PROJECT_ENV_BASENAMES,
    get_read_block_error,
)


# ---------------------------------------------------------------------------
# Project-local .env file blocking (issue #20734)
# ---------------------------------------------------------------------------


class TestEnvFileReadBlocking:
    """Secret-bearing .env files must be blocked by get_read_block_error."""

    @pytest.mark.parametrize("basename", [
        ".env",
        ".env.local",
        ".env.development",
        ".env.production",
        ".env.test",
        ".env.staging",
        ".envrc",
    ])
    def test_blocked_env_basenames(self, basename):
        """All secret-bearing .env basenames are blocked regardless of directory."""
        path = f"/tmp/project/{basename}"
        error = get_read_block_error(path)
        assert error is not None, f"{basename} should be blocked"
        assert "Access denied" in error
        assert "secret-bearing" in error.lower() or "environment file" in error.lower()


    @pytest.mark.parametrize("basename", [
        ".ENV",
        ".Env.Local",
        ".ENV.PRODUCTION",
        ".ENVRC",
    ])
    def test_blocked_env_basenames_case_insensitive(self, basename):
        """Secret-bearing .env basenames are blocked regardless of case."""
        error = get_read_block_error(f"/tmp/project/{basename}")
        assert error is not None, f"{basename} should be blocked"
        assert "Access denied" in error
        assert "environment file" in error.lower()


    def test_allowed_env_example(self):
        """"The .env.example file is explicitly allowed — it's documentation, not a secret."""
        error = get_read_block_error("/tmp/project/.env.example")
        assert error is None






# ---------------------------------------------------------------------------
# Existing cache-file blocking (regression — must still work)
# ---------------------------------------------------------------------------


class TestCacheFileReadBlocking:
    """Internal Hermes cache files must remain blocked."""

    def test_hub_index_cache_blocked(self, tmp_path):
        """Hub index-cache reads are blocked."""
        hermes_home = tmp_path / ".hermes"
        cache = hermes_home / "skills" / ".hub" / "index-cache" / "data.json"
        cache.parent.mkdir(parents=True)
        cache.write_text("{}")

        with patch("agent.file_safety._hermes_home_path", return_value=hermes_home):
            error = get_read_block_error(str(cache))
            assert error is not None
            assert "internal Hermes cache" in error

    def test_hub_directory_blocked(self, tmp_path):
        """Hub directory reads are blocked."""
        hermes_home = tmp_path / ".hermes"
        hub = hermes_home / "skills" / ".hub" / "metadata.json"
        hub.parent.mkdir(parents=True)
        hub.write_text("{}")

        with patch("agent.file_safety._hermes_home_path", return_value=hermes_home):
            error = get_read_block_error(str(hub))
            assert error is not None


# ---------------------------------------------------------------------------
# Combined: env guard + cache guard don't interfere
# ---------------------------------------------------------------------------


class TestCombinedGuards:
    """Both guards should work independently without interference."""

    def test_env_guard_works_regardless_of_hermes_home(self, tmp_path):
        """The env basename guard does not depend on HERMES_HOME resolution."""
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()

        with patch("agent.file_safety._hermes_home_path", return_value=hermes_home):
            # Regular project .env should still be blocked
            error = get_read_block_error("/workspace/.env")
            assert error is not None

            # .env.example should still be allowed
            error = get_read_block_error("/workspace/.env.example")
            assert error is None

    def test_cache_guard_still_works_with_env_guard(self, tmp_path):
        """Cache file blocking still works when env guard is active."""
        hermes_home = tmp_path / ".hermes"
        cache = hermes_home / "skills" / ".hub" / "index-cache" / "x"
        cache.parent.mkdir(parents=True)
        cache.write_text("")

        with patch("agent.file_safety._hermes_home_path", return_value=hermes_home):
            error = get_read_block_error(str(cache))
            assert error is not None
            assert "internal Hermes cache" in error


# ---------------------------------------------------------------------------
# Kernel character-device sinks — always writable, even outside safe root
# ---------------------------------------------------------------------------


class TestKernelSinkAllowlist:
    """Kernel character-device sinks (/dev/null etc.) must always be writable.

    Writing to /dev/null and its siblings is an OS-level no-op used routinely
    for shell redirection. Blocking them under HERMES_WRITE_SAFE_ROOT produces
    false-positive denials for legitimate patterns. Regression guard for the
    original report where writes to /dev/null failed with "outside
    HERMES_WRITE_SAFE_ROOT".
    """

    ALLOWED_SINKS = [
        "/dev/null",
        "/dev/stdout",
        "/dev/stderr",
        "/dev/tty",
        "/dev/zero",
    ]

    @pytest.mark.parametrize("sink", ALLOWED_SINKS)
    def test_kernel_sink_allowed_under_mismatched_safe_root(self, sink, monkeypatch):
        """Every allowlisted sink must return None (allowed) even when
        HERMES_WRITE_SAFE_ROOT points somewhere else entirely."""
        from agent.file_safety import _classify_write_denial

        monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", "/tmp/safe")
        assert _classify_write_denial(sink) is None

    def test_non_allowlisted_dev_path_still_rejected(self, monkeypatch):
        """A /dev/... path NOT on the allowlist (e.g. /dev/random) still
        goes through the normal safe-root check."""
        from agent.file_safety import _classify_write_denial

        monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", "/tmp/safe")
        # /dev/random is not on the allowlist and is not inside /tmp/safe
        assert _classify_write_denial("/dev/random") == "safe_root"

    def test_credential_path_still_rejected(self, monkeypatch, tmp_path):
        """Adding the kernel-sink short-circuit must NOT weaken credential
        denial for non-allowlisted paths."""
        from agent.file_safety import _classify_write_denial

        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", str(tmp_path))
        ssh_key = tmp_path / ".ssh" / "id_ed25519"
        ssh_key.parent.mkdir(parents=True)
        ssh_key.write_text("")
        assert _classify_write_denial(str(ssh_key)) == "credential"

    def test_out_of_tree_path_still_rejected(self, monkeypatch, tmp_path):
        """A genuinely out-of-safe-root path (not on the sink allowlist)
        must still be rejected as safe_root."""
        from agent.file_safety import _classify_write_denial

        safe = tmp_path / "safe"
        safe.mkdir()
        elsewhere = tmp_path / "elsewhere" / "foo.txt"
        elsewhere.parent.mkdir()
        monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", str(safe))
        assert _classify_write_denial(str(elsewhere)) == "safe_root"

    def test_symlink_to_dev_null_not_bypassed(self, monkeypatch, tmp_path):
        """A symlink whose target is /dev/null must NOT ride the allowlist —
        the check is on the user-supplied path's abspath, not its realpath,
        precisely so this attack shape is blocked.

        Skips gracefully on platforms without symlink support.
        """
        from agent.file_safety import _classify_write_denial

        symlink = tmp_path / "evil"
        try:
            symlink.symlink_to("/dev/null")
        except (OSError, NotImplementedError):
            pytest.skip("symlink creation not supported on this platform")

        safe = tmp_path / "safe"
        safe.mkdir()
        monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", str(safe))
        # The symlink itself is NOT literally "/dev/null" — it's tmp_path/evil.
        # Since it lives outside HERMES_WRITE_SAFE_ROOT, it must be rejected.
        assert _classify_write_denial(str(symlink)) == "safe_root"

    def test_is_kernel_sink_helper_recognizes_normalized_variants(self):
        """The helper must recognize path variants that normalize to
        an allowlisted sink (e.g. /dev/./null)."""
        from agent.file_safety import _is_kernel_sink

        assert _is_kernel_sink("/dev/null") is True
        assert _is_kernel_sink("/dev/./null") is True
        assert _is_kernel_sink("/dev/random") is False
        assert _is_kernel_sink("/tmp/null") is False

    def test_is_kernel_sink_handles_bad_input(self):
        """The helper must return False (not raise) for empty or bogus paths."""
        from agent.file_safety import _is_kernel_sink

        assert _is_kernel_sink("") is False
        # Non-existent parent components are fine — no filesystem is touched
        assert _is_kernel_sink("/nonexistent/random/thing") is False
