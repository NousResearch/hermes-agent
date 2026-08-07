"""Tests for cronjob-create script-path error messages (#59599).

Validates the user-facing guidance produced by
``tools.cronjob_tools._validate_cron_script_path`` and the public
``cronjob`` tool wrapper. The old "absolute or home-relative path" message
left users guessing; this suite locks in the new WHY/HOW structure for
every error class the validator emits:

  1. Absolute paths (POSIX root, Windows drive, UNC, ``~/...``)
  2. Path-traversal escapes (``../../etc/passwd``)

Plus the happy path — a valid relative script — to make sure the
friendlier messages don't accidentally reject what used to work.

Note: we deliberately do NOT block at create time on a missing relative
script. The runtime ``_run_job_script`` surfaces a clear "not found"
error on first fire, and the previous behaviour intentionally allowed
scheduling a job whose script is written later. See
``test_cron_script.TestRunJobScript.test_script_not_found`` for the
runtime-side coverage.
"""

import json
import sys
from pathlib import Path

import pytest

# Make project root importable (same pattern as the other cron tests).
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Cross-platform fixtures
# ---------------------------------------------------------------------------

POSIX_ABS = "/home/user/.hermes/scripts/foo.py"
HOME_REL = "~/.hermes/scripts/foo.py"
WIN_DRIVE = "C:\\Users\\stooovie\\.hermes\\scripts\\foo.py"
WIN_UNC = "\\\\server\\share\\scripts\\foo.py"


@pytest.fixture
def cron_env(tmp_path, monkeypatch):
    """Isolated cron environment with temp HERMES_HOME + scripts dir."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "cron").mkdir()
    (hermes_home / "cron" / "output").mkdir()
    (hermes_home / "scripts").mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    # Cache-bust module-level paths the same way test_cron_script.py does.
    import cron.jobs as jobs_mod
    monkeypatch.setattr(jobs_mod, "HERMES_DIR", hermes_home)
    monkeypatch.setattr(jobs_mod, "CRON_DIR", hermes_home / "cron")
    monkeypatch.setattr(jobs_mod, "JOBS_FILE", hermes_home / "cron" / "jobs.json")
    monkeypatch.setattr(jobs_mod, "OUTPUT_DIR", hermes_home / "cron" / "output")
    return hermes_home


# ---------------------------------------------------------------------------
# _is_cron_script_absolute — the cross-platform detector
# ---------------------------------------------------------------------------

class TestIsAbsoluteCrossPlatform:
    """The validator's cross-platform absolute-path detector.

    Why not just ``pathlib.Path.is_absolute()``? Because on Windows
    ``Path("/usr/local/bin/x.py").is_absolute()`` returns ``False`` —
    ``Path`` is parameterized to the running OS, so a POSIX-shaped
    string is treated as relative even though it absolutely is not for
    our purposes (cron scripts are written by humans on either OS and
    pasted across machines). The detector ORs the cross-platform
    POSIX/UNC/drive-letter heuristics with the local ``Path`` check
    plus an explicit ``~`` prefix so every shape is caught on every OS.
    """

    def test_posix_root_is_absolute(self):
        from tools.cronjob_tools import _is_cron_script_absolute
        assert _is_cron_script_absolute("/usr/local/bin/x.py") is True

    def test_home_relative_is_absolute(self):
        from tools.cronjob_tools import _is_cron_script_absolute
        assert _is_cron_script_absolute("~/scripts/x.py") is True

    def test_plain_relative_is_not_absolute(self):
        from tools.cronjob_tools import _is_cron_script_absolute
        assert _is_cron_script_absolute("scripts/x.py") is False

    def test_bare_filename_is_not_absolute(self):
        from tools.cronjob_tools import _is_cron_script_absolute
        assert _is_cron_script_absolute("foo.py") is False

    def test_empty_string_is_not_absolute(self):
        from tools.cronjob_tools import _is_cron_script_absolute
        assert _is_cron_script_absolute("") is False

    def test_dot_relative_is_not_absolute(self):
        from tools.cronjob_tools import _is_cron_script_absolute
        assert _is_cron_script_absolute("./foo.py") is False
        assert _is_cron_script_absolute("../foo.py") is False

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows-specific path shape")
    def test_windows_drive_letter_is_absolute(self):
        from tools.cronjob_tools import _is_cron_script_absolute
        # Drive letter + path → pathlib reports True on Windows.
        assert _is_cron_script_absolute("C:\\Users\\me\\x.py") is True

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows-specific path shape")
    def test_windows_unc_is_absolute(self):
        from tools.cronjob_tools import _is_cron_script_absolute
        assert _is_cron_script_absolute(WIN_UNC) is True

    def test_drive_letter_caught_on_any_platform(self):
        """Even on POSIX, a Windows-style path should still be rejected —
        users paste them across machines."""
        from tools.cronjob_tools import _is_cron_script_absolute
        # On any platform, "C:\\foo" is a drive-letter path and the
        # validator must flag it as absolute so the user sees the
        # security/policy error rather than a confusing "not found".
        assert _is_cron_script_absolute(WIN_DRIVE) is True

    def test_unc_caught_on_any_platform(self):
        from tools.cronjob_tools import _is_cron_script_absolute
        assert _is_cron_script_absolute(WIN_UNC) is True


# ---------------------------------------------------------------------------
# _validate_cron_script_path — direct unit tests on the error strings
# ---------------------------------------------------------------------------

class TestAbsolutePathError:
    """Absolute paths must produce a multi-line error with WHY + HOW."""

    def test_posix_absolute_path_blocked_with_rationale(self, cron_env):
        from tools.cronjob_tools import _validate_cron_script_path

        err = _validate_cron_script_path(POSIX_ABS)
        assert err is not None
        # Multi-line structure: WHY + HOW blocks
        assert "\n" in err, "error should be multi-line for actionability"
        # Headline echoes the policy
        assert "relative to" in err
        # Security rationale
        assert "Why:" in err
        assert "prompt-injected" in err or "prompt injection" in err.lower()
        # Actionable fix
        assert "How to fix:" in err
        assert "--script foo.py" in err
        # Platform-appropriate verify command
        expected_list_cmd = "dir" if sys.platform == "win32" else "ls -la"
        assert expected_list_cmd in err
        # Echo the rejected path back so the user can spot the typo
        assert repr(POSIX_ABS) in err

    def test_home_relative_path_blocked_with_rationale(self, cron_env):
        from tools.cronjob_tools import _validate_cron_script_path

        err = _validate_cron_script_path(HOME_REL)
        assert err is not None
        assert "Why:" in err
        assert "How to fix:" in err
        assert repr(HOME_REL) in err

    def test_windows_absolute_path_blocked_with_rationale(self, cron_env):
        from tools.cronjob_tools import _validate_cron_script_path

        err = _validate_cron_script_path(WIN_DRIVE)
        assert err is not None
        assert "Why:" in err
        assert "How to fix:" in err
        # The error embeds the rejected path via repr, so compare against
        # the repr form (which doubles every backslash) rather than the
        # raw Python-string literal.
        assert repr(WIN_DRIVE) in err


class TestTraversalError:
    """Paths that escape the scripts dir via '..' get the friendly treatment."""

    def test_traversal_blocked_with_rationale(self, cron_env):
        from tools.cronjob_tools import _validate_cron_script_path

        err = _validate_cron_script_path("../../etc/passwd")
        assert err is not None
        assert "Why:" in err
        assert "How to fix:" in err
        assert ".." in err
        # The original traversal message phrase is preserved (existing
        # downstream tests assert on it), but the new copy lives next to
        # it so the user sees both.
        assert "escapes the scripts directory" in err


class TestValidRelativePath:
    """The friendly errors must NOT regress the happy path.

    The validator deliberately does NOT block on a missing file (the
    runtime layer already reports "not found" on first fire, and the
    previous behaviour intentionally allowed scheduling a job whose
    script is written later). So a relative path that resolves within
    the scripts dir is accepted regardless of whether the file exists.
    """

    def test_existing_relative_path_passes(self, cron_env):
        from tools.cronjob_tools import _validate_cron_script_path

        script = cron_env / "scripts" / "monitor.py"
        script.write_text('print("hi")\n')

        err = _validate_cron_script_path("monitor.py")
        assert err is None

    def test_missing_relative_path_still_allowed(self, cron_env):
        """Backwards-compat: scheduling a job whose script is written
        later must still succeed at create time."""
        from tools.cronjob_tools import _validate_cron_script_path

        # No file exists, but the path is well-formed and contained.
        err = _validate_cron_script_path("future_script.py")
        assert err is None

    def test_existing_subdirectory_relative_path_passes(self, cron_env):
        from tools.cronjob_tools import _validate_cron_script_path

        subdir = cron_env / "scripts" / "monitors"
        subdir.mkdir()
        (subdir / "check.py").write_text('print("ok")\n')

        err = _validate_cron_script_path("monitors/check.py")
        assert err is None

    def test_empty_script_clears_field(self, cron_env):
        from tools.cronjob_tools import _validate_cron_script_path

        # Empty / whitespace → clearing the field, always allowed.
        assert _validate_cron_script_path("") is None
        assert _validate_cron_script_path("   ") is None
        assert _validate_cron_script_path(None) is None


# ---------------------------------------------------------------------------
# End-to-end through the cronjob() tool — what the user actually sees
# ---------------------------------------------------------------------------

class TestCronjobToolErrorMessages:
    """Same checks via the public cronjob() API boundary, since that's
    where the error reaches the LLM and the CLI."""

    def _create(self, monkeypatch, script):
        monkeypatch.setenv("HERMES_INTERACTIVE", "1")
        from tools.cronjob_tools import cronjob
        return json.loads(cronjob(
            action="create",
            schedule="every 1h",
            prompt="Monitor things",
            script=script,
        ))

    def test_absolute_path_via_tool_shows_friendly_error(self, cron_env, monkeypatch):
        result = self._create(monkeypatch, POSIX_ABS)
        assert result["success"] is False
        err = result["error"]
        # Multi-line + WHY + HOW + verify-command are all surfaced
        assert "\n" in err
        assert "Why:" in err
        assert "How to fix:" in err
        assert ("ls -la" in err) or ("dir" in err)

    def test_home_relative_path_via_tool_shows_friendly_error(self, cron_env, monkeypatch):
        result = self._create(monkeypatch, HOME_REL)
        assert result["success"] is False
        assert "Why:" in result["error"]
        assert "How to fix:" in result["error"]

    def test_traversal_via_tool_shows_friendly_error(self, cron_env, monkeypatch):
        result = self._create(monkeypatch, "../../etc/passwd")
        assert result["success"] is False
        assert "Why:" in result["error"]
        assert "How to fix:" in result["error"]

    def test_valid_relative_path_via_tool_still_succeeds(self, cron_env, monkeypatch):
        # Real file under scripts/ → success, no regression.
        (cron_env / "scripts" / "monitor.py").write_text('print("hi")\n')
        result = self._create(monkeypatch, "monitor.py")
        assert result["success"] is True
        assert result["job"]["script"] == "monitor.py"

    def test_windows_absolute_path_via_tool_shows_friendly_error(self, cron_env, monkeypatch):
        """Drive-letter absolute path rejected cross-platform."""
        result = self._create(monkeypatch, WIN_DRIVE)
        assert result["success"] is False
        err = result["error"]
        assert "Why:" in err
        assert "How to fix:" in err


# ---------------------------------------------------------------------------
# Backwards-compat: the existing one-liner keyword still appears
# ---------------------------------------------------------------------------

class TestLegacyKeywordCompatibility:
    """Downstream consumers (and older tests) look for the substring
    'relative' / 'absolute' in the error message. The new multi-line
    output keeps that surface so existing tests stay green."""

    def test_absolute_error_mentions_relative(self, cron_env):
        from tools.cronjob_tools import _validate_cron_script_path

        err = _validate_cron_script_path(POSIX_ABS) or ""
        assert "relative" in err.lower()
        assert "absolute" in err.lower() or "home-relative" in err.lower()

    def test_home_relative_error_mentions_relative(self, cron_env):
        from tools.cronjob_tools import _validate_cron_script_path

        err = _validate_cron_script_path(HOME_REL) or ""
        assert "relative" in err.lower()