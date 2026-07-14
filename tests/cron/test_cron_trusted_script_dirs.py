"""Tests for cron.trusted_script_dirs config feature.

When ``cron.trusted_script_dirs`` is set in config.yaml, symlinks in
``~/.hermes/scripts/`` that resolve into any trusted directory are allowed
to execute — without weakening path-traversal protection for untrusted paths.
"""

import json
import os
import sys
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@pytest.fixture
def cron_env(tmp_path, monkeypatch):
    """Isolated cron environment with temp HERMES_HOME."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "cron").mkdir()
    (hermes_home / "cron" / "output").mkdir()
    (hermes_home / "scripts").mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    import cron.jobs as jobs_mod
    monkeypatch.setattr(jobs_mod, "HERMES_DIR", hermes_home)
    monkeypatch.setattr(jobs_mod, "CRON_DIR", hermes_home / "cron")
    monkeypatch.setattr(jobs_mod, "JOBS_FILE", hermes_home / "cron" / "jobs.json")
    monkeypatch.setattr(jobs_mod, "OUTPUT_DIR", hermes_home / "cron" / "output")

    return hermes_home


@pytest.fixture
def trusted_dir(tmp_path):
    """A directory outside HERMES_HOME that will be trusted."""
    d = tmp_path / "inflection" / "projects" / "scripts"
    d.mkdir(parents=True)
    return d


@pytest.fixture
def untrusted_dir(tmp_path):
    """A directory outside HERMES_HOME that is NOT trusted."""
    d = tmp_path / "untrusted"
    d.mkdir(parents=True)
    return d


def _mock_config_with_trusted_dirs(dirs: list[str]):
    """Return a mock load_config that includes cron.trusted_script_dirs."""
    def _fake_load_config():
        return {"cron": {"trusted_script_dirs": dirs}}
    return _fake_load_config


# ---------------------------------------------------------------------------
# _get_trusted_script_dirs
# ---------------------------------------------------------------------------

class TestGetTrustedScriptDirs:
    """Unit tests for _get_trusted_script_dirs()."""

    def test_empty_config(self, cron_env):
        from cron.scheduler import _get_trusted_script_dirs
        with patch("cron.scheduler.load_config", return_value={}):
            assert _get_trusted_script_dirs() == []

    def test_no_cron_section(self, cron_env):
        from cron.scheduler import _get_trusted_script_dirs
        with patch("cron.scheduler.load_config", return_value={"model": {}}):
            assert _get_trusted_script_dirs() == []

    def test_returns_resolved_paths(self, cron_env, trusted_dir):
        from cron.scheduler import _get_trusted_script_dirs
        fake = _mock_config_with_trusted_dirs([str(trusted_dir)])
        with patch("cron.scheduler.load_config", fake):
            result = _get_trusted_script_dirs()
            assert len(result) == 1
            assert result[0] == trusted_dir.resolve()

    def test_skips_nonexistent_dirs(self, cron_env, tmp_path):
        from cron.scheduler import _get_trusted_script_dirs
        fake = _mock_config_with_trusted_dirs([str(tmp_path / "does_not_exist")])
        with patch("cron.scheduler.load_config", fake):
            assert _get_trusted_script_dirs() == []

    def test_skips_non_string_entries(self, cron_env, trusted_dir):
        from cron.scheduler import _get_trusted_script_dirs
        def _fake():
            return {"cron": {"trusted_script_dirs": [str(trusted_dir), 42, None, True]}}
        with patch("cron.scheduler.load_config", _fake):
            result = _get_trusted_script_dirs()
            assert len(result) == 1

    def test_handles_non_list_gracefully(self, cron_env):
        from cron.scheduler import _get_trusted_script_dirs
        def _fake():
            return {"cron": {"trusted_script_dirs": "/just/a/string"}}
        with patch("cron.scheduler.load_config", _fake):
            assert _get_trusted_script_dirs() == []

    def test_handles_load_config_exception(self, cron_env):
        from cron.scheduler import _get_trusted_script_dirs
        with patch("cron.scheduler.load_config", side_effect=RuntimeError("boom")):
            assert _get_trusted_script_dirs() == []

    def test_tilde_expanded(self, cron_env, monkeypatch, tmp_path):
        from cron.scheduler import _get_trusted_script_dirs
        # Create a dir under a fake HOME
        fake_home = tmp_path / "fakehome"
        fake_home.mkdir()
        target = fake_home / "myproject"
        target.mkdir()
        monkeypatch.setenv("HOME", str(fake_home))
        fake = _mock_config_with_trusted_dirs(["~/myproject"])
        with patch("cron.scheduler.load_config", fake):
            result = _get_trusted_script_dirs()
            assert len(result) == 1
            assert result[0] == target.resolve()


# ---------------------------------------------------------------------------
# _run_job_script with trusted_script_dirs
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Symlinks require elevated privileges on Windows",
)
class TestSymlinkToTrustedDir:
    """Symlinks in scripts/ pointing into a trusted dir should be allowed."""

    def test_symlink_to_trusted_dir_allowed(self, cron_env, trusted_dir):
        from cron.scheduler import _run_job_script

        # Create a real script in the trusted dir
        real_script = trusted_dir / "check.py"
        real_script.write_text('print("trusted ok")\n')

        # Symlink from scripts/ into the trusted dir
        link = cron_env / "scripts" / "check.py"
        link.symlink_to(real_script)

        fake = _mock_config_with_trusted_dirs([str(trusted_dir)])
        with patch("cron.scheduler.load_config", fake):
            success, output = _run_job_script("check.py")
            assert success is True
            assert output == "trusted ok"

    def test_symlink_to_untrusted_dir_still_blocked(self, cron_env, untrusted_dir):
        from cron.scheduler import _run_job_script

        real_script = untrusted_dir / "evil.py"
        real_script.write_text('print("should not run")\n')

        link = cron_env / "scripts" / "evil.py"
        link.symlink_to(real_script)

        # Trust a DIFFERENT dir, not untrusted_dir
        fake = _mock_config_with_trusted_dirs(["/some/other/dir"])
        with patch("cron.scheduler.load_config", fake):
            success, output = _run_job_script("evil.py")
            assert success is False
            assert "blocked" in output.lower()

    def test_symlink_with_no_trusted_dirs_blocked(self, cron_env, trusted_dir):
        """Without trusted_script_dirs config, symlinks outside are blocked."""
        from cron.scheduler import _run_job_script

        real_script = trusted_dir / "check.py"
        real_script.write_text('print("should not run")\n')

        link = cron_env / "scripts" / "check.py"
        link.symlink_to(real_script)

        with patch("cron.scheduler.load_config", return_value={}):
            success, output = _run_job_script("check.py")
            assert success is False
            assert "blocked" in output.lower()

    def test_symlink_to_subdir_of_trusted_dir(self, cron_env, trusted_dir):
        """Scripts in subdirectories of a trusted dir should also be allowed."""
        from cron.scheduler import _run_job_script

        subdir = trusted_dir / "monitors" / "health"
        subdir.mkdir(parents=True)
        real_script = subdir / "deep.py"
        real_script.write_text('print("deep ok")\n')

        link = cron_env / "scripts" / "deep.py"
        link.symlink_to(real_script)

        # Trust the top-level dir — subdirs should be included
        fake = _mock_config_with_trusted_dirs([str(trusted_dir.parent.parent)])
        with patch("cron.scheduler.load_config", fake):
            success, output = _run_job_script("deep.py")
            assert success is True
            assert output == "deep ok"

    def test_multiple_trusted_dirs(self, cron_env, tmp_path):
        """Scripts can be in any of multiple trusted dirs."""
        from cron.scheduler import _run_job_script

        dir_a = tmp_path / "inflection"
        dir_a.mkdir()
        dir_b = tmp_path / "meditation"
        dir_b.mkdir()

        script_a = dir_a / "a.py"
        script_a.write_text('print("from inflection")\n')
        script_b = dir_b / "b.py"
        script_b.write_text('print("from meditation")\n')

        (cron_env / "scripts" / "a.py").symlink_to(script_a)
        (cron_env / "scripts" / "b.py").symlink_to(script_b)

        fake = _mock_config_with_trusted_dirs([str(dir_a), str(dir_b)])
        with patch("cron.scheduler.load_config", fake):
            ok_a, out_a = _run_job_script("a.py")
            assert ok_a is True
            assert out_a == "from inflection"

            ok_b, out_b = _run_job_script("b.py")
            assert ok_b is True
            assert out_b == "from meditation"


# ---------------------------------------------------------------------------
# Security: trusted dirs must NOT weaken other protections
# ---------------------------------------------------------------------------

class TestTrustedDirsSecurityInvariants:
    """Trusted dirs must not open path-traversal or injection holes."""

    def test_traversal_still_blocked_with_trusted_dirs(self, cron_env, trusted_dir):
        from cron.scheduler import _run_job_script

        fake = _mock_config_with_trusted_dirs([str(trusted_dir)])
        with patch("cron.scheduler.load_config", fake):
            success, output = _run_job_script("../../etc/passwd")
            assert success is False
            assert "blocked" in output.lower() or "not found" in output.lower()

    def test_absolute_outside_all_dirs_still_blocked(self, cron_env, trusted_dir):
        from cron.scheduler import _run_job_script

        fake = _mock_config_with_trusted_dirs([str(trusted_dir)])
        with patch("cron.scheduler.load_config", fake):
            success, output = _run_job_script("/tmp/evil.py")
            assert success is False
            assert "blocked" in output.lower()

    def test_tilde_outside_all_dirs_still_blocked(self, cron_env, trusted_dir):
        from cron.scheduler import _run_job_script

        fake = _mock_config_with_trusted_dirs([str(trusted_dir)])
        with patch("cron.scheduler.load_config", fake):
            success, output = _run_job_script("~/evil.py")
            assert success is False
            assert "blocked" in output.lower()

    def test_real_file_in_scripts_dir_still_works(self, cron_env, trusted_dir):
        """Real files in scripts/ must still work regardless of trusted_dirs."""
        from cron.scheduler import _run_job_script

        script = cron_env / "scripts" / "local.py"
        script.write_text('print("local")\n')

        fake = _mock_config_with_trusted_dirs([str(trusted_dir)])
        with patch("cron.scheduler.load_config", fake):
            success, output = _run_job_script("local.py")
            assert success is True
            assert output == "local"
