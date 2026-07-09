"""Tests for file permissions hardening on sensitive files."""

import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


def _stat_with_fake_owner(fake_target_path, fake_uid, fake_gid, real_stat):
    """Build a side_effect for os.stat that returns a stat-like object with
    the given uid/gid when the path matches ``fake_target_path``, and
    delegates to the real os.stat for everything else.

    Used to simulate "HERMES_DIR is owned by a different user" in tests
    of _coerce_owner without needing CAP_CHOWN. The code under test only
    reads ``.st_uid`` and ``.st_gid`` off the result, so a SimpleNamespace
    is enough.
    """
    from types import SimpleNamespace
    target = os.fspath(fake_target_path)

    def _wrapped(path, *args, **kwargs):
        if os.fspath(path) == target:
            return SimpleNamespace(st_uid=fake_uid, st_gid=fake_gid)
        return real_stat(path, *args, **kwargs)

    return _wrapped


class TestCronFilePermissions(unittest.TestCase):
    """Verify cron files get secure permissions."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.cron_dir = Path(self.tmpdir) / "cron"
        self.output_dir = self.cron_dir / "output"

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    @patch("cron.jobs.CRON_DIR")
    @patch("cron.jobs.OUTPUT_DIR")
    @patch("cron.jobs.JOBS_FILE")
    def test_ensure_dirs_sets_0700(self, mock_jobs_file, mock_output, mock_cron):
        mock_cron.__class__ = Path
        # Use real paths
        cron_dir = Path(self.tmpdir) / "cron"
        output_dir = cron_dir / "output"

        with patch("cron.jobs.CRON_DIR", cron_dir), \
             patch("cron.jobs.OUTPUT_DIR", output_dir):
            from cron.jobs import ensure_dirs
            ensure_dirs()

            cron_mode = stat.S_IMODE(os.stat(cron_dir).st_mode)
            output_mode = stat.S_IMODE(os.stat(output_dir).st_mode)
            self.assertEqual(cron_mode, 0o700)
            self.assertEqual(output_mode, 0o700)

    @patch("cron.jobs.CRON_DIR")
    @patch("cron.jobs.OUTPUT_DIR")
    @patch("cron.jobs.JOBS_FILE")
    def test_save_jobs_sets_0600(self, mock_jobs_file, mock_output, mock_cron):
        cron_dir = Path(self.tmpdir) / "cron"
        output_dir = cron_dir / "output"
        jobs_file = cron_dir / "jobs.json"

        with patch("cron.jobs.CRON_DIR", cron_dir), \
             patch("cron.jobs.OUTPUT_DIR", output_dir), \
             patch("cron.jobs.JOBS_FILE", jobs_file):
            from cron.jobs import save_jobs
            save_jobs([{"id": "test", "prompt": "hello"}])

            file_mode = stat.S_IMODE(os.stat(jobs_file).st_mode)
            self.assertEqual(file_mode, 0o600)

    @unittest.skipUnless(hasattr(os, "getuid"), "POSIX only")
    def test_coerce_owner_called_with_hermes_dir_uid_gid(self):
        """Verify _coerce_owner passes the HERMES_DIR owner's uid/gid to
        os.chown on POSIX systems. Uses a recorder so we don't require
        CAP_CHOWN in the test process."""
        jobs_file = Path(self.tmpdir) / "sentinel.json"
        jobs_file.touch()
        fake_uid = 12345
        fake_gid = 23456
        recorded = []

        with patch("cron.jobs.HERMES_DIR", self.tmpdir), \
             patch("os.stat", side_effect=_stat_with_fake_owner(self.tmpdir, fake_uid, fake_gid, os.stat)), \
             patch("os.chown", side_effect=lambda p, u, g: recorded.append((os.fspath(p), u, g))):
            from cron.jobs import _coerce_owner
            _coerce_owner(jobs_file)

        self.assertEqual(len(recorded), 1, "os.chown should be called exactly once")
        self.assertEqual(recorded[0][1], fake_uid)
        self.assertEqual(recorded[0][2], fake_gid)

    @unittest.skipUnless(hasattr(os, "getuid"), "POSIX only")
    def test_coerce_owner_no_op_when_hermes_dir_missing(self):
        """If HERMES_DIR does not exist (extremely unusual), _coerce_owner
        must not raise. Best-effort semantics, never breaks the write."""
        missing = Path(self.tmpdir) / "does_not_exist"
        target = Path(self.tmpdir) / "sentinel.json"
        target.touch()
        chown_called = []

        with patch("cron.jobs.HERMES_DIR", missing), \
             patch("os.chown", side_effect=lambda *a, **k: chown_called.append(a)):
            from cron.jobs import _coerce_owner
            _coerce_owner(target)  # should not raise

        self.assertEqual(chown_called, [], "os.chown must not be called when HERMES_DIR is missing")

    @unittest.skipUnless(hasattr(os, "getuid"), "POSIX only")
    def test_coerce_owner_swallows_permission_error(self):
        """If os.chown raises PermissionError (we lack CAP_CHOWN), _coerce_owner
        must not propagate the exception. The write still succeeded; we just
        couldn't fix the ownership. Better to log later than to break the save."""
        jobs_file = Path(self.tmpdir) / "sentinel.json"
        jobs_file.touch()
        fake_uid = 12345
        fake_gid = 23456

        def deny_chown(*args, **kwargs):
            raise PermissionError("test: missing CAP_CHOWN")

        with patch("cron.jobs.HERMES_DIR", self.tmpdir), \
             patch("os.stat", side_effect=_stat_with_fake_owner(self.tmpdir, fake_uid, fake_gid, os.stat)), \
             patch("os.chown", side_effect=deny_chown):
            from cron.jobs import _coerce_owner
            _coerce_owner(jobs_file)  # must not raise

    def test_save_job_output_sets_0600(self):
        output_dir = Path(self.tmpdir) / "output"
        with patch("cron.jobs.OUTPUT_DIR", output_dir), \
             patch("cron.jobs.CRON_DIR", Path(self.tmpdir)), \
             patch("cron.jobs.ensure_dirs"):
            output_dir.mkdir(parents=True, exist_ok=True)
            from cron.jobs import save_job_output
            output_file = save_job_output("test-job", "test output content")

            file_mode = stat.S_IMODE(os.stat(output_file).st_mode)
            self.assertEqual(file_mode, 0o600)

            # Job output dir should also be 0700
            job_dir = output_dir / "test-job"
            dir_mode = stat.S_IMODE(os.stat(job_dir).st_mode)
            self.assertEqual(dir_mode, 0o700)


class TestConfigFilePermissions(unittest.TestCase):
    """Verify config files get secure permissions."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_save_config_sets_0600(self):
        config_path = Path(self.tmpdir) / "config.yaml"
        with patch("hermes_cli.config.get_config_path", return_value=config_path), \
             patch("hermes_cli.config.ensure_hermes_home"):
            from hermes_cli.config import save_config
            save_config({"model": "test/model"})

            file_mode = stat.S_IMODE(os.stat(config_path).st_mode)
            self.assertEqual(file_mode, 0o600)

    def test_save_env_value_sets_0600(self):
        env_path = Path(self.tmpdir) / ".env"
        with patch("hermes_cli.config.get_env_path", return_value=env_path), \
             patch("hermes_cli.config.ensure_hermes_home"):
            from hermes_cli.config import save_env_value
            save_env_value("TEST_KEY", "test_value")

            file_mode = stat.S_IMODE(os.stat(env_path).st_mode)
            self.assertEqual(file_mode, 0o600)

    def test_ensure_hermes_home_sets_0700(self):
        home = Path(self.tmpdir) / ".hermes"
        with patch("hermes_cli.config.get_hermes_home", return_value=home):
            from hermes_cli.config import ensure_hermes_home
            ensure_hermes_home()

            home_mode = stat.S_IMODE(os.stat(home).st_mode)
            self.assertEqual(home_mode, 0o700)

            for subdir in ("cron", "sessions", "logs", "memories"):
                subdir_mode = stat.S_IMODE(os.stat(home / subdir).st_mode)
                self.assertEqual(subdir_mode, 0o700, f"{subdir} should be 0700")


class TestSecureHelpers(unittest.TestCase):
    """Test the _secure_file and _secure_dir helpers."""

    def test_secure_file_nonexistent_no_error(self):
        from cron.jobs import _secure_file
        _secure_file(Path("/nonexistent/path/file.json"))  # Should not raise

    def test_secure_dir_nonexistent_no_error(self):
        from cron.jobs import _secure_dir
        _secure_dir(Path("/nonexistent/path"))  # Should not raise


if __name__ == "__main__":
    unittest.main()
