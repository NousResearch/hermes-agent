"""Tests for file permissions hardening on sensitive files."""

import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


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
