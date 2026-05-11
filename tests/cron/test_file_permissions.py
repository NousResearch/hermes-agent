"""Tests for file permissions hardening on sensitive files."""

import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cron.jobs import CronStore, ensure_dirs, save_jobs


class TestCronFilePermissions(unittest.TestCase):
    """Verify cron files get secure permissions."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.cron_dir = Path(self.tmpdir) / "cron"
        self.output_dir = self.cron_dir / "output"

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_ensure_dirs_sets_0700(self):
        """ensure_dirs() via CronStore creates dirs with 0700 permissions."""
        root = Path(self.tmpdir)
        store = CronStore(
            scope="profile",
            root=root,
            cron_dir=root / "cron",
            jobs_file=root / "cron" / "jobs.json",
            output_dir=root / "cron" / "output",
            lock_file=root / "cron" / ".tick.lock",
        )
        ensure_dirs(store=store)

        cron_dir = root / "cron"
        output_dir = cron_dir / "output"
        cron_mode = stat.S_IMODE(os.stat(cron_dir).st_mode)
        output_mode = stat.S_IMODE(os.stat(output_dir).st_mode)
        self.assertEqual(cron_mode, 0o700)
        self.assertEqual(output_mode, 0o700)

    def test_save_jobs_sets_0600(self):
        """save_jobs() via CronStore creates jobs.json with 0600 permissions."""
        root = Path(self.tmpdir)
        store = CronStore(
            scope="profile",
            root=root,
            cron_dir=root / "cron",
            jobs_file=root / "cron" / "jobs.json",
            output_dir=root / "cron" / "output",
            lock_file=root / "cron" / ".tick.lock",
        )
        save_jobs([{"id": "test", "prompt": "hello"}], store=store)

        jobs_file = root / "cron" / "jobs.json"
        file_mode = stat.S_IMODE(os.stat(jobs_file).st_mode)
        self.assertEqual(file_mode, 0o600)

    def test_save_job_output_sets_0600(self):
        """save_job_output() via CronStore creates output with 0600 permissions."""
        root = Path(self.tmpdir)
        store = CronStore(
            scope="profile",
            root=root,
            cron_dir=root / "cron",
            jobs_file=root / "cron" / "jobs.json",
            output_dir=root / "cron" / "output",
            lock_file=root / "cron" / ".tick.lock",
        )
        from cron.jobs import save_job_output
        output_file = save_job_output("test-job", "test output content", store=store)

        file_mode = stat.S_IMODE(os.stat(output_file).st_mode)
        self.assertEqual(file_mode, 0o600)

        # Job output dir should also be 0700
        job_dir = store.output_dir / "test-job"
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
