from pathlib import Path
from tempfile import TemporaryDirectory
import os
import unittest

from scripts.legacy_jobs import job_files


class JobFilesTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.repo = Path(self.tmp.name)
        (self.repo / "jobs" / "team" / "foo").mkdir(parents=True)

    def tearDown(self):
        self.tmp.cleanup()

    def test_write_read_append_delete(self):
        job_files.write_file("jobs/team/foo", "a.txt", b"one", repo_dir=self.repo)
        job_files.write_file("jobs/team/foo", "a.txt", b"two", append=True, repo_dir=self.repo)
        self.assertEqual(job_files.read_file("jobs/team/foo", "a.txt", self.repo), b"onetwo")
        self.assertTrue(job_files.delete_file("jobs/team/foo", "a.txt", self.repo))
        with self.assertRaises(job_files.JobFileError):
            job_files.read_file("jobs/team/foo", "a.txt", self.repo)

    def test_nested_relative_path_allowed(self):
        path = job_files.write_file("jobs/team/foo", "nested/a.txt", b"x", repo_dir=self.repo)
        self.assertTrue(path.exists())
        self.assertIn(("nested/a.txt", 1, job_files.list_files("jobs/team/foo", self.repo)[0][2]), job_files.list_files("jobs/team/foo", self.repo))

    def test_rejects_absolute_path(self):
        with self.assertRaises(job_files.JobFileError):
            job_files.resolve_data_path("jobs/team/foo", "/tmp/x", self.repo)

    def test_rejects_parent_escape(self):
        with self.assertRaises(job_files.JobFileError):
            job_files.resolve_data_path("jobs/team/foo", "../x", self.repo)

    def test_rejects_symlink_escape(self):
        base = self.repo / "jobs" / "team" / "foo" / "data"
        base.mkdir(parents=True)
        outside = self.repo / "outside"
        outside.mkdir()
        (base / "link").symlink_to(outside)
        with self.assertRaises(job_files.JobFileError):
            job_files.resolve_data_path("jobs/team/foo", "link/x", self.repo)

    def test_enforces_file_cap(self):
        old = job_files.MAX_FILE_BYTES
        try:
            job_files.MAX_FILE_BYTES = 3
            with self.assertRaises(job_files.JobFileError):
                job_files.write_file("jobs/team/foo", "big.bin", b"1234", repo_dir=self.repo)
        finally:
            job_files.MAX_FILE_BYTES = old


if __name__ == "__main__":
    unittest.main()
