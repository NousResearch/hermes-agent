"""Native subprocess regressions; also runnable with stdlib unittest (no pytest).

The synthetic git replacement really spawns a descendant holding stdout open.
No provider, worker, configuration, or live checkout is used.
"""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from hermes_cli import version_info


class GitWaitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="version-git-")
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name)

    def synthetic_git(self, code):
        real_popen = subprocess.Popen

        def spawn(command, *args, **kwargs):
            return real_popen([sys.executable, "-c", code], *args, **kwargs)

        return patch.object(version_info.subprocess, "Popen", side_effect=spawn)

    def test_inherited_stdout_does_not_wait_for_descendant(self):
        # A finite descendant makes RED bounded even when communicate ignores timeout.
        code = (
            "import os, subprocess, sys, tempfile; "
            "subprocess.Popen([sys.executable, '-c', "
            "'import threading; threading.Event().wait(7)'], "
            "stdout=sys.stdout, stderr=sys.stderr, cwd=tempfile.gettempdir()); "
            "print(' inherited-output ', flush=True)"
        )
        with self.synthetic_git(code):
            start = time.monotonic()
            value = version_info._run_git(self.repo, "rev-parse", "HEAD")
            elapsed = time.monotonic() - start
        print(f"inherited stdout: {elapsed:.3f}s, result={value!r}", flush=True)
        self.assertLess(elapsed, 4.5, "waited for descendant pipe EOF past 3s timeout")
        self.assertEqual(value, "inherited-output")

    def test_timeout_returns_none_boundedly(self):
        with self.synthetic_git("import threading; threading.Event().wait(30)"):
            start = time.monotonic()
            self.assertIsNone(version_info._run_git(self.repo, "status"))
            self.assertLess(time.monotonic() - start, 4.5)

    def test_output_decoding_empty_and_nonzero(self):
        for code, expected in [
            ("import os; os.write(1, b'  value\\xff\\r\\n')", "value\ufffd"),
            ("print('   ')", None),
            ("import sys; print('not-success'); sys.exit(2)", None),
        ]:
            with self.subTest(code=code), self.synthetic_git(code):
                self.assertEqual(version_info._run_git(self.repo, "status"), expected)

    def test_spawn_failure_returns_none(self):
        with patch.object(version_info.subprocess, "Popen", side_effect=OSError("missing")):
            self.assertIsNone(version_info._run_git(self.repo, "status"))

    def test_native_git_output_and_error(self):
        checkout = Path(version_info.__file__).resolve().parents[1]
        expected = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout).decode().strip()
        self.assertEqual(version_info._run_git(checkout, "rev-parse", "HEAD"), expected)
        self.assertIsNone(version_info._run_git(checkout, "rev-parse", "refs/heads/does-not-exist-version-test"))

    def test_native_dirty_status_untracked_policy(self):
        subprocess.run(["git", "init", "-q", str(self.repo)], check=True)
        (self.repo / "untracked.txt").write_text("synthetic fixture", encoding="utf-8")
        self.assertFalse(version_info._git_version_info(self.repo).dirty)
        self.assertTrue(version_info._git_version_info(self.repo, include_untracked=True).dirty)
        self.assertFalse((self.repo / ".git" / "index.lock").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
