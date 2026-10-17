"""Canonical Windows launchers must also resolve correctly from Git Bash."""

import json
import os
import shutil
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hermes_cli import _launchers

try:
    import pytest
except ModuleNotFoundError:
    pass
else:
    pytestmark = pytest.mark.platforms("windows")


@unittest.skipUnless(sys.platform == "win32", "native Windows launcher publication")
class WindowsShellLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR"))
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "source with spaces"
        self.root.mkdir()
        self.out = self.root / ".hermes" / "bin"
        self.out.mkdir(parents=True)
        self.python = Path(getattr(sys, "_base_executable", sys.executable))

    def mint(self):
        with patch.object(_launchers, "_load_script_maker", return_value=None):
            return _launchers.mint_launcher("hermes", self.root, self.out, self.python, None)

    def test_cmd_fallback_also_publishes_shell_companion(self):
        native = self.mint()
        self.assertEqual(native, self.out / "hermes.cmd")
        shell = self.out / "hermes"
        self.assertTrue(shell.is_file(), "Bash otherwise skips the .cmd and runs a stale venv script")
        self.assertTrue(os.access(shell, os.X_OK))
        self.assertIn(self.python.as_posix(), shell.read_text(encoding="utf-8"))
        before = shell.stat().st_mtime_ns
        self.mint()
        self.assertEqual(before, shell.stat().st_mtime_ns)

    def test_shell_executes_shared_bootstrap_and_preserves_arguments(self):
        bash = shutil.which("bash")
        if not bash:
            self.skipTest("Git Bash unavailable")
        (self.root / "hermes_bootstrap.py").write_text("", encoding="utf-8")
        package = self.root / "hermes_cli"
        package.mkdir()
        (package / "__init__.py").write_text("", encoding="utf-8")
        (package / "main.py").write_text(
            "import json, sys\ndef main():\n print(json.dumps(sys.argv[1:], ensure_ascii=True))\n return 0\n",
            encoding="utf-8",
        )
        self.mint()
        shell = self.out / "hermes"
        self.assertTrue(shell.exists(), "shell entry point must be published before execution")
        args = ["chat", "space value", 'quote"value', "single'quote", "$HOME;echo no", "đơn hàng"]
        env = dict(os.environ, HERMES_HOME=str(self.root / "isolated home"),
                   PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1")
        # Enter through real shell syntax. Windows list2cmdline does not model
        # MSYS's native argv parser for quoted positional arguments to bash.exe.
        invocation = self.root / "invoke.sh"
        invocation.write_text("#!/bin/sh\nexec " + shlex.join([shell.as_posix(), *args]) + "\n",
                              encoding="utf-8", newline="\n")
        result = subprocess.run([bash, invocation.as_posix()], env=env,
                                capture_output=True, text=True, encoding="utf-8", timeout=20, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), args)

    def test_foreign_shell_command_is_preserved_while_native_launcher_is_published(self):
        shell = self.out / "hermes"
        content = "#!/bin/sh\nexec foreign-hermes \"$@\"\n"
        shell.write_text(content, encoding="utf-8")
        self.assertEqual(self.mint(), self.out / "hermes.cmd")
        self.assertEqual(shell.read_text(encoding="utf-8"), content)

    def test_shell_publication_failure_is_not_reported_as_success(self):
        with patch.object(_launchers, "_write_shell", return_value=None):
            self.assertIsNone(self.mint())


if __name__ == "__main__":
    unittest.main()
