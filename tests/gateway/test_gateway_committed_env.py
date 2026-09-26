"""The Windows gateway must load the dependency generation committed by PM."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from gateway import run as gateway_run
from pm import environments


class GatewayCommittedEnvironmentTests(unittest.TestCase):
    def test_committed_environment_precedes_legacy_venv(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "project"
            committed = root / "committed"
            legacy = project / "venv"
            for venv in (committed, legacy):
                (venv / "Lib" / "site-packages").mkdir(parents=True)

            with (
                patch.object(gateway_run, "__file__", str(project / "gateway" / "run.py")),
                patch.object(sys, "platform", "win32"),
                patch.object(sys, "path", list(sys.path)),
                patch.object(environments, "committed_venv", return_value=committed),
                patch.object(gateway_run.site, "addsitedir"),
                patch.dict(os.environ, {"VIRTUAL_ENV": str(legacy)}, clear=False),
            ):
                gateway_run._ensure_windows_gateway_venv_imports()
                self.assertEqual(os.environ["VIRTUAL_ENV"], str(committed.resolve()))
                self.assertIn(str(committed / "Lib" / "site-packages"), sys.path)
                self.assertNotIn(str(legacy / "Lib" / "site-packages"), sys.path)


if __name__ == "__main__":
    unittest.main()
