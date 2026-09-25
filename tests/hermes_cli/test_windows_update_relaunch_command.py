"""Windows update relaunch commands must be visible to gateway discovery."""

import subprocess
import unittest
from unittest.mock import patch

from gateway.status import looks_like_gateway_runtime_command_line
from hermes_cli import gateway


class WindowsUpdateRelaunchCommandTests(unittest.TestCase):
    @patch.object(gateway, "is_windows", return_value=True)
    def test_default_profile_uses_discoverable_gateway_command(self, _windows):
        argv = gateway._gateway_run_args_for_profile("default")
        self.assertEqual(argv[1:5], ["-m", "hermes_cli.main", "gateway", "run"])
        self.assertEqual(argv[-1], "--replace")
        self.assertTrue(looks_like_gateway_runtime_command_line(subprocess.list2cmdline(argv)))

    @patch.object(gateway, "is_windows", return_value=True)
    def test_named_profile_preserves_selector(self, _windows):
        argv = gateway._gateway_run_args_for_profile("research")
        self.assertEqual(
            argv[1:], ["-m", "hermes_cli.main", "--profile", "research", "gateway", "run", "--replace"]
        )
        self.assertTrue(looks_like_gateway_runtime_command_line(subprocess.list2cmdline(argv)))


if __name__ == "__main__":
    unittest.main()
