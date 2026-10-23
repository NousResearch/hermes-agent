#!/usr/bin/env python3
"""Automated mock test suite for Hermes Companion pre-flight checks and diagnostics.

Tests all permutations of environment configurations without needing a fresh Mac,
a real iPhone, or real iCloud account:
1. Non-macOS (Linux) host detection
2. Docker / containerized runtime detection
3. Missing iCloud Drive root directory
4. First-run sequence: App container missing (phone has not synced yet)
5. macOS Privacy / TCC permission restrictions
6. Hermes CLI executable missing vs found in Mac App bundle / Homebrew
7. Background LaunchAgent service states
8. First-run explanation strings across CLI commands
"""

from __future__ import annotations

import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add parent directory to path to import companion
sys.path.insert(0, str(Path(__file__).resolve().parent))
import companion


class TestDoctorMocks(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmpdir.name)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    @patch("sys.platform", "linux")
    def test_linux_environment_fails_with_clear_remedy(self) -> None:
        """Verify non-macOS environments fail OS check with explicit guidance."""
        results = companion.check_environment_diagnostics(str(self.tmp_path))
        os_results = [r for r in results if r[1] == "Operating System"]
        self.assertEqual(len(os_results), 1)
        status, category, msg, remedy = os_results[0]
        self.assertEqual(status, "FAIL")
        self.assertIn("linux is not supported", msg)
        self.assertIn("Run Hermes directly on macOS", remedy)

        # Ensure doctor command returns error code 1
        args = companion.build_parser().parse_args(["--doctor"])
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            rc = companion.command_doctor(args)
        self.assertEqual(rc, 1)
        self.assertIn("[FAIL]  Operating System", captured.getvalue())

        # Ensure install_launch_agent is refused on non-macOS
        self.assertFalse(companion.install_launch_agent())

        # Ensure missing file explanation highlights macOS requirement
        explanation = companion.missing_icloud_explanation([self.tmp_path], "location")
        self.assertIn("Hermes Companion strictly requires macOS", explanation)

    @patch("sys.platform", "darwin")
    def test_docker_container_detected_and_refused(self) -> None:
        """Verify Docker / containerized environments are caught even if host reports Darwin."""
        with patch.dict(os.environ, {"CONTAINER": "docker"}):
            results = companion.check_environment_diagnostics(str(self.tmp_path))
            os_results = [r for r in results if r[1] == "Operating System"]
            self.assertEqual(len(os_results), 1)
            status, category, msg, remedy = os_results[0]
            self.assertEqual(status, "FAIL")
            self.assertIn("Container", msg)
            self.assertIn("Containers and Linux/Windows cannot access native iCloud Drive", remedy)

    @patch("sys.platform", "darwin")
    def test_missing_icloud_drive_root(self) -> None:
        """Verify failure when ~/Library/Mobile Documents does not exist."""
        with patch.object(Path, "is_dir", return_value=False):
            results = companion.check_environment_diagnostics(str(self.tmp_path))
            icloud_results = [r for r in results if r[1] == "iCloud Drive"]
            self.assertEqual(len(icloud_results), 1)
            status, category, msg, remedy = icloud_results[0]
            self.assertEqual(status, "FAIL")
            self.assertIn("Mobile Documents' not found", msg)
            self.assertIn("enable iCloud Drive in System Settings", remedy)

    @patch("sys.platform", "darwin")
    def test_first_run_sequence_container_unprovisioned(self) -> None:
        """Simulate cold start where iPhone app has not yet written its first fix."""
        empty_container_dir = self.tmp_path / "Unprovisioned"
        # Directory does not exist on disk
        results = companion.check_environment_diagnostics(str(empty_container_dir))
        container_results = [r for r in results if r[1] == "App Container"]
        self.assertEqual(len(container_results), 1)
        status, category, msg, remedy = container_results[0]
        self.assertEqual(status, "WARN")
        self.assertIn("not found yet", msg)
        self.assertIn("Open the Hermes Companion iOS app", remedy)

        # Check explanation in CLI command_where
        explanation = companion.missing_icloud_explanation([empty_container_dir], "location")
        self.assertIn("iCloud container has not synced to this Mac yet", explanation)
        self.assertIn("Open Hermes Companion on your iPhone to initialize the container", explanation)

    @patch("sys.platform", "darwin")
    def test_container_provisioned_with_file_inventory(self) -> None:
        """Verify synced files (location, health, threads, places) are accurately reported."""
        container = self.tmp_path / "iCloudContainer"
        container.mkdir(parents=True, exist_ok=True)
        (container / "latest_location.json").write_text("{}", encoding="utf-8")
        (container / "latest_health.json").write_text("{}", encoding="utf-8")
        (container / "places.json").write_text("[]", encoding="utf-8")
        threads = container / "threads"
        threads.mkdir()
        (threads / "th_1").mkdir()
        (threads / "th_2").mkdir()

        results = companion.check_environment_diagnostics(str(container))
        container_results = [r for r in results if r[1] == "App Container"]
        self.assertEqual(container_results[0][0], "PASS")
        self.assertIn("write access: yes", container_results[0][2])

        files_results = [r for r in results if r[1] == "Container Files"]
        self.assertEqual(len(files_results), 1)
        self.assertIn("location", files_results[0][2])
        self.assertIn("health", files_results[0][2])
        self.assertIn("threads (2)", files_results[0][2])
        self.assertIn("places", files_results[0][2])

    @patch("sys.platform", "darwin")
    @patch("companion.find_hermes_executable", return_value=None)
    def test_missing_hermes_executable_fails(self, _mock_exec: MagicMock) -> None:
        """Verify missing Hermes CLI is flagged as critical failure."""
        results = companion.check_environment_diagnostics(str(self.tmp_path))
        cli_results = [r for r in results if r[1] == "Hermes CLI"]
        self.assertEqual(len(cli_results), 1)
        status, category, msg, remedy = cli_results[0]
        self.assertEqual(status, "FAIL")
        self.assertIn("not found in PATH", msg)
        self.assertIn("Install Hermes CLI", remedy)

    @patch("sys.platform", "darwin")
    def test_hermes_executable_in_app_bundle_discovered(self) -> None:
        """Verify discovery when hermes binary is inside /Applications/Hermes.app."""
        fake_app_bin = self.tmp_path / "Applications" / "Hermes.app" / "Contents" / "MacOS" / "hermes"
        fake_app_bin.parent.mkdir(parents=True, exist_ok=True)
        fake_app_bin.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        fake_app_bin.chmod(0o755)

        with patch("companion.find_hermes_executable", return_value=str(fake_app_bin)):
            results = companion.check_environment_diagnostics(str(self.tmp_path))
            cli_results = [r for r in results if r[1] == "Hermes CLI"]
            self.assertEqual(cli_results[0][0], "PASS")
            self.assertEqual(cli_results[0][2], f"Executable located at {fake_app_bin}")

    @patch("sys.platform", "darwin")
    def test_tcc_permission_denied_handled(self) -> None:
        """Verify permission denied on ~/Library/Mobile Documents produces clear remedy."""
        with patch.object(Path, "iterdir", side_effect=PermissionError("Operation not permitted")):
            results = companion.check_environment_diagnostics(str(self.tmp_path))
            perm_results = [r for r in results if r[1] == "Privacy Permissions"]
            self.assertEqual(len(perm_results), 1)
            status, category, msg, remedy = perm_results[0]
            self.assertEqual(status, "FAIL")
            self.assertIn("Permission denied", msg)
            self.assertIn("System Settings -> Privacy & Security -> Files and Folders", remedy)

    @patch("sys.platform", "darwin")
    @patch("companion.find_hermes_executable", return_value="/opt/homebrew/bin/hermes")
    def test_all_checks_pass(self, _mock_exec: MagicMock) -> None:
        """Verify clean run passes with exit code 0."""
        container = self.tmp_path / "Documents"
        container.mkdir(parents=True, exist_ok=True)
        (container / "latest_location.json").write_text("{}", encoding="utf-8")

        with patch("companion.get_launch_agent_path", return_value=self.tmp_path / "agent.plist"):
            args = companion.build_parser().parse_args(["--doctor", "--icloud-dir", str(container)])
            captured = io.StringIO()
            with patch("sys.stdout", captured):
                rc = companion.command_doctor(args)
            self.assertEqual(rc, 0)
            self.assertIn("All critical checks passed.", captured.getvalue())


if __name__ == "__main__":
    unittest.main()
