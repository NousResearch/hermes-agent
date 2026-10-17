"""Windows process discovery contracts; runnable without optional pytest dependencies."""

import base64
import json
import os
import subprocess
import sys
import unittest
from unittest.mock import patch

from hermes_cli import dashboard_procs

# Keep the canonical pytest OS lane while allowing dependency-free local TDD.
try:
    import pytest
except ModuleNotFoundError:
    pass
else:
    pytestmark = pytest.mark.platforms("windows")


@unittest.skipUnless(sys.platform == "win32", "native Windows process discovery")
class WindowsProcessTableTests(unittest.TestCase):
    def test_cim_multiple_records_preserve_full_commands(self):
        records = [
            {"ProcessId": 4242, "CommandLine": '"C:\\Ứng dụng\\python.exe" -m hermes_cli.main dashboard'},
            {"ProcessId": 4243, "CommandLine": "hermes.exe serve --port 9119"},
        ]
        with patch("hermes_cli._subprocess_compat.bounded_probe_run", side_effect=[
            None, subprocess.CompletedProcess([], 0, json.dumps(records), ""),
        ]):
            self.assertEqual(dashboard_procs._iter_process_table(), [
                (r["ProcessId"], r["CommandLine"]) for r in records
            ])

    def test_unusable_wmic_results_fall_back_to_cim(self):
        command = "hermes.exe dashboard"
        fallback = subprocess.CompletedProcess([], 0, json.dumps({
            "ProcessId": 4242, "CommandLine": command,
        }), "")
        failures = [
            FileNotFoundError("wmic"), OSError("wmic unavailable"),
            subprocess.TimeoutExpired("wmic", 10),
            None, subprocess.CompletedProcess([], 1, "error", ""),
            subprocess.CompletedProcess([], 0, None, ""),
            subprocess.CompletedProcess([], 0, "", ""),
            subprocess.CompletedProcess([], 0, "  \r\n", ""),
            subprocess.CompletedProcess([], 0, "ProcessId=invalid\n", ""),
        ]
        for failure in failures:
            with self.subTest(failure=failure):
                with patch("hermes_cli._subprocess_compat.bounded_probe_run",
                           side_effect=[failure, fallback]) as probe:
                    self.assertEqual(dashboard_procs._iter_process_table(), [(4242, command)])
                    self.assertEqual(probe.call_count, 2)

    def test_malformed_cim_payloads_return_no_rows(self):
        for payload in ["not-json", "{", "null", "42", '"text"', "[]", "{}"]:
            with self.subTest(payload=payload):
                with patch("hermes_cli._subprocess_compat.bounded_probe_run", side_effect=[
                    None, subprocess.CompletedProcess([], 0, payload, ""),
                ]):
                    self.assertEqual(dashboard_procs._iter_process_table(), [])

    def test_cim_skips_invalid_records_without_losing_valid_neighbors(self):
        command = "hermes.exe dashboard"
        records = [None, 42, "text", [], {},
                   {"ProcessId": None, "CommandLine": command},
                   {"ProcessId": "oops", "CommandLine": command},
                   {"ProcessId": True, "CommandLine": command},
                   {"ProcessId": 42.5, "CommandLine": command},
                   {"ProcessId": 0, "CommandLine": command},
                   {"ProcessId": -42, "CommandLine": command},
                   {"ProcessId": 43, "CommandLine": None},
                   {"ProcessId": 44, "CommandLine": 123},
                   {"ProcessId": 45, "CommandLine": []},
                   {"ProcessId": 46, "CommandLine": ""},
                   {"ProcessId": 47},
                   {"ProcessId": 4242, "CommandLine": command}]
        with patch("hermes_cli._subprocess_compat.bounded_probe_run", side_effect=[
            None, subprocess.CompletedProcess([], 0, json.dumps(records), ""),
        ]):
            self.assertEqual(dashboard_procs._iter_process_table(), [(4242, command)])

    def test_unavailable_or_failed_powershell_returns_no_rows(self):
        failures = [None, FileNotFoundError("powershell.exe"), OSError("denied"),
                    subprocess.TimeoutExpired("powershell.exe", 10),
                    subprocess.CompletedProcess([], 1, "CIM failed", ""),
                    subprocess.CompletedProcess([], 0, None, ""),
                    subprocess.CompletedProcess([], 0, "", "")]
        for failure in failures:
            with self.subTest(failure=failure):
                with patch("hermes_cli._subprocess_compat.bounded_probe_run",
                           side_effect=[None, failure]) as probe:
                    self.assertEqual(dashboard_procs._iter_process_table(), [])
                    self.assertEqual(probe.call_count, 2)

    def test_successful_wmic_does_not_launch_powershell(self):
        output = ("CommandLine=hermes.exe dashboard\r\nProcessId=4242\r\n\r\n"
                  "CommandLine=python observer.py\r\nProcessId=4243\r\n")
        with patch("hermes_cli._subprocess_compat.bounded_probe_run", return_value=
                   subprocess.CompletedProcess([], 0, output, "")) as probe:
            self.assertEqual(dashboard_procs._iter_process_table(), [
                (4242, "hermes.exe dashboard"), (4243, "python observer.py"),
            ])
            probe.assert_called_once_with(
                ["wmic", "process", "get", "ProcessId,CommandLine", "/FORMAT:LIST"],
                timeout=10, errors="ignore")

    def test_fallback_scan_preserves_canonical_matcher_and_pid_exclusions(self):
        source = ("import os, re, sys\nfrom hermes_cli.main import main\n"
                  "sys.exit(main())")
        encoded = base64.b64encode(source.encode()).decode()
        bootstrap = ('python -I -c "import base64; exec(base64.b64decode(\''
                     + encoded + "'))\" dashboard --skip-build --no-open")
        commands = [
            (4242, "hermes.exe dashboard --no-open"),
            (4243, "hermes.exe serve --port 9119"),
            (4244, bootstrap),
            (4245, "hermes.exe dashboard"),
            (os.getpid(), "hermes.exe dashboard"),
            (5000, "tool --name hermes server 30"),
            (5001, "python observer.py dashboard --serve"),
            (5002, 'python -c "print(1)" -m hermes_cli.main dashboard'),
            (5003, "hermes.exe chat --model dashboard"),
            (5004, "hermes.exe kanban --preserve-cache"),
            (5005, "hermes.exe server"),
            (5006, 'powershell.exe -Command "hermes dashboard"'),
        ]
        payload = json.dumps([{"ProcessId": pid, "CommandLine": command}
                              for pid, command in commands])
        with patch("hermes_cli._subprocess_compat.bounded_probe_run", side_effect=[
            None, subprocess.CompletedProcess([], 0, payload, ""),
        ]), patch("hermes_cli.process_identity.ledger_entries", return_value=[]):
            self.assertEqual(dashboard_procs._scan_dashboard_processes(exclude_pids={4245}),
                             commands[:3])

    def test_scan_does_not_report_its_own_shell_launcher_as_a_backend(self):
        commands = [(4242, "sh.exe /source/.hermes/bin/hermes dashboard --status"),
                    (4243, "python -m hermes_cli.main dashboard --no-open")]
        with patch.object(dashboard_procs, "_iter_process_table", return_value=commands), \
                patch("hermes_cli.update_cmd_windows._self_and_non_gateway_ancestor_pids",
                      return_value={os.getpid(), 4242}), \
                patch("hermes_cli.process_identity.ledger_entries", return_value=[]):
            self.assertEqual(dashboard_procs._scan_dashboard_processes(), commands[1:])

    def test_missing_wmic_falls_back_to_cim_single_record(self):
        command = "python -m hermes_cli.main dashboard --no-open"
        result = subprocess.CompletedProcess([], 0, json.dumps({
            "ProcessId": 4242, "CommandLine": command,
        }), "")
        with patch("hermes_cli._subprocess_compat.bounded_probe_run",
                   side_effect=[None, result]) as probe:
            self.assertEqual(dashboard_procs._iter_process_table(), [(4242, command)])
        self.assertEqual(probe.call_count, 2)
        argv, kwargs = probe.call_args
        self.assertEqual(argv[0][0], "powershell.exe")
        self.assertIn("-NoProfile", argv[0])
        self.assertIn("-NonInteractive", argv[0])
        self.assertIn("Get-CimInstance", argv[0][-1])
        self.assertIn("Win32_Process", argv[0][-1])
        self.assertIn("ConvertTo-Json", argv[0][-1])
        self.assertGreater(kwargs["timeout"], 0)
        self.assertLessEqual(kwargs["timeout"], 10)


if __name__ == "__main__":
    unittest.main()
