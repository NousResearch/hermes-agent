import subprocess
import unittest
from unittest import mock

from hermes_cli import gateway_windows


class GatewayWindowsEncodingTests(unittest.TestCase):
    def test_exec_schtasks_uses_utf8_replace_for_captured_output(self):
        run = mock.Mock(
            return_value=subprocess.CompletedProcess(
                args=[r"C:\Windows\System32\schtasks.exe", "/Query"],
                returncode=0,
                stdout="ok",
                stderr="",
            )
        )
        with (
            mock.patch.object(gateway_windows, "_assert_windows", lambda: None),
            mock.patch.object(
                gateway_windows.shutil,
                "which",
                lambda name: r"C:\Windows\System32\schtasks.exe",
            ),
            mock.patch.object(gateway_windows.subprocess, "run", run),
        ):
            code, out, err = gateway_windows._exec_schtasks(["/Query"])

        run.assert_called_once()
        argv = run.call_args.args[0]
        kwargs = run.call_args.kwargs

        self.assertEqual(code, 0)
        self.assertEqual(out, "ok")
        self.assertEqual(err, "")
        self.assertEqual(
            argv,
            [
                r"C:\Windows\System32\schtasks.exe",
                "/Query",
            ],
        )
        self.assertIs(kwargs["text"], True)
        self.assertEqual(kwargs["encoding"], "utf-8")
        self.assertEqual(kwargs["errors"], "replace")
