"""Coverage for OmpACPClient and the _ACPProcessClient base it shares with
CopilotACPClient (added alongside the omp-acp provider).

Focus: the subclass-attribute seams (_display_name, _default_api_key,
_default_base_url, _install_hint, _resolve_command/_resolve_args) that the
Copilot->generic-base refactor introduced, plus one shared-base regression
(fs write safe-root) run against OmpACPClient to confirm the extraction
didn't silently leave any Copilot-only behavior behind in the base class.
"""

from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent.copilot_acp_client import (
    OMP_ACP_MARKER_BASE_URL,
    CopilotACPClient,
    OmpACPClient,
)


class _FakeProcess:
    def __init__(self) -> None:
        self.stdin = io.StringIO()


class OmpACPClientDefaultsTests(unittest.TestCase):
    def test_default_identity(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            client = OmpACPClient(acp_cwd="/tmp")
        self.assertEqual(client.api_key, "omp-acp")
        self.assertEqual(client.base_url, OMP_ACP_MARKER_BASE_URL)
        self.assertEqual(client.base_url, "acp://omp")

    def test_default_command_and_args_when_env_unset(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            client = OmpACPClient(acp_cwd="/tmp")
        self.assertEqual(client._acp_command, "omp")
        self.assertEqual(client._acp_args, ["acp"])

    def test_hermes_omp_acp_command_wins_over_omp_cli_path(self) -> None:
        env = {
            "HERMES_OMP_ACP_COMMAND": "/opt/omp/bin/omp",
            "OMP_CLI_PATH": "/usr/local/bin/omp",
        }
        with patch.dict(os.environ, env, clear=True):
            client = OmpACPClient(acp_cwd="/tmp")
        self.assertEqual(client._acp_command, "/opt/omp/bin/omp")

    def test_omp_cli_path_used_when_hermes_var_unset(self) -> None:
        with patch.dict(os.environ, {"OMP_CLI_PATH": "/usr/local/bin/omp"}, clear=True):
            client = OmpACPClient(acp_cwd="/tmp")
        self.assertEqual(client._acp_command, "/usr/local/bin/omp")

    def test_hermes_omp_acp_args_overrides_default(self) -> None:
        with patch.dict(os.environ, {"HERMES_OMP_ACP_ARGS": "acp --verbose"}, clear=True):
            client = OmpACPClient(acp_cwd="/tmp")
        self.assertEqual(client._acp_args, ["acp", "--verbose"])

    def test_explicit_constructor_args_beat_env(self) -> None:
        with patch.dict(os.environ, {"HERMES_OMP_ACP_COMMAND": "should-not-win"}, clear=True):
            client = OmpACPClient(acp_cwd="/tmp", acp_command="/explicit/omp")
        self.assertEqual(client._acp_command, "/explicit/omp")

    def test_does_not_inherit_copilot_deprecation_detection(self) -> None:
        """OmpACPClient has no _detect_known_bad_binary override; the base
        class default (None) must apply rather than Copilot's gh-deprecation
        fingerprinting, even when stderr happens to match that fingerprint."""
        client = OmpACPClient(acp_cwd="/tmp")
        copilot_deprecation_stderr = (
            "gh: unknown command \"copilot\" for \"gh\"\n"
            "the `gh copilot` extension is deprecated"
        )
        self.assertIsNone(client._detect_known_bad_binary(copilot_deprecation_stderr))


class OmpACPClientErrorMessageTests(unittest.TestCase):
    def test_missing_binary_error_names_omp_not_copilot(self) -> None:
        with patch.dict(os.environ, {"HERMES_OMP_ACP_COMMAND": "/no/such/omp"}, clear=True):
            client = OmpACPClient(acp_cwd="/tmp")

        with patch("subprocess.Popen", side_effect=FileNotFoundError()):
            with self.assertRaises(RuntimeError) as ctx:
                client._run_prompt("hello", timeout_seconds=1.0)

        message = str(ctx.exception)
        self.assertIn("Oh My Pi ACP", message)
        self.assertIn("/no/such/omp", message)
        self.assertIn("HERMES_OMP_ACP_COMMAND", message)
        self.assertNotIn("Copilot", message)

    def test_copilot_and_omp_missing_binary_messages_differ(self) -> None:
        with patch("subprocess.Popen", side_effect=FileNotFoundError()):
            with self.assertRaises(RuntimeError) as omp_ctx:
                OmpACPClient(acp_cwd="/tmp")._run_prompt("hi", timeout_seconds=1.0)
            with self.assertRaises(RuntimeError) as copilot_ctx:
                CopilotACPClient(acp_cwd="/tmp")._run_prompt("hi", timeout_seconds=1.0)

        self.assertNotEqual(str(omp_ctx.exception), str(copilot_ctx.exception))


class OmpACPClientSharedBaseRegressionTests(unittest.TestCase):
    """_ACPProcessClient behavior verified generic, not Copilot-only, by
    re-running one of CopilotACPClient's existing safety regressions
    (test_copilot_acp_client.py::test_write_text_file_respects_safe_root)
    against the other concrete subclass."""

    def setUp(self) -> None:
        self.client = OmpACPClient(acp_cwd="/tmp")

    def test_write_text_file_respects_safe_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            safe_root = root / "workspace"
            safe_root.mkdir()
            outside = root / "outside.txt"

            with patch.dict(os.environ, {"HERMES_WRITE_SAFE_ROOT": str(safe_root)}, clear=False):
                process = _FakeProcess()
                handled = self.client._handle_server_message(
                    {
                        "jsonrpc": "2.0",
                        "id": 5,
                        "method": "fs/write_text_file",
                        "params": {
                            "path": str(outside),
                            "content": "should-not-write",
                        },
                    },
                    process=process,
                    cwd=str(root),
                    text_parts=[],
                    reasoning_parts=[],
                )

        self.assertTrue(handled)
        response = json.loads(process.stdin.getvalue().strip())
        self.assertIn("error", response)
        self.assertIn("HERMES_WRITE_SAFE_ROOT", str(response["error"]))
        self.assertFalse(outside.exists())


if __name__ == "__main__":
    unittest.main()
