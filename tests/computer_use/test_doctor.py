"""Tests for ``tools.computer_use.doctor``.

The doctor module drives cua-driver's stable ``health_report`` MCP tool over
stdio JSON-RPC and renders the structured response. Most of the surface is
about parsing what cua-driver hands back, plus the exit-code contract
downstream consumers (CI / `hermes update`) rely on:

* Exit 0 when overall == "ok"
* Exit 1 when overall in ("degraded", "failed") — at least one check
  failed but the tool itself ran successfully
* Exit 2 when the cua-driver binary is missing or the protocol breaks

We do NOT spin up a real cua-driver — that lives in the cua-driver
integration test suite (libs/cua-driver/rust/tests/integration/
test_health_report_mcp.py). Here we mock the subprocess and assert the
Hermes-side adapter behaves correctly against the documented response
shape.
"""

from __future__ import annotations

import json
from io import StringIO
from unittest.mock import MagicMock, patch


# ── helpers ────────────────────────────────────────────────────────────────


def _fake_proc_with_responses(*responses: dict) -> MagicMock:
    """Build a MagicMock subprocess.Popen handle that yields one JSON-RPC
    response per `readline()` call, then returns "" (EOF)."""
    lines = [json.dumps(r) + "\n" for r in responses] + [""]
    proc = MagicMock()
    proc.stdin = MagicMock()
    proc.stdout = MagicMock()
    proc.stdout.readline = MagicMock(side_effect=lines)
    proc.stderr = MagicMock()
    proc.stderr.read = MagicMock(return_value="")
    proc.wait = MagicMock(return_value=0)
    proc.kill = MagicMock()
    return proc


def _ok_report() -> dict:
    """Minimal well-formed health_report response."""
    return {
        "schema_version": "1",
        "platform": "darwin",
        "driver_version": "0.5.8",
        "overall": "ok",
        "checks": [
            {"name": "binary_version", "status": "pass", "message": "cua-driver 0.5.8"},
            {"name": "tcc_accessibility", "status": "pass", "message": "Accessibility is granted."},
        ],
    }


def _degraded_report() -> dict:
    """Report with one failing check — overall=degraded."""
    return {
        "schema_version": "1",
        "platform": "darwin",
        "driver_version": "0.5.8",
        "overall": "degraded",
        "checks": [
            {"name": "binary_version", "status": "pass", "message": "cua-driver 0.5.8"},
            {
                "name": "bundle_identity",
                "status": "fail",
                "message": "Process has no CFBundleIdentifier.",
                "hint": "Run inside CuaDriver.app",
                "data": {"executable_path": "/tmp/cua-driver"},
            },
        ],
    }


# ── exit codes ─────────────────────────────────────────────────────────────


class TestDoctorExitCodes:
    def test_ok_exits_0(self):
        from tools.computer_use import doctor

        proc = _fake_proc_with_responses(
            {"jsonrpc": "2.0", "id": 1, "result": {}},
            {"jsonrpc": "2.0", "id": 2, "result": {"structuredContent": _ok_report()}},
        )
        with patch("shutil.which", return_value="/fake/cua-driver"), \
             patch("subprocess.Popen", return_value=proc), \
             patch("sys.stdout", new_callable=StringIO):
            code = doctor.run_doctor()
        assert code == 0

    def test_degraded_exits_1(self):
        from tools.computer_use import doctor

        proc = _fake_proc_with_responses(
            {"jsonrpc": "2.0", "id": 1, "result": {}},
            {"jsonrpc": "2.0", "id": 2, "result": {"structuredContent": _degraded_report()}},
        )
        with patch("shutil.which", return_value="/fake/cua-driver"), \
             patch("subprocess.Popen", return_value=proc), \
             patch("sys.stdout", new_callable=StringIO):
            code = doctor.run_doctor()
        assert code == 1

    def test_failed_overall_exits_1(self):
        """`failed` overall (every check failed) is also exit 1, not 2 —
        the tool ran successfully; the diagnosis was bad."""
        from tools.computer_use import doctor

        report = _degraded_report()
        report["overall"] = "failed"
        proc = _fake_proc_with_responses(
            {"jsonrpc": "2.0", "id": 1, "result": {}},
            {"jsonrpc": "2.0", "id": 2, "result": {"structuredContent": report}},
        )
        with patch("shutil.which", return_value="/fake/cua-driver"), \
             patch("subprocess.Popen", return_value=proc), \
             patch("sys.stdout", new_callable=StringIO):
            code = doctor.run_doctor()
        assert code == 1

    def test_missing_binary_exits_2(self):
        from tools.computer_use import doctor

        with patch("shutil.which", return_value=None), \
             patch("sys.stdout", new_callable=StringIO):
            code = doctor.run_doctor()
        assert code == 2

    def test_protocol_error_exits_2(self, capsys):
        """An empty stdout response (driver crashed during handshake) is a
        protocol failure → exit 2."""
        from tools.computer_use import doctor

        proc = MagicMock()
        proc.stdin = MagicMock()
        proc.stdout = MagicMock()
        proc.stdout.readline = MagicMock(return_value="")  # EOF on initialize
        proc.stderr = MagicMock()
        proc.stderr.read = MagicMock(return_value="boom\n")
        proc.wait = MagicMock(return_value=0)
        proc.kill = MagicMock()

        with patch("shutil.which", return_value="/fake/cua-driver"), \
             patch("subprocess.Popen", return_value=proc):
            code = doctor.run_doctor()
        assert code == 2
        # stderr should mention the failure
        captured = capsys.readouterr()
        assert "cua-driver" in captured.err.lower() or "health_report" in captured.err.lower()


# ── response-shape parsing ─────────────────────────────────────────────────


class TestResponseShapeParsing:
    def test_prefers_structuredContent(self):
        from tools.computer_use import doctor

        proc = _fake_proc_with_responses(
            {"jsonrpc": "2.0", "id": 1, "result": {}},
            {"jsonrpc": "2.0", "id": 2, "result": {"structuredContent": _ok_report()}},
        )
        with patch("shutil.which", return_value="/fake/cua-driver"), \
             patch("subprocess.Popen", return_value=proc), \
             patch("sys.stdout", new_callable=StringIO) as out:
            doctor.run_doctor()
        # Header line includes driver version + platform + overall.
        text = out.getvalue()
        assert "darwin" in text
        assert "ok" in text

    def test_falls_back_to_text_content_when_structuredContent_absent(self):
        """Older cua-driver builds may emit health_report as a text content
        item carrying the JSON — the doctor should still parse it."""
        from tools.computer_use import doctor

        proc = _fake_proc_with_responses(
            {"jsonrpc": "2.0", "id": 1, "result": {}},
            {
                "jsonrpc": "2.0", "id": 2,
                "result": {
                    "content": [
                        {"type": "text", "text": json.dumps(_ok_report())},
                    ],
                },
            },
        )
        with patch("shutil.which", return_value="/fake/cua-driver"), \
             patch("subprocess.Popen", return_value=proc), \
             patch("sys.stdout", new_callable=StringIO) as out:
            code = doctor.run_doctor()
        assert code == 0
        assert "ok" in out.getvalue()

    def test_jsonrpc_error_response_exits_2(self, capsys):
        from tools.computer_use import doctor

        proc = _fake_proc_with_responses(
            {"jsonrpc": "2.0", "id": 1, "result": {}},
            {"jsonrpc": "2.0", "id": 2, "error": {"code": -32601, "message": "method not found"}},
        )
        with patch("shutil.which", return_value="/fake/cua-driver"), \
             patch("subprocess.Popen", return_value=proc):
            code = doctor.run_doctor()
        assert code == 2
        assert "method not found" in capsys.readouterr().err


# ── args / arg passthrough ─────────────────────────────────────────────────


class TestArgPassthrough:
    def test_include_passed_through_to_tools_call(self):
        from tools.computer_use import doctor

        proc = _fake_proc_with_responses(
            {"jsonrpc": "2.0", "id": 1, "result": {}},
            {"jsonrpc": "2.0", "id": 2, "result": {"structuredContent": _ok_report()}},
        )
        with patch("shutil.which", return_value="/fake/cua-driver"), \
             patch("subprocess.Popen", return_value=proc), \
             patch("sys.stdout", new_callable=StringIO):
            doctor.run_doctor(include=["binary_version", "tcc_accessibility"])

        # Inspect the second write to stdin — the tools/call payload.
        writes = [call.args[0] for call in proc.stdin.write.call_args_list]
        call_payload = next(json.loads(w) for w in writes if "tools/call" in w)
        assert call_payload["params"]["arguments"]["include"] == [
            "binary_version", "tcc_accessibility",
        ]

    def test_skip_passed_through(self):
        from tools.computer_use import doctor

        proc = _fake_proc_with_responses(
            {"jsonrpc": "2.0", "id": 1, "result": {}},
            {"jsonrpc": "2.0", "id": 2, "result": {"structuredContent": _ok_report()}},
        )
        with patch("shutil.which", return_value="/fake/cua-driver"), \
             patch("subprocess.Popen", return_value=proc), \
             patch("sys.stdout", new_callable=StringIO):
            doctor.run_doctor(skip=["bundle_identity"])
        writes = [call.args[0] for call in proc.stdin.write.call_args_list]
        call_payload = next(json.loads(w) for w in writes if "tools/call" in w)
        assert call_payload["params"]["arguments"]["skip"] == ["bundle_identity"]

    def test_no_filters_sends_empty_arguments(self):
        """When neither include nor skip is given, the arguments object is
        empty — not present-but-null — so the driver's default 'run every
        check' branch fires."""
        from tools.computer_use import doctor

        proc = _fake_proc_with_responses(
            {"jsonrpc": "2.0", "id": 1, "result": {}},
            {"jsonrpc": "2.0", "id": 2, "result": {"structuredContent": _ok_report()}},
        )
        with patch("shutil.which", return_value="/fake/cua-driver"), \
             patch("subprocess.Popen", return_value=proc), \
             patch("sys.stdout", new_callable=StringIO):
            doctor.run_doctor()
        writes = [call.args[0] for call in proc.stdin.write.call_args_list]
        call_payload = next(json.loads(w) for w in writes if "tools/call" in w)
        assert call_payload["params"]["arguments"] == {}


# ── json output ────────────────────────────────────────────────────────────


class TestJsonOutput:
    def test_json_output_is_parseable_round_trip(self):
        from tools.computer_use import doctor

        proc = _fake_proc_with_responses(
            {"jsonrpc": "2.0", "id": 1, "result": {}},
            {"jsonrpc": "2.0", "id": 2, "result": {"structuredContent": _ok_report()}},
        )
        with patch("shutil.which", return_value="/fake/cua-driver"), \
             patch("subprocess.Popen", return_value=proc), \
             patch("sys.stdout", new_callable=StringIO) as out:
            doctor.run_doctor(json_output=True)
        # Verify the captured text round-trips through json.loads and matches
        # the input report (the contract: --json passes the structured payload
        # through unchanged so downstream tooling can consume it directly).
        parsed = json.loads(out.getvalue())
        assert parsed == _ok_report()


# ── HERMES_CUA_DRIVER_CMD resolution ───────────────────────────────────────


class TestDriverCmdResolution:
    def test_explicit_driver_cmd_arg_wins(self):
        from tools.computer_use import doctor

        proc = _fake_proc_with_responses(
            {"jsonrpc": "2.0", "id": 1, "result": {}},
            {"jsonrpc": "2.0", "id": 2, "result": {"structuredContent": _ok_report()}},
        )
        # Mock _resolve_mcp_invocation to pass through the binary unchanged
        # (the spawnability check in _drive_health_report calls it).
        with patch("shutil.which", return_value="/fake/explicit-binary") as which_mock, \
             patch("subprocess.Popen", return_value=proc), \
             patch("tools.computer_use.cua_backend._resolve_mcp_invocation",
                   return_value=("/fake/explicit-binary", ["mcp"])), \
             patch("sys.stdout", new_callable=StringIO):
            doctor.run_doctor(driver_cmd="/custom/path/cua-driver")
        # shutil.which should have been called with the explicit arg, not
        # the env-var / default resolver.
        which_mock.assert_any_call("/custom/path/cua-driver")

    def test_env_var_used_when_no_arg_given(self, monkeypatch):
        from tools.computer_use import doctor

        monkeypatch.setenv("HERMES_CUA_DRIVER_CMD", "/env/path/cua-driver")
        proc = _fake_proc_with_responses(
            {"jsonrpc": "2.0", "id": 1, "result": {}},
            {"jsonrpc": "2.0", "id": 2, "result": {"structuredContent": _ok_report()}},
        )
        with patch("shutil.which", return_value="/env/path/cua-driver") as which_mock, \
             patch("subprocess.Popen", return_value=proc), \
             patch("tools.computer_use.cua_backend._resolve_mcp_invocation",
                   return_value=("/env/path/cua-driver", ["mcp"])), \
             patch("sys.stdout", new_callable=StringIO):
            doctor.run_doctor()
        # shutil.which should have been called with the env var.
        which_mock.assert_any_call("/env/path/cua-driver")
