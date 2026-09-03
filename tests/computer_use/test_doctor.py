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
import sys
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest


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




@pytest.fixture(autouse=True)
def _default_cli_version_matches_report(monkeypatch):
    """Existing tests mock only the MCP Popen handshake. ``subprocess.run``
    (used for ``--version``) goes through Popen too, so without this the
    mock breaks version probing. Default to a CLI version that matches
    ``_ok_report`` / ``_degraded_report`` (0.5.8); identity tests override.
    """
    from tools.computer_use import doctor

    monkeypatch.setattr(
        doctor,
        "_read_cli_version",
        lambda binary, timeout=5.0: "cua-driver 0.5.8",
    )


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
        # Verify the captured text round-trips through json.loads. Upstream
        # health_report keys are preserved; Hermes adds hermes_identity.
        parsed = json.loads(out.getvalue())
        report = _ok_report()
        for key, value in report.items():
            assert parsed[key] == value
        assert "hermes_identity" in parsed
        assert parsed["hermes_identity"]["resolved_binary"]

    def test_linux_wayland_output_identifies_cli_environment_scope(self, monkeypatch):
        from tools.computer_use import doctor

        monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-1")
        monkeypatch.setenv("DISPLAY", ":0")
        report = _ok_report()
        report["platform"] = "linux"
        proc = _fake_proc_with_responses(
            {"jsonrpc": "2.0", "id": 1, "result": {}},
            {"jsonrpc": "2.0", "id": 2, "result": {"structuredContent": report}},
        )

        with patch("shutil.which", return_value="/fake/cua-driver"), \
             patch("subprocess.Popen", return_value=proc), \
             patch("sys.stdout", new_callable=StringIO) as out:
            doctor.run_doctor(json_output=True)

        context = json.loads(out.getvalue())["hermes_environment"]
        assert context == {
            "gateway_environment_checked": False,
            "scope": "cli_process",
        }

    def test_linux_wayland_text_warns_gateway_environment_may_differ(self, monkeypatch):
        from tools.computer_use import doctor

        monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-1")
        report = _ok_report()
        report["platform"] = "linux"
        proc = _fake_proc_with_responses(
            {"jsonrpc": "2.0", "id": 1, "result": {}},
            {"jsonrpc": "2.0", "id": 2, "result": {"structuredContent": report}},
        )

        with patch("shutil.which", return_value="/fake/cua-driver"), \
             patch("subprocess.Popen", return_value=proc), \
             patch("sys.stdout", new_callable=StringIO) as out:
            doctor.run_doctor()

        text = out.getvalue().lower()
        assert "current cli process" in text
        assert "gateway environment was not checked" in text


# ── HERMES_CUA_DRIVER_CMD resolution ───────────────────────────────────────


class TestDriverCmdResolution:
    def test_explicit_driver_cmd_arg_wins(self):
        from tools.computer_use import doctor

        proc = _fake_proc_with_responses(
            {"jsonrpc": "2.0", "id": 1, "result": {}},
            {"jsonrpc": "2.0", "id": 2, "result": {"structuredContent": _ok_report()}},
        )
        with patch("shutil.which", return_value="/fake/explicit-binary") as which_mock, \
             patch("subprocess.Popen", return_value=proc), \
             patch("sys.stdout", new_callable=StringIO):
            doctor.run_doctor(driver_cmd="/custom/path/cua-driver")
        # shutil.which should have been called with the explicit arg, not
        # the env-var / default resolver.
        which_mock.assert_called_with("/custom/path/cua-driver")

    def test_env_var_used_when_no_arg_given(self, monkeypatch):
        from tools.computer_use import doctor

        monkeypatch.setenv("HERMES_CUA_DRIVER_CMD", "/env/path/cua-driver")
        proc = _fake_proc_with_responses(
            {"jsonrpc": "2.0", "id": 1, "result": {}},
            {"jsonrpc": "2.0", "id": 2, "result": {"structuredContent": _ok_report()}},
        )
        with patch("shutil.which", return_value="/env/path/cua-driver") as which_mock, \
             patch("subprocess.Popen", return_value=proc), \
             patch("sys.stdout", new_callable=StringIO), \
             patch("hermes_cli.tools_config._cua_driver_cmd", side_effect=Exception("force env")):
            # Force env-var resolution path inside run_doctor.
            doctor.run_doctor()
        which_mock.assert_called_with("/env/path/cua-driver")

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX user-local path regression")
    def test_user_local_driver_is_found_when_path_omits_it(self, tmp_path, monkeypatch):
        """Doctor must inspect the same user-local driver as the runtime."""
        from tools.computer_use import doctor

        driver = tmp_path / ".local" / "bin" / "cua-driver"
        driver.parent.mkdir(parents=True)
        driver.write_text("#!/bin/sh\nexit 0\n")
        driver.chmod(0o755)

        monkeypatch.delenv("HERMES_CUA_DRIVER_CMD", raising=False)
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("PATH", "/usr/bin:/bin:/usr/sbin:/sbin")

        with patch("tools.computer_use.doctor._drive_health_report", return_value=_ok_report()) as health, \
             patch("sys.stdout", new_callable=StringIO):
            assert doctor.run_doctor() == 0

        health.assert_called_once_with(str(driver), include=(), skip=(), timeout=12.0)


# ── cua-driver 0.10 unclassified health_report fallback ────────────────────


def _unclassified_health_result() -> dict:
    """MCP tools/call result shape from cua-driver 0.10.x denial."""
    return {
        "isError": True,
        "content": [
            {
                "type": "text",
                "text": (
                    "Permission denied: tool 'health_report' has no "
                    "reviewed risk classification"
                ),
            }
        ],
        "structuredContent": {"exit_code": 1},
    }


def _perms_ok_result() -> dict:
    return {
        "isError": False,
        "content": [{"type": "text", "text": "ok"}],
        "structuredContent": {
            "accessibility": True,
            "screen_recording": True,
            "screen_recording_capturable": True,
        },
    }


def _list_apps_ok_result() -> dict:
    return {
        "isError": False,
        "content": [{"type": "text", "text": "Found 1 app"}],
        "structuredContent": {
            "apps": [{"name": "Finder", "pid": 1, "running": True}],
        },
    }


class TestHealthReportFallback:
    """cua-driver 0.10 marks health_report risk-unclassified → isError.

    Doctor must NOT treat structuredContent={exit_code:1} as a real report
    (that produced '• cua-driver ? on ? — ?'). It synthesizes schema_version=1
    via check_permissions / list_apps / CLI --version instead.
    """





    def test_extract_raises_health_report_unavailable_on_isError(self):
        from tools.computer_use import doctor

        with __import__("pytest").raises(doctor.HealthReportUnavailable) as ei:
            doctor._extract_health_report_from_result(_unclassified_health_result())
        assert "Permission denied" in str(ei.value) or "unclassified" in str(ei.value).lower() or "risk" in str(ei.value).lower()



# ── binary identity (CLI --version vs health_report) ───────────────────────


class TestDoctorVersionIdentity:
    def test_header_prefers_cli_version_on_mismatch(self):
        """Windows has been observed reporting 0.8.3 via health_report while
        the resolved binary is 0.12.6 — doctor must surface the real version."""
        from tools.computer_use import doctor

        proc = _fake_proc_with_responses(
            {"jsonrpc": "2.0", "id": 1, "result": {}},
            {"jsonrpc": "2.0", "id": 2, "result": {"structuredContent": _ok_report()}},
        )
        # _ok_report claims 0.5.8; CLI says 0.12.6
        with patch("shutil.which", return_value="/fake/cua-driver"), \
             patch("subprocess.Popen", return_value=proc), \
             patch.object(doctor, "_read_cli_version", return_value="cua-driver 0.12.6"), \
             patch("sys.stdout", new_callable=StringIO) as out:
            code = doctor.run_doctor()
        assert code == 0
        text = out.getvalue()
        assert "0.12.6" in text
        assert "version mismatch" in text.lower()
        assert "0.5.8" in text  # health_report value still shown


    def test_matching_versions_no_mismatch_flag(self):
        from tools.computer_use import doctor

        proc = _fake_proc_with_responses(
            {"jsonrpc": "2.0", "id": 1, "result": {}},
            {"jsonrpc": "2.0", "id": 2, "result": {"structuredContent": _ok_report()}},
        )
        with patch("shutil.which", return_value="/fake/cua-driver"), \
             patch("subprocess.Popen", return_value=proc), \
             patch.object(doctor, "_read_cli_version", return_value="cua-driver 0.5.8"), \
             patch("sys.stdout", new_callable=StringIO) as out:
            code = doctor.run_doctor(json_output=True)
        assert code == 0
        payload = json.loads(out.getvalue())
        assert payload["hermes_identity"]["version_mismatch"] is False

