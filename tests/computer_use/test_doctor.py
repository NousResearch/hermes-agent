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
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


# ── helpers ────────────────────────────────────────────────────────────────


def _pm_driver(binary: str):
    """PM reports *binary* as the installed cua-driver (the runtime's selection)."""
    return patch("pm.installed_package", return_value=SimpleNamespace(binary=Path(binary)))


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
        with _pm_driver("/fake/cua-driver"), \
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
        with _pm_driver("/fake/cua-driver"), \
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
        with _pm_driver("/fake/cua-driver"), \
             patch("subprocess.Popen", return_value=proc), \
             patch("sys.stdout", new_callable=StringIO):
            code = doctor.run_doctor()
        assert code == 1

    def test_spawn_denied_exits_2_with_diagnosis(self, capsys):
        """The runtime interpreter cannot execute the resolved binary (Windows WinError 5 on a
        `WindowsApps` install): a diagnosis + exit 2, never a raw traceback."""
        from tools.computer_use import doctor

        with _pm_driver("/protected/cua-driver"), \
             patch("subprocess.Popen", side_effect=PermissionError(13, "Access is denied")):
            code = doctor.run_doctor()
        assert code == 2
        err = capsys.readouterr().err
        assert "Access is denied" in err and "HERMES_CUA_DRIVER_CMD" in err

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

        with _pm_driver("/fake/cua-driver"), \
             patch("subprocess.Popen", return_value=proc):
            code = doctor.run_doctor()
        assert code == 2
        # stderr should mention the failure
        captured = capsys.readouterr()
        assert "cua-driver" in captured.err.lower() or "health_report" in captured.err.lower()


# ── response-shape parsing ─────────────────────────────────────────────────


class TestResponseShapeParsing:


    def test_jsonrpc_error_response_exits_2(self, capsys):
        from tools.computer_use import doctor

        proc = _fake_proc_with_responses(
            {"jsonrpc": "2.0", "id": 1, "result": {}},
            {"jsonrpc": "2.0", "id": 2, "error": {"code": -32601, "message": "method not found"}},
        )
        with _pm_driver("/fake/cua-driver"), \
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
        with _pm_driver("/fake/cua-driver"), \
             patch("subprocess.Popen", return_value=proc), \
             patch("sys.stdout", new_callable=StringIO):
            doctor.run_doctor(include=["binary_version", "tcc_accessibility"])

        # Inspect the second write to stdin — the tools/call payload.
        writes = [call.args[0] for call in proc.stdin.write.call_args_list]
        call_payload = next(json.loads(w) for w in writes if "tools/call" in w)
        assert call_payload["params"]["arguments"]["include"] == [
            "binary_version", "tcc_accessibility",
        ]


# ── json output ────────────────────────────────────────────────────────────


class TestJsonOutput:
    def test_json_output_is_parseable_round_trip(self):
        from tools.computer_use import doctor

        proc = _fake_proc_with_responses(
            {"jsonrpc": "2.0", "id": 1, "result": {}},
            {"jsonrpc": "2.0", "id": 2, "result": {"structuredContent": _ok_report()}},
        )
        with _pm_driver("/fake/cua-driver"), \
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


# ── HERMES_CUA_DRIVER_CMD resolution ───────────────────────────────────────


class TestDriverCmdResolution:
    @staticmethod
    def _executable(path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("#!/bin/sh\nexit 0\n")
        path.chmod(0o755)
        return path

    def _inspected_binary(self, **kwargs) -> Path:
        from tools.computer_use import doctor

        with _pm_driver("/pm/store/cua-driver"), \
             patch("tools.computer_use.doctor._drive_health_report", return_value=_ok_report()) as health, \
             patch("sys.stdout", new_callable=StringIO):
            assert doctor.run_doctor(**kwargs) == 0
        return Path(health.call_args.args[0])

    def test_explicit_driver_cmd_arg_wins(self, tmp_path, monkeypatch):
        explicit = self._executable(tmp_path / "custom" / "cua-driver")
        monkeypatch.setenv("HERMES_CUA_DRIVER_CMD", str(self._executable(tmp_path / "env" / "cua-driver")))

        assert self._inspected_binary(driver_cmd=str(explicit)) == explicit

    def test_env_var_used_when_no_arg_given(self, tmp_path, monkeypatch):
        from_env = self._executable(tmp_path / "env" / "cua-driver")
        monkeypatch.setenv("HERMES_CUA_DRIVER_CMD", str(from_env))

        assert self._inspected_binary() == from_env

    def test_doctor_inspects_the_pm_selected_driver_not_path(self, tmp_path, monkeypatch):
        """Doctor must diagnose the driver the runtime invokes: PM's pin, not a PATH copy."""
        monkeypatch.delenv("HERMES_CUA_DRIVER_CMD", raising=False)
        monkeypatch.setenv("PATH", str(self._executable(tmp_path / "bin" / "cua-driver").parent))

        assert self._inspected_binary() == Path("/pm/store/cua-driver")


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
        with _pm_driver("/fake/cua-driver"), \
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
        with _pm_driver("/fake/cua-driver"), \
             patch("subprocess.Popen", return_value=proc), \
             patch.object(doctor, "_read_cli_version", return_value="cua-driver 0.5.8"), \
             patch("sys.stdout", new_callable=StringIO) as out:
            code = doctor.run_doctor(json_output=True)
        assert code == 0
        payload = json.loads(out.getvalue())
        assert payload["hermes_identity"]["version_mismatch"] is False


def test_failed_tcc_row_from_health_report_names_the_stale_row_reset_for_that_service():
    """A failed ``tcc_*`` row from the driver's own health_report (0.22+, not only the 0.10 fallback probes) must
    carry the stale-grant recovery for its own TCC service, because System Settings can show the toggle ON while
    the daemon is denied (trycua/cua#3170)."""
    from tools.computer_use import doctor

    report = _ok_report()
    report["checks"] = [{"name": "tcc_screen_recording", "status": "fail", "message": "Screen Recording is not granted.",
                         "hint": "Grant it in System Settings."},
                        {"name": "tcc_accessibility", "status": "pass", "message": "granted"}]
    proc = _fake_proc_with_responses(
        {"jsonrpc": "2.0", "id": 1, "result": {}},
        {"jsonrpc": "2.0", "id": 2, "result": {"structuredContent": report}},
    )
    with _pm_driver("/fake/cua-driver"), patch("subprocess.Popen", return_value=proc), \
         patch("sys.stdout", new_callable=StringIO) as out:
        doctor.run_doctor(json_output=True)
    checks = {c["name"]: c for c in json.loads(out.getvalue())["checks"]}

    assert checks["tcc_screen_recording"]["hint"].startswith("Grant it in System Settings.")
    assert "tccutil reset ScreenCapture com.trycua.driver" in checks["tcc_screen_recording"]["hint"]
    assert "reset Accessibility" not in checks["tcc_screen_recording"]["hint"]
    assert "tccutil" not in checks["tcc_accessibility"].get("hint", "")
