"""Tests for the macOS 26 gateway-fleet restart fixes.

Covers:
1. ``_run_ps_gateway_scan`` — falls back from the removed ``ps -A eww``
   form to ``ps -axww`` when the first invocation fails (macOS 26 dropped
   the SysV ``e`` flag; ``ps: illegal argument: eww``).
2. ``list_launchd_gateway_labels`` — enumerates the whole
   ``ai.hermes.gateway*`` fleet (default + per-profile labels), not just
   the current profile's label.
3. ``_get_service_pids`` macOS branch — collects PIDs from every loaded
   gateway label so the post-update manual sweep does not kill freshly
   restarted launchd services.
"""

import subprocess
from types import SimpleNamespace

import pytest

import hermes_cli.gateway as gateway_cli


def _run_result(stdout: str = "", returncode: int = 0) -> SimpleNamespace:
    return SimpleNamespace(stdout=stdout, returncode=returncode)


class TestRunPsGatewayScan:
    def test_prefers_historical_eww_form_when_it_works(self, monkeypatch):
        """On older systems ``ps -A eww`` still works — use it unchanged."""
        good = _run_result(stdout="123 cmd-one\n456 cmd-two\n")
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            return good

        monkeypatch.setattr(gateway_cli.subprocess, "run", fake_run)
        out = gateway_cli._run_ps_gateway_scan()
        assert out == "123 cmd-one\n456 cmd-two\n"
        assert calls == [["ps", "-A", "eww", "-o", "pid=,command="]]

    def test_falls_back_to_bsd_form_when_eww_fails(self, monkeypatch):
        """macOS 26 drops ``eww`` (illegal argument) — retry with ``-axww``."""
        eww_bad = _run_result(stdout="ps: illegal argument: eww\n", returncode=1)
        axww_good = _run_result(stdout="123 cmd-one\n456 cmd-two\n")
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            if cmd == ["ps", "-A", "eww", "-o", "pid=,command="]:
                return eww_bad
            return axww_good

        monkeypatch.setattr(gateway_cli.subprocess, "run", fake_run)
        out = gateway_cli._run_ps_gateway_scan()
        assert out == "123 cmd-one\n456 cmd-two\n"
        assert calls == [
            ["ps", "-A", "eww", "-o", "pid=,command="],
            ["ps", "-axww", "-o", "pid=,command="],
        ]

    def test_falls_back_when_eww_returns_empty_but_succeeds(self, monkeypatch):
        """Guard against a platform where eww exits 0 but prints nothing."""
        eww_empty = _run_result(stdout="")
        axww_good = _run_result(stdout="123 cmd-one\n")

        def fake_run(cmd, **kwargs):
            if cmd == ["ps", "-A", "eww", "-o", "pid=,command="]:
                return eww_empty
            return axww_good

        monkeypatch.setattr(gateway_cli.subprocess, "run", fake_run)
        assert gateway_cli._run_ps_gateway_scan() == "123 cmd-one\n"

    def test_returns_none_when_both_forms_fail(self, monkeypatch):
        def fake_run(cmd, **kwargs):
            return _run_result(stdout="", returncode=2)

        monkeypatch.setattr(gateway_cli.subprocess, "run", fake_run)
        assert gateway_cli._run_ps_gateway_scan() is None

    def test_scan_pids_uses_helper_and_finds_gateways(self, monkeypatch):
        """``_scan_gateway_pids`` parses the fallback output for gateway procs."""
        ps_out = (
            "472 /venv/bin/python -m hermes_cli.main --profile orchestrator gateway run\n"
            "484 /venv/bin/python -m hermes_cli.main --profile reviewer gateway run\n"
            "999 /usr/bin/launchd\n"
        )
        monkeypatch.setattr(
            gateway_cli, "_run_ps_gateway_scan", lambda: ps_out
        )
        from gateway import status as gateway_status
        monkeypatch.setattr(
            gateway_status,
            "looks_like_gateway_command_line",
            lambda cmd: "gateway run" in cmd,
        )
        monkeypatch.setattr(
            gateway_status,
            "looks_like_gateway_runtime_command_line",
            lambda cmd: "gateway run" in cmd,
        )
        pids = gateway_cli._scan_gateway_pids(set(), all_profiles=True)
        assert pids == [472, 484]


class TestListLaunchdGatewayLabels:
    def test_enumerates_default_and_profile_labels(self, monkeypatch):
        launchctl_out = (
            "472\t0\tai.hermes.gateway-orchestrator\n"
            "484\t0\tai.hermes.gateway-reviewer\n"
            "895\t0\tai.hermes.gateway\n"
            "485\t0\tai.hermes.gateway-coder\n"
            "123\t0\tcom.apple.something\n"
            "-   0\tcom.nousresearch.hermes-gateway\n"
        )
        monkeypatch.setattr(
            gateway_cli.subprocess,
            "run",
            lambda cmd, **kw: _run_result(stdout=launchctl_out),
        )
        labels = gateway_cli.list_launchd_gateway_labels()
        assert labels == [
            "ai.hermes.gateway-orchestrator",
            "ai.hermes.gateway-reviewer",
            "ai.hermes.gateway",
            "ai.hermes.gateway-coder",
        ]

    def test_returns_empty_on_launchctl_failure(self, monkeypatch):
        monkeypatch.setattr(
            gateway_cli.subprocess,
            "run",
            lambda cmd, **kw: _run_result(stdout="", returncode=1),
        )
        assert gateway_cli.list_launchd_gateway_labels() == []

    def test_returns_empty_on_file_not_found(self, monkeypatch):
        def fake_run(cmd, **kw):
            raise FileNotFoundError("no launchctl")

        monkeypatch.setattr(gateway_cli.subprocess, "run", fake_run)
        assert gateway_cli.list_launchd_gateway_labels() == []

    def test_dedupes_repeated_labels(self, monkeypatch):
        launchctl_out = (
            "895\t0\tai.hermes.gateway\n"
            "896\t0\tai.hermes.gateway\n"
        )
        monkeypatch.setattr(
            gateway_cli.subprocess,
            "run",
            lambda cmd, **kw: _run_result(stdout=launchctl_out),
        )
        labels = gateway_cli.list_launchd_gateway_labels()
        assert labels == ["ai.hermes.gateway"]


class TestGetServicePidsMacos:
    def test_collects_pids_from_all_loaded_labels(self, monkeypatch):
        """macOS branch must return every launchd gateway PID, not just the
        current profile's, so the update's manual sweep excludes them."""
        labels = [
            "ai.hermes.gateway",
            "ai.hermes.gateway-coder",
            "ai.hermes.gateway-trader",
        ]
        monkeypatch.setattr(gateway_cli, "list_launchd_gateway_labels", lambda: labels)
        monkeypatch.setattr(gateway_cli, "is_macos", lambda: True)
        monkeypatch.setattr(gateway_cli, "supports_systemd_services", lambda: False)

        def fake_run(cmd, **kw):
            label = cmd[-1]
            pid_by_label = {
                "ai.hermes.gateway": '    "PID" = 895;\n',
                "ai.hermes.gateway-coder": '    "PID" = 485;\n',
                "ai.hermes.gateway-trader": "485\t0\tai.hermes.gateway-trader\n",
            }
            return _run_result(stdout=pid_by_label[label])

        monkeypatch.setattr(gateway_cli.subprocess, "run", fake_run)
        pids = gateway_cli._get_service_pids()
        assert pids == {895, 485}

    def test_no_labels_means_no_pids(self, monkeypatch):
        monkeypatch.setattr(gateway_cli, "list_launchd_gateway_labels", lambda: [])
        monkeypatch.setattr(gateway_cli, "is_macos", lambda: True)
        monkeypatch.setattr(gateway_cli, "supports_systemd_services", lambda: False)
        assert gateway_cli._get_service_pids() == set()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
