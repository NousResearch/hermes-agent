"""Regression for #68523 — one systemctl timeout must not abort fleet restarts.

On hosts with many profile-backed ``hermes-gateway*.service`` units,
``hermes update`` used to wrap the entire per-scope unit loop in a single
``except subprocess.TimeoutExpired``. A timeout on unit N skipped units
N+1…, leaving later gateways on pre-update in-memory modules while the
checkout on disk was already new (mixed-generation crashes).
"""

from __future__ import annotations

import subprocess

import pytest

from hermes_cli.update_cmd import _for_each_systemd_gateway_unit, _service_unit_supports_graceful_sigusr1_restart, _warn_incomplete_gateway_fleet_restart


def _list_units_stdout(names: list[str]) -> str:
    return "\n".join(f"{name}.service loaded active running" for name in names)


class TestFleetRestartTimeoutIsolation:
    def test_timeout_on_middle_unit_continues_remaining_units(self):
        units = [
            "hermes-gateway-xiaomo1",
            "hermes-gateway-xiaomo2",
            "hermes-gateway-xiaomo3",
            "hermes-gateway-xiaomo4",
            "hermes-gateway-xiaomo5",
            "hermes-gateway-xiaomo6",
            "hermes-gateway-xiaomo7",
            "hermes-gateway",
        ]
        restarted: list[str] = []
        failed: list[str] = []
        timeout_cmds: list = []

        def process_unit(svc_name: str) -> None:
            if svc_name == "hermes-gateway-xiaomo5":
                raise subprocess.TimeoutExpired(
                    cmd=["systemctl", "--user", "--no-ask-password", "restart", svc_name],
                    timeout=15,
                )
            restarted.append(svc_name)

        def on_unit_timeout(svc_name: str, exc: subprocess.TimeoutExpired) -> None:
            failed.append(svc_name)
            timeout_cmds.append(exc.cmd)

        _for_each_systemd_gateway_unit(
            _list_units_stdout(units),
            process_unit=process_unit,
            on_unit_timeout=on_unit_timeout,
        )

        assert failed == ["hermes-gateway-xiaomo5"]
        assert restarted == [
            "hermes-gateway-xiaomo1",
            "hermes-gateway-xiaomo2",
            "hermes-gateway-xiaomo3",
            "hermes-gateway-xiaomo4",
            "hermes-gateway-xiaomo6",
            "hermes-gateway-xiaomo7",
            "hermes-gateway",
        ]
        assert set(restarted) | set(failed) == set(units)
        assert timeout_cmds == [
            ["systemctl", "--user", "--no-ask-password", "restart", "hermes-gateway-xiaomo5"]
        ]

    def test_non_gateway_units_in_list_output_are_ignored(self):
        seen: list[str] = []

        _for_each_systemd_gateway_unit(
            "\n".join(
                [
                    "ssh.service loaded active running",
                    "hermes-gateway-coder.service loaded active running",
                    "not-a-service loaded active running",
                    "",
                ]
            ),
            process_unit=seen.append,
            on_unit_timeout=lambda *_: pytest.fail("unexpected timeout"),
        )

        assert seen == ["hermes-gateway-coder"]

    def test_hermes_serve_units_are_included(self):
        # #83438 — hermes update restarted hermes-gateway* units but left
        # hermes-serve* (the Desktop app's backend) on stale pre-update code.
        seen: list[str] = []

        _for_each_systemd_gateway_unit(
            "\n".join(
                [
                    "ssh.service loaded active running",
                    "hermes-serve.service loaded active running",
                    "hermes-serve-work.service loaded active running",
                    "hermes-gateway.service loaded active running",
                    "",
                ]
            ),
            process_unit=seen.append,
            on_unit_timeout=lambda *_: pytest.fail("unexpected timeout"),
        )

        assert seen == ["hermes-serve", "hermes-serve-work", "hermes-gateway"]

    def test_hermes_dashboard_units_are_included(self):
        # #125297 — the same blind spot for the systemd-supervised dashboard: the
        # unit pass skipped hermes-dashboard*, so a successful update left the
        # dashboard on pre-update code with outcome "deferred" and nothing
        # restarted it. Reconciliation already credits hermes-dashboard{,-<profile>}
        # unit restarts; the pass must produce one.
        seen: list[str] = []

        _for_each_systemd_gateway_unit(
            "\n".join(
                [
                    "ssh.service loaded active running",
                    "hermes-dashboard.service loaded active running",
                    "hermes-dashboard-work.service loaded active running",
                    "hermes-serve.service loaded active running",
                    "",
                ]
            ),
            process_unit=seen.append,
            on_unit_timeout=lambda *_: pytest.fail("unexpected timeout"),
        )

        assert seen == ["hermes-dashboard", "hermes-dashboard-work", "hermes-serve"]

    def test_hermes_dashboard_near_prefix_is_rejected(self):
        # Same strict shape on the dashboard side: a bare
        # ``startswith("hermes-dashboard")`` gate would also accept the
        # unrelated ``hermes-dashboardd.service``.
        seen: list[str] = []

        _for_each_systemd_gateway_unit(
            _list_units_stdout(["hermes-dashboardd", "hermes-dashboard-work"]),
            process_unit=seen.append,
            on_unit_timeout=lambda *_: pytest.fail("unexpected timeout"),
        )

        assert seen == ["hermes-dashboard-work"]

    def test_hermes_webui_units_are_included(self):
        # #95882: companion WebUI units must not retain stale pre-update code.
        seen: list[str] = []

        _for_each_systemd_gateway_unit(
            "\n".join(
                [
                    "ssh.service loaded active running",
                    "hermes-webui.service loaded active running",
                    "hermes-webui-prod.service loaded active running",
                    "hermes-serve.service loaded active running",
                    "hermes-gateway.service loaded active running",
                    "",
                ]
            ),
            process_unit=seen.append,
            on_unit_timeout=lambda *_: pytest.fail("unexpected timeout"),
        )

        assert seen == ["hermes-webui", "hermes-webui-prod", "hermes-serve", "hermes-gateway"]

    def test_hermes_webui_near_prefix_is_rejected(self):
        # A bare prefix would also accept the unrelated hermes-webuictl unit.
        seen: list[str] = []

        _for_each_systemd_gateway_unit(
            _list_units_stdout(["hermes-webuictl", "hermes-webui-coder"]),
            process_unit=seen.append,
            on_unit_timeout=lambda *_: pytest.fail("unexpected timeout"),
        )

        assert seen == ["hermes-webui-coder"]

    def test_hermes_server_near_prefix_is_rejected(self):
        # Review on #83595: a bare ``startswith("hermes-serve")`` gate also
        # accepts the unrelated ``hermes-server.service``. Only the exact
        # base unit or the hyphenated profile family should pass.
        seen: list[str] = []

        _for_each_systemd_gateway_unit(
            _list_units_stdout(["hermes-server"]),
            process_unit=seen.append,
            on_unit_timeout=lambda *_: pytest.fail("unexpected timeout"),
        )

        assert seen == []

    def test_hermes_gateway_near_prefix_is_rejected(self):
        # Same strict shape on the gateway side: profile units are
        # ``hermes-gateway-<profile>``, so a hypothetical
        # ``hermes-gatewayd.service`` must not enter the restart path.
        seen: list[str] = []

        _for_each_systemd_gateway_unit(
            _list_units_stdout(["hermes-gatewayd", "hermes-gateway-coder"]),
            process_unit=seen.append,
            on_unit_timeout=lambda *_: pytest.fail("unexpected timeout"),
        )

        assert seen == ["hermes-gateway-coder"]


class TestFleetRestartBoundary:
    def test_discovers_and_restarts_hermes_webui_units(self, monkeypatch, tmp_path):
        # Exercise current discovery and per-unit restart without faking the OS.
        # Only subprocess responses and the test's home locations are controlled.
        from hermes_cli.update_cmd_fleet import (
            _restart_one_systemd_gateway_unit,
            _systemd_gateway_unit_listings,
        )

        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
        monkeypatch.setattr("hermes_cli.gateway._SYSTEM_UNIT_DIR", tmp_path / "system")
        calls: list[list[str]] = []

        def fake_run(cmd, **kwargs):
            calls.append(list(cmd))
            if "list-units" in cmd:
                return subprocess.CompletedProcess(
                    cmd, 0, stdout="\n".join(
                        [
                            "hermes-webui.service loaded active running",
                            "hermes-webui-prod.service loaded active running",
                            "hermes-webui-foreign.service loaded active running",
                            "hermes-webuictl.service loaded active running",
                            "hermes-serve.service loaded active running",
                        ]
                    )
                )
            stdout = "active\n" if "is-active" in cmd else ""
            if "--property=MainPID" in cmd:
                stdout = "0\n"
            if "--property=Environment" in cmd:
                home = tmp_path / ("foreign" if "hermes-webui-foreign" in cmd else "hermes")
                stdout = f"HERMES_HOME={home}\n"
            return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        failed: list[str] = []
        restarted: list[str] = []
        for scope, scope_cmd, result in _systemd_gateway_unit_listings():
            _for_each_systemd_gateway_unit(
                result.stdout,
                process_unit=lambda name: _restart_one_systemd_gateway_unit(
                    name, scope=scope, scope_cmd=scope_cmd, drain_budget=5.0,
                    _manage_cmd_cache={scope: scope_cmd + ["--no-ask-password"]},
                    restarted_services=restarted, failed_or_stale_units=failed,
                ),
                on_unit_timeout=lambda *_: pytest.fail("unexpected timeout"),
            )

        list_units_calls = [c for c in calls if "list-units" in c]
        assert list_units_calls == [
            prefix + [
                "list-units", "hermes-gateway*", "hermes-serve*",
                "hermes-dashboard*", "hermes-webui*", "--plain", "--no-legend", "--no-pager",
            ]
            for prefix in (["systemctl", "--user"], ["systemctl"])
        ]

        restart_calls = [
            c for c in calls if "restart" in c and c[-1].startswith("hermes-webui")
        ]
        assert restart_calls == [
            prefix + ["--no-ask-password", "restart", name]
            for prefix in (["systemctl", "--user"], ["systemctl"])
            for name in ("hermes-webui", "hermes-webui-prod")
        ]
        for name in ("hermes-webui", "hermes-webui-prod"):
            assert [c for c in calls if c[-2:] == ["is-active", name]] == [
                prefix + ["is-active", name]
                for prefix in (["systemctl", "--user"], ["systemctl"])
                for _ in range(2)  # Before restart and after the new process starts.
            ]
        assert restarted == ["hermes-webui", "hermes-webui-prod", "hermes-serve"] * 2
        assert failed == []


class TestGracefulSigusr1Eligibility:
    def test_gateway_units_are_eligible(self):
        assert _service_unit_supports_graceful_sigusr1_restart("hermes-gateway")
        assert _service_unit_supports_graceful_sigusr1_restart(
            "hermes-gateway-work"
        )

    def test_serve_units_are_not_eligible(self):
        # hermes-serve doesn't run gateway/run.py, so it never installs the
        # SIGUSR1 handler — sending it the signal would just terminate the
        # process (the default action) instead of draining gracefully.
        assert not _service_unit_supports_graceful_sigusr1_restart("hermes-serve")
        assert not _service_unit_supports_graceful_sigusr1_restart(
            "hermes-serve-work"
        )

    def test_webui_units_are_not_eligible(self):
        assert not _service_unit_supports_graceful_sigusr1_restart("hermes-webui")
        assert not _service_unit_supports_graceful_sigusr1_restart("hermes-webui-prod")

    def test_process_errors_other_than_timeout_still_propagate(self):
        def process_unit(_svc_name: str) -> None:
            raise RuntimeError("not a timeout")

        with pytest.raises(RuntimeError, match="not a timeout"):
            _for_each_systemd_gateway_unit(
                _list_units_stdout(["hermes-gateway"]),
                process_unit=process_unit,
                on_unit_timeout=lambda *_: pytest.fail("timeout handler must not run"),
            )


class TestIncompleteFleetRestartWarning:
    def test_warns_with_exact_unrestarted_units(self, capsys):
        _warn_incomplete_gateway_fleet_restart(
            ["hermes-gateway-xiaomo5", "hermes-gateway-xiaomo6", "hermes-gateway-xiaomo5"]
        )
        out = capsys.readouterr().out
        assert "Update incomplete" in out
        assert out.count("hermes-gateway-xiaomo5") == 1
        assert "hermes-gateway-xiaomo6" in out
        assert "pre-update code" in out
