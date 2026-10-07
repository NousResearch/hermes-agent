"""Regression for #68523 — one systemctl timeout must not abort fleet restarts.

On hosts with many profile-backed ``rabbit-gateway*.service`` units,
``rabbit update`` used to wrap the entire per-scope unit loop in a single
``except subprocess.TimeoutExpired``. A timeout on unit N skipped units
N+1…, leaving later gateways on pre-update in-memory modules while the
checkout on disk was already new (mixed-generation crashes).
"""

from __future__ import annotations

import subprocess

import pytest

from rabbit_cli.update_cmd import _for_each_systemd_gateway_unit, _service_unit_supports_graceful_sigusr1_restart, _warn_incomplete_gateway_fleet_restart


def _list_units_stdout(names: list[str]) -> str:
    return "\n".join(f"{name}.service loaded active running" for name in names)


class TestFleetRestartTimeoutIsolation:
    def test_timeout_on_middle_unit_continues_remaining_units(self):
        units = [
            "rabbit-gateway-xiaomo1",
            "rabbit-gateway-xiaomo2",
            "rabbit-gateway-xiaomo3",
            "rabbit-gateway-xiaomo4",
            "rabbit-gateway-xiaomo5",
            "rabbit-gateway-xiaomo6",
            "rabbit-gateway-xiaomo7",
            "rabbit-gateway",
        ]
        restarted: list[str] = []
        failed: list[str] = []
        timeout_cmds: list = []

        def process_unit(svc_name: str) -> None:
            if svc_name == "rabbit-gateway-xiaomo5":
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

        assert failed == ["rabbit-gateway-xiaomo5"]
        assert restarted == [
            "rabbit-gateway-xiaomo1",
            "rabbit-gateway-xiaomo2",
            "rabbit-gateway-xiaomo3",
            "rabbit-gateway-xiaomo4",
            "rabbit-gateway-xiaomo6",
            "rabbit-gateway-xiaomo7",
            "rabbit-gateway",
        ]
        assert set(restarted) | set(failed) == set(units)
        assert timeout_cmds == [
            ["systemctl", "--user", "--no-ask-password", "restart", "rabbit-gateway-xiaomo5"]
        ]

    def test_non_gateway_units_in_list_output_are_ignored(self):
        seen: list[str] = []

        _for_each_systemd_gateway_unit(
            "\n".join(
                [
                    "ssh.service loaded active running",
                    "rabbit-gateway-coder.service loaded active running",
                    "not-a-service loaded active running",
                    "",
                ]
            ),
            process_unit=seen.append,
            on_unit_timeout=lambda *_: pytest.fail("unexpected timeout"),
        )

        assert seen == ["rabbit-gateway-coder"]

    def test_rabbit_serve_units_are_included(self):
        # #83438 — rabbit update restarted rabbit-gateway* units but left
        # rabbit-serve* (the Desktop app's backend) on stale pre-update code.
        seen: list[str] = []

        _for_each_systemd_gateway_unit(
            "\n".join(
                [
                    "ssh.service loaded active running",
                    "rabbit-serve.service loaded active running",
                    "rabbit-serve-work.service loaded active running",
                    "rabbit-gateway.service loaded active running",
                    "",
                ]
            ),
            process_unit=seen.append,
            on_unit_timeout=lambda *_: pytest.fail("unexpected timeout"),
        )

        assert seen == ["rabbit-serve", "rabbit-serve-work", "rabbit-gateway"]

    def test_rabbit_dashboard_units_are_included(self):
        # #125297 — the same blind spot for the systemd-supervised dashboard: the
        # unit pass skipped rabbit-dashboard*, so a successful update left the
        # dashboard on pre-update code with outcome "deferred" and nothing
        # restarted it. Reconciliation already credits rabbit-dashboard{,-<profile>}
        # unit restarts; the pass must produce one.
        seen: list[str] = []

        _for_each_systemd_gateway_unit(
            "\n".join(
                [
                    "ssh.service loaded active running",
                    "rabbit-dashboard.service loaded active running",
                    "rabbit-dashboard-work.service loaded active running",
                    "rabbit-serve.service loaded active running",
                    "",
                ]
            ),
            process_unit=seen.append,
            on_unit_timeout=lambda *_: pytest.fail("unexpected timeout"),
        )

        assert seen == ["rabbit-dashboard", "rabbit-dashboard-work", "rabbit-serve"]

    def test_rabbit_dashboard_near_prefix_is_rejected(self):
        # Same strict shape on the dashboard side: a bare
        # ``startswith("rabbit-dashboard")`` gate would also accept the
        # unrelated ``rabbit-dashboardd.service``.
        seen: list[str] = []

        _for_each_systemd_gateway_unit(
            _list_units_stdout(["rabbit-dashboardd", "rabbit-dashboard-work"]),
            process_unit=seen.append,
            on_unit_timeout=lambda *_: pytest.fail("unexpected timeout"),
        )

        assert seen == ["rabbit-dashboard-work"]

    def test_rabbit_server_near_prefix_is_rejected(self):
        # Review on #83595: a bare ``startswith("rabbit-serve")`` gate also
        # accepts the unrelated ``rabbit-server.service``. Only the exact
        # base unit or the hyphenated profile family should pass.
        seen: list[str] = []

        _for_each_systemd_gateway_unit(
            _list_units_stdout(["rabbit-server"]),
            process_unit=seen.append,
            on_unit_timeout=lambda *_: pytest.fail("unexpected timeout"),
        )

        assert seen == []

    def test_rabbit_gateway_near_prefix_is_rejected(self):
        # Same strict shape on the gateway side: profile units are
        # ``rabbit-gateway-<profile>``, so a hypothetical
        # ``rabbit-gatewayd.service`` must not enter the restart path.
        seen: list[str] = []

        _for_each_systemd_gateway_unit(
            _list_units_stdout(["rabbit-gatewayd", "rabbit-gateway-coder"]),
            process_unit=seen.append,
            on_unit_timeout=lambda *_: pytest.fail("unexpected timeout"),
        )

        assert seen == ["rabbit-gateway-coder"]


class TestGracefulSigusr1Eligibility:
    def test_gateway_units_are_eligible(self):
        assert _service_unit_supports_graceful_sigusr1_restart("rabbit-gateway")
        assert _service_unit_supports_graceful_sigusr1_restart(
            "rabbit-gateway-work"
        )

    def test_serve_units_are_not_eligible(self):
        # rabbit-serve doesn't run gateway/run.py, so it never installs the
        # SIGUSR1 handler — sending it the signal would just terminate the
        # process (the default action) instead of draining gracefully.
        assert not _service_unit_supports_graceful_sigusr1_restart("rabbit-serve")
        assert not _service_unit_supports_graceful_sigusr1_restart(
            "rabbit-serve-work"
        )

    def test_process_errors_other_than_timeout_still_propagate(self):
        def process_unit(_svc_name: str) -> None:
            raise RuntimeError("not a timeout")

        with pytest.raises(RuntimeError, match="not a timeout"):
            _for_each_systemd_gateway_unit(
                _list_units_stdout(["rabbit-gateway"]),
                process_unit=process_unit,
                on_unit_timeout=lambda *_: pytest.fail("timeout handler must not run"),
            )


class TestIncompleteFleetRestartWarning:
    def test_warns_with_exact_unrestarted_units(self, capsys):
        _warn_incomplete_gateway_fleet_restart(
            ["rabbit-gateway-xiaomo5", "rabbit-gateway-xiaomo6", "rabbit-gateway-xiaomo5"]
        )
        out = capsys.readouterr().out
        assert "Update incomplete" in out
        assert out.count("rabbit-gateway-xiaomo5") == 1
        assert "rabbit-gateway-xiaomo6" in out
        assert "pre-update code" in out

