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

from hermes_cli.main import (
    _for_each_systemd_gateway_unit,
    _service_unit_supports_graceful_sigusr1_restart,
    _warn_incomplete_gateway_fleet_restart,
)


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

    def test_hermes_webui_units_are_included(self):
        # #95882 — hermes update restarted hermes-gateway* and hermes-serve*
        # units but left companion hermes-webui* (sharing the source tree) on
        # stale pre-update code, producing HTTP 409 until a manual restart.
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

        assert seen == [
            "hermes-webui",
            "hermes-webui-prod",
            "hermes-serve",
            "hermes-gateway",
        ]

    def test_hermes_webui_near_prefix_is_rejected(self):
        # A bare ``startswith("hermes-webui")`` is fine for the profile family,
        # but the existing strict exact/hyphenated gate must still reject a
        # hypothetical near-prefix unit such as ``hermes-webuictl.service``.
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

class TestFleetRestartBestEffort:
    def test_discovers_and_restarts_hermes_webui_units(self, monkeypatch):
        # #95882 — the user-facing boundary is that ``hermes update`` runs
        # ``systemctl list-units`` with the right globs and restarts the units.
        # Mock the subprocess boundary so this is testable on macOS/CI.
        from hermes_cli.update_cmd import _restart_systemd_gateway_units_best_effort

        calls: list[list[str]] = []
        class FakeResult:
            def __init__(self, returncode: int = 0, stdout: str = ""):
                self.returncode = returncode
                self.stdout = stdout

        def fake_run(cmd, **kwargs):
            calls.append(list(cmd))
            if "list-units" in cmd:
                return FakeResult(
                    stdout="\n".join(
                        [
                            "hermes-webui.service loaded active running",
                            "hermes-serve.service loaded active running",
                        ]
                    )
                )
            return FakeResult()

        monkeypatch.setattr(subprocess, "run", fake_run)
        failed: list[str] = []
        _restart_systemd_gateway_units_best_effort(failed)

        list_units_calls = [c for c in calls if "list-units" in c]
        assert all(
            any("hermes-webui*" == arg for arg in c) for c in list_units_calls
        ), f"missing hermes-webui* in list-units patterns: {list_units_calls}"

        restart_calls = [
            c for c in calls if "restart" in c and "hermes-webui" in c
        ]
        assert len(restart_calls) >= 1, f"no hermes-webui restart issued: {calls}"
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

