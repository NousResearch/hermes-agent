"""Regression guard for #98790 — profile isolation in cron status.

Issue #98790: `hermes cron status` in profile B reports profile A's gateway
PID as proof that B's jobs will fire. Root causes:

1. systemd branch of ``_get_service_pids()`` returns every hermes-gateway*
   unit's MainPID even when ``all_profiles=False``, contradicting its
   docstring («Default-scope callers keep seeing only the current profile's
   service») and ``find_gateway_pids()``'s contract.

2. Missing ticker heartbeat is reported healthy: when a profile's ticker has
   never run (``ticker_heartbeat`` absent -> ``hb_age is None``), the stalled
   branch requires ``hb_age is not None``, so the profile falls through to
   «Gateway is running — cron jobs will fire automatically».

The fix: filter systemd units by the current profile's service name when
``all_profiles=False``, and treat a missing heartbeat as «ticker has never
ticked» unless the gateway process itself started less than STALE_AFTER ago
(fresh restart grace period).
"""

from __future__ import annotations
from gateway import service_identity
from gateway import systemd_runtime

import time
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def tmp_cron_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")
    return tmp_path






class TestGetServicePidsProfileScope:
    """systemd branch must honor ``all_profiles`` and filter by profile."""

    def test_default_scope_filters_current_profile_systemd_unit(self, monkeypatch):
        from gateway import process_discovery as gateway_mod

        def _run_side_effect(args, **kwargs):
            cmd_str = " ".join(str(a) for a in args)
            if "list-units" in cmd_str:
                # systemctl now filters: return only the unit that matches the pattern
                if "hermes-gateway-jarvis" in cmd_str:
                    stdout = "hermes-gateway-jarvis.service loaded active running\n"
                elif "hermes-gateway-coder" in cmd_str:
                    stdout = "hermes-gateway-coder.service loaded active running\n"
                elif "hermes-gateway*" in cmd_str:
                    stdout = (
                        "hermes-gateway-jarvis.service loaded active running\n"
                        "hermes-gateway-coder.service loaded active running\n"
                    )
                else:
                    stdout = ""
                return MagicMock(returncode=0, stdout=stdout, stderr="")
            if "show" in cmd_str and "MainPID" in cmd_str:
                if "hermes-gateway-jarvis" in cmd_str:
                    return MagicMock(returncode=0, stdout="123\n", stderr="")
                elif "hermes-gateway-coder" in cmd_str:
                    return MagicMock(returncode=0, stdout="456\n", stderr="")
            return MagicMock(returncode=0, stdout="", stderr="")

        with (
            patch("gateway.process_discovery.is_macos", return_value=False),
            patch("gateway.process_discovery._systemd_runtime.supports_services", return_value=True),
            patch("gateway.process_discovery._service_identity.service_name", return_value="hermes-gateway-jarvis"),
            patch("subprocess.run", side_effect=_run_side_effect),
        ):
            pids = gateway_mod._get_service_pids()

        assert pids == {123}, "default scope must filter to current profile's unit"

    def test_all_profiles_true_enumerates_fleet(self, monkeypatch):
        from gateway import process_discovery as gateway_mod

        def _run_side_effect(args, **kwargs):
            cmd_str = " ".join(str(a) for a in (args[:4] if args else []))
            if "list-units" in cmd_str:
                return MagicMock(
                    returncode=0,
                    stdout=(
                        "hermes-gateway.service loaded active running\n"
                        "hermes-gateway-profile-b.service loaded active running\n"
                    ),
                    stderr="",
                )
            if "show" in cmd_str and "MainPID" in cmd_str:
                if "hermes-gateway.service" in " ".join(args):
                    return MagicMock(returncode=0, stdout="111\n", stderr="")
                elif "hermes-gateway-profile-b.service" in " ".join(args):
                    return MagicMock(returncode=0, stdout="222\n", stderr="")
            return MagicMock(returncode=0, stdout="", stderr="")

        with (
            patch("gateway.process_discovery.is_macos", return_value=False),
            patch("gateway.process_discovery._systemd_runtime.supports_services", return_value=True),
            patch("gateway.process_discovery._service_identity.service_name", return_value="hermes-gateway"),
            patch("subprocess.run", side_effect=_run_side_effect),
        ):
            pids = gateway_mod._get_service_pids(all_profiles=True)

        assert pids == {111, 222}, "all_profiles=True must enumerate the whole fleet"


class TestCronStatusMissingHeartbeat:
    """Missing ticker heartbeat must be reported honestly, not as healthy."""


    def test_missing_heartbeat_green_when_gateway_just_started(self, tmp_cron_dir, capsys, monkeypatch):
        import io
        from contextlib import redirect_stdout
        import hermes_cli.cron as cron_cli
        from cron.jobs import create_job

        create_job(prompt="Test", schedule="every 1h")

        now = time.time()
        out = io.StringIO()
        with (
            patch("hermes_cli.cron._active_cron_provider_name", return_value="builtin"),
            patch("hermes_cli.gateway.find_gateway_pids", return_value=[4242]),
            patch("gateway.status.is_gateway_runtime_lock_active", return_value=True),
            patch("gateway.status.get_running_pid", return_value=4242),
            patch("cron.jobs.get_ticker_heartbeat_age", return_value=None),
            patch("cron.jobs.get_ticker_success_age", return_value=None),
            patch("cron.jobs.TICKER_INTERVAL_SECONDS", 60),
            # Gateway started a few seconds ago, ticker hasn't had its first tick yet
            patch("gateway.status._read_pid_record", return_value={"pid": 4242, "start_time": int(now)}),
            patch("runtime.process_identity.get_process_start_time", return_value=int(now)),
            patch("time.time", return_value=now + 5),
            redirect_stdout(out),
        ):
            cron_cli.cron_status()

        text = out.getvalue()
        assert "will fire" in text or "running" in text
        assert "never ticked" not in text.lower()
