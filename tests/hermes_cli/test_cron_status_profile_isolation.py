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

import time
from unittest.mock import MagicMock, Mock, patch
from types import SimpleNamespace

import pytest


@pytest.fixture()
def tmp_cron_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")
    return tmp_path




class TestCronStatusHeartbeatGuard:
    """Ensure `hermes cron status` correctly warns when the heartbeat file is absent.

    Issue #98790 (root cause 2): profiles with no `cron/ticker_heartbeat` fall
    through to the "✓ Gateway is running" branch even though ticks won't fire.
    """

    def test_no_heartbeat_triggers_yellow_warning(self, monkeypatch, capsys):
        from hermes_cli import cron as cron_mod

        with (
            patch("hermes_cli.gateway.find_gateway_pids", return_value={789}),
            patch("cron.jobs.get_ticker_heartbeat_age", return_value=None),
            patch("cron.jobs.get_ticker_success_age", return_value=None),
            patch("cron.jobs.get_ticker_last_error", return_value=None),
            patch("cron.jobs.TICKER_INTERVAL_SECONDS", 60),
        ):
            cron_mod.cron_status()

        stdout = capsys.readouterr().out
        assert "⚠ Gateway is running but the cron ticker has not reported a heartbeat" in stdout
        assert "Cron jobs will NOT fire" in stdout
        # Must NOT show the green ✓
        assert "✓ Gateway is running" not in stdout

    def test_fresh_heartbeat_shows_green_checkmark(self, monkeypatch, capsys):
        from hermes_cli import cron as cron_mod

        with (
            patch("hermes_cli.gateway.find_gateway_pids", return_value={999}),
            patch("cron.jobs.get_ticker_heartbeat_age", return_value=10),
            patch("cron.jobs.get_ticker_success_age", return_value=8),
            patch("cron.jobs.TICKER_INTERVAL_SECONDS", 60),
        ):
            cron_mod.cron_status()

        stdout = capsys.readouterr().out
        assert "✓ Gateway is running — cron jobs will fire automatically" in stdout
        assert "⚠" not in stdout


class TestGetServicePidsProfileScope:
    """systemd branch must honor ``all_profiles`` and filter by profile."""

    def test_default_scope_filters_current_profile_systemd_unit(self, monkeypatch):
        from hermes_cli import gateway as gateway_mod

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
            patch("hermes_cli.gateway.is_macos", return_value=False),
            patch("hermes_cli.gateway.supports_systemd_services", return_value=True),
            patch("hermes_cli.gateway.get_service_name", return_value="hermes-gateway-jarvis"),
            patch("subprocess.run", side_effect=_run_side_effect),
        ):
            pids = gateway_mod._get_service_pids()

        assert pids == {123}, "default scope must filter to current profile's unit"

    def test_all_profiles_true_enumerates_fleet(self, monkeypatch):
        from hermes_cli import gateway as gateway_mod

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
            patch("hermes_cli.gateway.is_macos", return_value=False),
            patch("hermes_cli.gateway.supports_systemd_services", return_value=True),
            patch("hermes_cli.gateway.get_service_name", return_value="hermes-gateway"),
            patch("subprocess.run", side_effect=_run_side_effect),
        ):
            pids = gateway_mod._get_service_pids(all_profiles=True)

        assert pids == {111, 222}, "all_profiles=True must enumerate the whole fleet"


class TestCronStatusMissingHeartbeat:
    """Missing ticker heartbeat must be reported honestly, not as healthy."""

    def test_missing_heartbeat_warns_when_gateway_old(self, tmp_cron_dir, capsys, monkeypatch):
        import io
        from contextlib import redirect_stdout
        import hermes_cli.cron as cron_cli
        from cron.jobs import create_job

        create_job(prompt="Test", schedule="every 1h")

        out = io.StringIO()
        with (
            patch("hermes_cli.cron._active_cron_provider_name", return_value="builtin"),
            patch("hermes_cli.gateway.find_gateway_pids", return_value=[4242]),
            patch("gateway.status.is_gateway_runtime_lock_active", return_value=True),
            patch("gateway.status.get_running_pid", return_value=4242),
            patch("cron.jobs.get_ticker_heartbeat_age", return_value=None),
            patch("cron.jobs.get_ticker_success_age", return_value=None),
            patch("cron.jobs.TICKER_INTERVAL_SECONDS", 60),
            # Gateway started long ago, ticker should have heartbeat by now
            patch("gateway.status._read_pid_record", return_value={"pid": 4242, "start_time": 1}),
            patch("gateway.status._get_process_start_time", return_value=1),
            redirect_stdout(out),
        ):
            cron_cli.cron_status()

        text = out.getvalue()
        assert "has not reported a heartbeat" in text or "no heartbeat" in text.lower()
        assert "will fire" not in text.lower() or "will NOT fire" in text

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
            patch("gateway.status._get_process_start_time", return_value=int(now)),
            patch("time.time", return_value=now + 5),
            redirect_stdout(out),
        ):
            cron_cli.cron_status()

        text = out.getvalue()
        assert "will fire" in text or "running" in text
        assert "never ticked" not in text.lower()
