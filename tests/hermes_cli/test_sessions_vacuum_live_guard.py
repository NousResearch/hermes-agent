"""Refuse `hermes sessions optimize` VACUUM while gateway/dashboard live (#84525)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from hermes_cli import sessions_cmd


def test_live_holders_reports_gateway_and_dashboard():
    with (
        patch("gateway.status.is_gateway_running", return_value=True),
        patch(
            "hermes_cli.dashboard_procs._scan_dashboard_processes",
            return_value=[(1234, "hermes dashboard")],
        ),
    ):
        assert sessions_cmd._live_session_db_holders() == [
            "gateway",
            "dashboard/serve",
        ]


def test_optimize_refuses_when_gateway_running(capsys):
    with patch.object(
        sessions_cmd,
        "_live_session_db_holders",
        return_value=["gateway"],
    ):
        assert sessions_cmd._refuse_vacuum_if_live_holders() is True
        out = capsys.readouterr().out
        assert "refusing VACUUM" in out
        assert "gateway" in out
        assert "hermes gateway stop" in out


def test_optimize_allows_when_no_holders(capsys):
    with patch.object(sessions_cmd, "_live_session_db_holders", return_value=[]):
        assert sessions_cmd._refuse_vacuum_if_live_holders() is False
        assert capsys.readouterr().out == ""
