"""Tests for the _spawn_gateway_restart repeat-request cooldown (#89034).

A finished ``gateway-restart`` child does not mean the gateway is back, so the
pre-existing "reuse the in-flight child" guard stops coalescing exactly when
repeat requests are most harmful.  A stale cached dashboard frontend re-firing
its restart every few seconds therefore produced a fresh restart every time
(the reporter measured 77, 17 of them inside one minute), killing the gateway
often enough mid-FTS5-write to corrupt ``state.db``.
"""
from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def reset_restart_cooldown():
    """Keep the module-level cooldown state out of neighbouring tests."""
    import hermes_cli.web_server as web_server

    web_server._LAST_GATEWAY_RESTART = None
    yield
    web_server._LAST_GATEWAY_RESTART = None


def _exited_proc(pid: int = 4242) -> MagicMock:
    """A Popen handle for a child that has already finished."""
    proc = MagicMock(spec=subprocess.Popen)
    proc.poll.return_value = 0
    proc.pid = pid
    return proc


class TestRepeatRestartWithinCooldown:
    """Repeats within the window ride the previous spawn."""

    @patch(
        "hermes_cli.web_server._gateway_subcommand",
        return_value=["gateway", "restart"],
    )
    @patch("hermes_cli.web_server._spawn_hermes_action")
    @patch("hermes_cli.web_server._ACTION_PROCS", {})
    def test_second_request_after_the_child_exits_is_coalesced(
        self, mock_spawn, mock_subcmd
    ):
        """The exact #89034 shape: child gone, gateway still coming up."""
        from hermes_cli.web_server import _spawn_gateway_restart

        proc = _exited_proc()
        mock_spawn.return_value = proc

        with patch("hermes_cli.gateway._reap_unsupervised_gateway_orphans"), patch(
            "hermes_cli.web_server.time.monotonic", side_effect=[100.0, 103.5]
        ):
            first, first_reused = _spawn_gateway_restart()
            second, second_reused = _spawn_gateway_restart()

        assert mock_spawn.call_count == 1, (
            "a repeat request 3.5s later must not start a second restart"
        )
        assert first is proc and first_reused is False
        assert second is proc and second_reused is True

    @patch(
        "hermes_cli.web_server._gateway_subcommand",
        return_value=["gateway", "restart"],
    )
    @patch("hermes_cli.web_server._spawn_hermes_action")
    @patch("hermes_cli.web_server._ACTION_PROCS", {})
    def test_a_storm_of_requests_produces_exactly_one_restart(
        self, mock_spawn, mock_subcmd
    ):
        """17 requests inside one minute, the reported burst rate."""
        from hermes_cli.web_server import _spawn_gateway_restart

        mock_spawn.return_value = _exited_proc()
        # One read to stamp the spawn, then one per coalesced repeat.  The
        # window is anchored to the SPAWN, not to the previous request:
        # anchoring it to the previous request would let a 3.5s-spaced storm
        # walk the deadline forward forever and never restart at all.
        clock = [100.0, 103.5, 107.0]

        with patch("hermes_cli.gateway._reap_unsupervised_gateway_orphans"), patch(
            "hermes_cli.web_server.time.monotonic", side_effect=clock
        ):
            _spawn_gateway_restart()
            _spawn_gateway_restart()
            _spawn_gateway_restart()

        assert mock_spawn.call_count == 1

    @patch(
        "hermes_cli.web_server._gateway_subcommand",
        return_value=["gateway", "restart"],
    )
    @patch("hermes_cli.web_server._spawn_hermes_action")
    @patch("hermes_cli.web_server._ACTION_PROCS", {})
    def test_cooldown_survives_the_action_table_being_cleared(
        self, mock_spawn, mock_subcmd
    ):
        """The guard must not live in ``_ACTION_PROCS``.

        Completed action children get reaped out of that table, and a guard
        that disappears when the child is reaped is the bug this fixes.
        """
        import hermes_cli.web_server as web_server
        from hermes_cli.web_server import _spawn_gateway_restart

        mock_spawn.return_value = _exited_proc()

        with patch("hermes_cli.gateway._reap_unsupervised_gateway_orphans"), patch(
            "hermes_cli.web_server.time.monotonic", side_effect=[100.0, 102.0]
        ):
            _spawn_gateway_restart()
            web_server._ACTION_PROCS.clear()
            web_server._ACTION_COMMANDS.clear()
            _, reused = _spawn_gateway_restart()

        assert mock_spawn.call_count == 1
        assert reused is True


class TestCooldownReleases:
    """The window always expires; it never wedges the restart action."""

    @patch(
        "hermes_cli.web_server._gateway_subcommand",
        return_value=["gateway", "restart"],
    )
    @patch("hermes_cli.web_server._spawn_hermes_action")
    @patch("hermes_cli.web_server._ACTION_PROCS", {})
    def test_request_after_the_window_starts_a_real_restart(
        self, mock_spawn, mock_subcmd
    ):
        from hermes_cli.web_server import _spawn_gateway_restart

        mock_spawn.side_effect = [_exited_proc(1), _exited_proc(2)]

        with patch("hermes_cli.gateway._reap_unsupervised_gateway_orphans"), patch(
            "hermes_cli.web_server.time.monotonic",
            side_effect=[100.0, 111.0, 111.0],
        ):
            _spawn_gateway_restart()
            second, reused = _spawn_gateway_restart()

        assert mock_spawn.call_count == 2
        assert reused is False
        assert second.pid == 2

    @patch("hermes_cli.web_server._spawn_hermes_action")
    @patch("hermes_cli.web_server._ACTION_PROCS", {})
    def test_a_different_profile_is_never_coalesced(self, mock_spawn):
        """Two profiles are two services; one's restart is not the other's."""
        from hermes_cli.web_server import _spawn_gateway_restart

        mock_spawn.side_effect = [_exited_proc(1), _exited_proc(2)]

        with patch("hermes_cli.gateway._reap_unsupervised_gateway_orphans"), patch(
            "hermes_cli.web_server.time.monotonic",
            side_effect=[100.0, 101.0],
        ), patch(
            "hermes_cli.web_server._gateway_subcommand",
            side_effect=[["gateway", "restart"], ["-p", "coder", "gateway", "restart"]],
        ):
            _spawn_gateway_restart()
            second, reused = _spawn_gateway_restart(profile="coder")

        assert mock_spawn.call_count == 2
        assert reused is False
        assert second.pid == 2


class TestExistingBehaviourIsPreserved:
    """Regression guards on the pre-existing in-flight reuse."""

    @patch(
        "hermes_cli.web_server._gateway_subcommand",
        return_value=["gateway", "restart"],
    )
    @patch("hermes_cli.web_server._spawn_hermes_action")
    def test_live_child_is_still_reused_without_consulting_the_clock(
        self, mock_spawn, mock_subcmd
    ):
        from hermes_cli.web_server import _spawn_gateway_restart

        live = MagicMock(spec=subprocess.Popen)
        live.poll.return_value = None
        live.pid = 7

        with patch(
            "hermes_cli.web_server._ACTION_PROCS", {"gateway-restart": live}
        ), patch(
            "hermes_cli.web_server._ACTION_COMMANDS",
            {"gateway-restart": ("gateway", "restart")},
        ), patch(
            "hermes_cli.gateway._reap_unsupervised_gateway_orphans"
        ):
            proc, reused = _spawn_gateway_restart()

        assert proc is live
        assert reused is True
        mock_spawn.assert_not_called()

    @patch(
        "hermes_cli.web_server._gateway_subcommand",
        return_value=["gateway", "restart"],
    )
    @patch("hermes_cli.web_server._spawn_hermes_action")
    def test_live_child_for_another_profile_still_raises(self, mock_spawn, mock_subcmd):
        from hermes_cli.web_server import _spawn_gateway_restart

        live = MagicMock(spec=subprocess.Popen)
        live.poll.return_value = None

        with patch(
            "hermes_cli.web_server._ACTION_PROCS", {"gateway-restart": live}
        ), patch(
            "hermes_cli.web_server._ACTION_COMMANDS",
            {"gateway-restart": ("-p", "coder", "gateway", "restart")},
        ), patch(
            "hermes_cli.gateway._reap_unsupervised_gateway_orphans"
        ):
            with pytest.raises(RuntimeError, match="another profile"):
                _spawn_gateway_restart()

        mock_spawn.assert_not_called()
