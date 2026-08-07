"""Tests for ``hermes egress health`` subcommand."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

from hermes_cli.proxy_cli import cmd_health


def _args(watch=False, timeout=0):
    ns = argparse.Namespace()
    ns.watch = watch
    ns.timeout = timeout
    return ns


def _make_status(pid, listening, tunnel_port=9090):
    s = MagicMock()
    s.pid = pid
    s.listening = listening
    s.tunnel_port = tunnel_port
    return s


@contextmanager
def _env(status, listen=("127.0.0.1", 9090), capture_console=False):
    """Patch the iron-proxy hooks cmd_health depends on.

    ``listen`` is the ``(host, port)`` the daemon binds, as reported by
    ``_read_http_listen_from_config`` — the source of truth for the probe
    host (loopback on macOS/Windows, the docker bridge gateway on Linux).
    """
    patches = [
        patch("hermes_cli.proxy_cli.ip.get_status", return_value=status),
        patch(
            "hermes_cli.proxy_cli.ip._read_http_listen_from_config",
            return_value=listen,
        ),
    ]
    console = None
    if capture_console:
        console = MagicMock()
        patches.append(
            patch("hermes_cli.proxy_cli.Console", return_value=console)
        )
    started = [p.start() for p in patches]
    try:
        yield console
    finally:
        for p in patches:
            p.stop()
    del started


def _printed(console) -> str:
    return "\n".join(str(c.args[0]) for c in console.print.call_args_list if c.args)


class TestCmdHealth:
    def test_healthy_returns_0(self):
        with _env(_make_status(pid=1234, listening=True)):
            assert cmd_health(_args()) == 0

    def test_not_running_returns_2(self):
        with _env(_make_status(pid=None, listening=False)):
            assert cmd_health(_args()) == 2

    def test_pid_exists_not_listening_returns_1(self):
        with _env(_make_status(pid=1234, listening=False)):
            assert cmd_health(_args()) == 1

    def test_watch_returns_0_when_healthy(self):
        with _env(_make_status(pid=1234, listening=True)):
            assert cmd_health(_args(watch=True)) == 0

    def test_watch_timeout_returns_nonzero(self):
        with _env(_make_status(pid=None, listening=False)):
            with patch("time.sleep"):
                assert cmd_health(_args(watch=True, timeout=1)) != 0

    def test_timeout_without_watch_polls_until_healthy(self):
        """A positive --timeout without --watch must enter the polling loop,
        not return after a single probe.  An initially-down proxy that comes
        up before the deadline should be reported healthy (exit 0) — this is
        the documented `hermes egress health --timeout N` standalone contract.
        """
        down = _make_status(pid=None, listening=False)
        up = _make_status(pid=1234, listening=True)
        with _env(down):
            with patch(
                "hermes_cli.proxy_cli.ip.get_status",
                side_effect=[down, down, up],
            ) as gs, patch("time.sleep"):
                assert cmd_health(_args(watch=False, timeout=30)) == 0
            # Proves it polled repeatedly rather than probing once and
            # returning the initial down state.
            assert gs.call_count == 3

    def test_timeout_without_watch_returns_nonzero_at_deadline(self):
        """--timeout without --watch that never becomes healthy must poll
        until the deadline and then exit non-zero (not hang, not exit 0)."""
        down = _make_status(pid=None, listening=False)
        # Fake monotonic clock advancing 1s per call so the timeout=3
        # deadline is crossed after a few polls instead of blocking.
        ticks = iter([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        with _env(down):
            with patch(
                "hermes_cli.proxy_cli.ip.get_status", return_value=down
            ) as gs, patch("time.sleep"), patch(
                "time.monotonic", side_effect=lambda: next(ticks)
            ):
                assert cmd_health(_args(watch=False, timeout=3)) == 2
            # Polled more than once before giving up at the deadline.
            assert gs.call_count > 1

    def test_watch_keyboard_interrupt_returns_2(self):
        with _env(_make_status(pid=None, listening=False)):
            with patch("time.sleep", side_effect=KeyboardInterrupt):
                assert cmd_health(_args(watch=True)) == 2

    def test_reports_configured_bind_host_not_loopback(self):
        """On Linux the daemon binds the docker bridge; the health output
        must report that address, never a hardcoded 127.0.0.1 (which would
        be an unreachable probe target for a perfectly healthy daemon)."""
        status = _make_status(pid=1234, listening=True, tunnel_port=9070)
        with _env(status, listen=("172.17.0.1", 9070), capture_console=True) as console:
            assert cmd_health(_args()) == 0
        out = _printed(console)
        assert "172.17.0.1:9070" in out
        assert "127.0.0.1" not in out

    def test_not_listening_message_uses_configured_host(self):
        status = _make_status(pid=1234, listening=False, tunnel_port=9070)
        with _env(status, listen=("172.17.0.1", 9070), capture_console=True) as console:
            assert cmd_health(_args()) == 1
        assert "172.17.0.1:9070" in _printed(console)

    def test_falls_back_to_loopback_when_config_absent(self):
        """No proxy.yaml (helper returns None) → loopback is the correct
        display default, matching get_status()'s own fallback."""
        status = _make_status(pid=1234, listening=True, tunnel_port=9090)
        with _env(status, listen=None, capture_console=True) as console:
            assert cmd_health(_args()) == 0
        assert "127.0.0.1:9090" in _printed(console)
