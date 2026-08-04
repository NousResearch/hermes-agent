"""Orphan Desktop-local ``hermes serve`` reap at backend start.

When Desktop dies uncleanly, local ``serve --host 127.0.0.1 --port 0``
children can be reparented to pid 1 and keep full MCP trees alive. The next
boot must clear those corpses without touching intentional fixed-port serves
(e.g. ``--port 9119`` remote dashboards).
"""

from __future__ import annotations

import os
from unittest.mock import patch

from hermes_cli.dashboard_procs import (
    _is_desktop_local_serve_cmdline,
    _reap_orphaned_desktop_local_serves,
)


def test_desktop_local_serve_shape_matches_ephemeral_loopback():
    assert _is_desktop_local_serve_cmdline(
        "python -m hermes_cli.main serve --host 127.0.0.1 --port 0"
    )
    assert _is_desktop_local_serve_cmdline(
        "hermes serve --isolated --host 127.0.0.1 --port 0 --ssh-owner-nonce abc"
    )
    assert _is_desktop_local_serve_cmdline(
        "/venv/bin/hermes serve --host=127.0.0.1 --port=0"
    )


def test_desktop_local_serve_shape_spares_fixed_port_and_non_serve():
    assert not _is_desktop_local_serve_cmdline(
        "hermes serve --host 100.106.105.2 --port 9119 --skip-build"
    )
    assert not _is_desktop_local_serve_cmdline(
        "hermes serve --host 127.0.0.1 --port 9119"
    )
    assert not _is_desktop_local_serve_cmdline("hermes gateway run --replace")
    assert not _is_desktop_local_serve_cmdline(
        "vim notes about hermes serve --port 0"
    )


def test_reap_only_kills_ppid1_local_serves():
    scanned = [
        (111, "hermes serve --host 127.0.0.1 --port 0"),  # orphan local
        (222, "hermes serve --host 127.0.0.1 --port 0"),  # still has parent
        (333, "hermes serve --host 100.1.2.3 --port 9119"),  # fixed remote
        (444, "hermes serve --isolated --host 127.0.0.1 --port 0"),  # orphan isolated
    ]
    ppids = {111: 1, 222: 50, 333: 1, 444: 1}
    terms: list[int] = []
    live = {111, 222, 333, 444}

    def fake_kill(pid, sig):
        if sig == 0:
            if pid in live:
                return None
            raise ProcessLookupError()
        if sig == 15:
            terms.append(pid)
            live.discard(pid)
            return None
        if sig == 9:
            live.discard(pid)
            return None
        return None

    with (
        patch(
            "hermes_cli.dashboard_procs._scan_dashboard_processes",
            return_value=scanned,
        ),
        patch(
            "hermes_cli.dashboard_procs._process_ppid",
            side_effect=lambda pid: ppids.get(pid),
        ),
        patch("os.kill", side_effect=fake_kill),
        patch("sys.platform", "darwin"),
    ):
        os.environ.pop("HERMES_DESKTOP_CHILD_PID", None)
        result = _reap_orphaned_desktop_local_serves(
            sleep_fn=lambda _s: None,
            signal_term=15,
            signal_kill=9,
        )

    assert set(result["matched"]) == {111, 444}
    assert set(terms) == {111, 444}
    assert set(result["killed"]) == {111, 444}
    assert 222 not in terms
    assert 333 not in terms


def test_reap_passes_child_pid_exclude_to_scan():
    with (
        patch(
            "hermes_cli.dashboard_procs._scan_dashboard_processes",
            return_value=[],
        ) as scan,
        patch("sys.platform", "darwin"),
        patch.dict(os.environ, {"HERMES_DESKTOP_CHILD_PID": "999,111"}, clear=False),
    ):
        result = _reap_orphaned_desktop_local_serves(sleep_fn=lambda _s: None)

    assert result["matched"] == []
    exclude = scan.call_args.kwargs["exclude_pids"]
    assert 111 in exclude
    assert 999 in exclude
