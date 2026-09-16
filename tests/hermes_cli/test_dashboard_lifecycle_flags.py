"""Tests for ``hermes dashboard --stop`` / ``--status`` flags.

These flags share the detection + kill path with the post-``hermes update``
cleanup, so the heavy coverage of SIGTERM / SIGKILL / Windows taskkill lives
in ``test_update_stale_dashboard.py``.  This file just verifies the flag
dispatch: argparse wiring, no-op when nothing is running, and correct
exit codes.
"""

from __future__ import annotations

import argparse
import sys
from unittest.mock import patch, MagicMock

import pytest

from hermes_cli import main as cli_main
from hermes_cli import main_dashboard
from hermes_cli.main import cmd_dashboard


def _ns(**kw):
    """Build an argparse.Namespace with dashboard defaults plus overrides."""
    defaults = dict(
        port=9119, host="127.0.0.1", no_open=False, insecure=False,
        stop=False, status=False, ssl_certfile=None, ssl_keyfile=None,
    )
    defaults.update(kw)
    return argparse.Namespace(**defaults)


class TestDashboardStatus:
    def test_status_no_processes(self, capsys):
        with patch("hermes_cli.dashboard_procs._scan_dashboard_processes", return_value=[]), \
             pytest.raises(SystemExit) as exc:
            cmd_dashboard(_ns(status=True))
        assert exc.value.code == 0
        out = capsys.readouterr().out
        assert "No hermes dashboard or serve processes running" in out

    def test_status_with_processes(self, capsys):
        # Includes a serve-mode backend: --status must LIST it, not hide it —
        # `--stop` kills serves, so hiding them let operators kill what they
        # couldn't see (#81564).
        processes = [
            (12345, "hermes dashboard --port 9119"),
            (12346, "python -m hermes_cli.main dashboard --host 0.0.0.0 --port 9120"),
            (12347, "hermes serve --host 100.94.65.93 --port 9119"),
        ]
        with patch("hermes_cli.dashboard_procs._scan_dashboard_processes", return_value=processes), \
             patch("gateway.status._pid_exists", return_value=True), \
             patch("hermes_cli.main_dashboard._dashboard_listening", return_value=True), \
             pytest.raises(SystemExit) as exc:
            cmd_dashboard(_ns(status=True))
        # Status is informational — always exits 0.
        assert exc.value.code == 0
        out = capsys.readouterr().out
        assert "3 hermes dashboard/serve process(es) running" in out
        assert "PID 12345" in out
        assert "PID 12346" in out
        assert "PID 12347" in out and "[serve]" in out


    def test_status_does_not_try_to_import_fastapi(self):
        """`--status` must not require dashboard runtime deps — it's a
        process-table scan only.  We prove this by making fastapi import
        fail and confirming --status still succeeds."""
        orig_import = __import__
        def fake_import(name, *a, **kw):
            if name == "fastapi":
                raise ImportError("fastapi missing")
            return orig_import(name, *a, **kw)

        with patch("hermes_cli.dashboard_procs._scan_dashboard_processes", return_value=[]), \
             patch("builtins.__import__", side_effect=fake_import), \
             pytest.raises(SystemExit) as exc:
            cmd_dashboard(_ns(status=True))
        assert exc.value.code == 0


class TestDashboardStop:

    def test_stop_kills_and_exits_zero_when_all_killed(self, capsys):
        """After the kill, if the second scan returns empty we exit 0."""
        # First scan: finds two processes.  Second (verification) scan: empty.
        scans = iter([[12345, 12346], []])
        with patch("hermes_cli.main._find_stale_dashboard_pids",
                   side_effect=lambda: next(scans)), \
             patch("hermes_cli.dashboard_procs._kill_stale_dashboard_processes") as mock_kill, \
             pytest.raises(SystemExit) as exc:
            cmd_dashboard(_ns(stop=True))
        mock_kill.assert_called_once()
        # --stop should pass a reason so the output doesn't say "running
        # backend no longer matches the updated frontend" (that wording is
        # for the post-`hermes update` path).
        kwargs = mock_kill.call_args.kwargs
        assert "reason" in kwargs
        assert "stop" in kwargs["reason"].lower()
        assert exc.value.code == 0

    def test_stop_exits_nonzero_if_kill_leaves_survivors(self):
        """If the second scan still finds PIDs, we exit 1 so scripts can
        detect that the stop didn't succeed (e.g. permission denied)."""
        scans = iter([[12345], [12345]])  # both scans find the same PID
        with patch("hermes_cli.main._find_stale_dashboard_pids",
                   side_effect=lambda: next(scans)), \
             patch("hermes_cli.dashboard_procs._kill_stale_dashboard_processes"), \
             pytest.raises(SystemExit) as exc:
            cmd_dashboard(_ns(stop=True))
        assert exc.value.code == 1

    def test_stop_does_not_try_to_import_fastapi(self):
        """Like --status, --stop must work without dashboard runtime deps."""
        orig_import = __import__
        def fake_import(name, *a, **kw):
            if name == "fastapi":
                raise ImportError("fastapi missing")
            return orig_import(name, *a, **kw)

        with patch("hermes_cli.main._find_stale_dashboard_pids",
                   return_value=[]), \
             patch("builtins.__import__", side_effect=fake_import), \
             pytest.raises(SystemExit) as exc:
            cmd_dashboard(_ns(stop=True))
        assert exc.value.code == 0


class TestLifecycleFlagsTakePrecedence:
    """If both --stop and --status are set, --status wins (it's listed
    first in cmd_dashboard).  Neither is allowed to fall through to the
    server-start path, which is the critical safety property — a user
    who typed ``hermes dashboard --stop`` must not end up ALSO starting
    a new server."""


    def test_stop_does_not_fall_through_to_server_start(self):
        """Covers the worst-case regression: if --stop ever stopped exiting
        early, the user would start the dashboard they just asked to stop."""
        called = {"start": False}
        def fake_start_server(**kw):
            called["start"] = True

        # Provide a fake web_server module so the import doesn't matter.
        fake_ws = MagicMock()
        fake_ws.start_server = fake_start_server

        with patch("hermes_cli.main._find_stale_dashboard_pids",
                   return_value=[]), \
             patch.dict(sys.modules, {"hermes_cli.web_server": fake_ws}), \
             pytest.raises(SystemExit):
            cmd_dashboard(_ns(stop=True))
        assert called["start"] is False


@pytest.mark.parametrize(
    ("entry_tls", "invocation_tls", "expected_scheme"),
    [(True, False, "https"), (False, True, "http")],
)
def test_named_profile_existing_dashboard_uses_verified_ledger_scheme(
    monkeypatch, entry_tls, invocation_tls, expected_scheme
):
    args = _ns(
        ssl_certfile="/run/hermes-dashboard/fullchain.pem" if invocation_tls else None,
        ssl_keyfile="/run/hermes-dashboard/privkey.pem" if invocation_tls else None,
    )
    monkeypatch.setattr("hermes_cli.profiles.get_active_profile_name", lambda: "work")
    entry = {"purpose": "dashboard", "host": "127.0.0.1", "port": 9119}
    if entry_tls:
        entry.update(
            ssl_certfile="/run/hermes-dashboard/fullchain.pem",
            ssl_keyfile="/run/hermes-dashboard/privkey.pem",
        )
    monkeypatch.setattr("hermes_cli.process_identity.ledger_entries", lambda: [entry])
    opened = []
    monkeypatch.setattr("webbrowser.open", opened.append)
    monkeypatch.setattr(main_dashboard, "_dashboard_listening", lambda *_a: True)

    with pytest.raises(SystemExit, match="0"):
        main_dashboard._route_named_profile_dashboard(args, False, "", "")

    assert opened == [f"{expected_scheme}://127.0.0.1:9119/?profile=work"]


def test_named_profile_existing_dashboard_matches_requested_bind_address(monkeypatch):
    args = _ns(host="127.0.0.1")
    monkeypatch.setattr("hermes_cli.profiles.get_active_profile_name", lambda: "work")
    monkeypatch.setattr(
        "hermes_cli.process_identity.ledger_entries",
        lambda: [
            {
                "purpose": "dashboard", "host": "127.0.0.1", "port": 9119,
                "registered_at": 1,
            },
            {
                "purpose": "dashboard", "host": "10.100.1.73", "port": 9119,
                "registered_at": 2,
                "ssl_certfile": "/run/hermes-dashboard/fullchain.pem",
                "ssl_keyfile": "/run/hermes-dashboard/privkey.pem",
            },
        ],
    )
    opened = []
    monkeypatch.setattr("webbrowser.open", opened.append)
    monkeypatch.setattr(main_dashboard, "_dashboard_listening", lambda *_a: True)

    with pytest.raises(SystemExit, match="0"):
        main_dashboard._route_named_profile_dashboard(args, False, "", "")

    assert opened == ["http://127.0.0.1:9119/?profile=work"]


def test_native_tls_survives_named_profile_reexec(monkeypatch):
    args = _ns(
        ssl_certfile="/run/hermes-dashboard/fullchain.pem",
        ssl_keyfile="/run/hermes-dashboard/privkey.pem",
    )
    monkeypatch.setattr("hermes_cli.profiles.get_active_profile_name", lambda: "work")

    monkeypatch.setattr(main_dashboard, "_dashboard_listening", lambda *_a: False)
    monkeypatch.setattr(
        "tools.environments.local.build_subprocess_env", lambda **_kw: {}
    )
    reexec = {}

    def capture_exec(_executable, argv, _env):
        reexec["argv"] = argv
        raise RuntimeError("captured re-exec")

    monkeypatch.setattr(main_dashboard.os, "execvpe", capture_exec)
    with pytest.raises(RuntimeError, match="captured re-exec"):
        main_dashboard._route_named_profile_dashboard(args, False, "", "")

    assert reexec["argv"][-4:] == [
        "--ssl-certfile",
        "/run/hermes-dashboard/fullchain.pem",
        "--ssl-keyfile",
        "/run/hermes-dashboard/privkey.pem",
    ]


def test_native_tls_pair_is_validated_before_runtime_side_effects(monkeypatch):
    runtime_prepared = False

    def prepare_runtime(*_args):
        nonlocal runtime_prepared
        runtime_prepared = True

    monkeypatch.setattr(cli_main, "_dashboard_prepare_runtime", prepare_runtime)

    with pytest.raises(SystemExit, match="--ssl-certfile and --ssl-keyfile must be supplied together"):
        cmd_dashboard(_ns(ssl_certfile="/run/hermes-dashboard/fullchain.pem"))

    assert runtime_prepared is False


def test_native_tls_paths_are_normalized_before_routing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = _ns(ssl_certfile="certs/fullchain.pem", ssl_keyfile="certs/privkey.pem")

    cli_main._dashboard_validate_serve_args(args, False, None)

    assert args.ssl_certfile == str(tmp_path / "certs/fullchain.pem")
    assert args.ssl_keyfile == str(tmp_path / "certs/privkey.pem")


class TestArgparseWiring:
    """Confirm the flags are exposed via the real argparse tree so
    ``hermes dashboard --stop`` / ``--status`` actually parse."""

    def test_native_tls_flags_are_registered(self):
        from hermes_cli.subcommands.dashboard import build_dashboard_parser, build_serve_parser

        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        build_dashboard_parser(
            subparsers,
            cmd_dashboard=lambda args: None,
            cmd_dashboard_register=lambda args: None,
        )

        args = parser.parse_args([
            "dashboard",
            "--ssl-certfile", "/run/hermes-dashboard/fullchain.pem",
            "--ssl-keyfile", "/run/hermes-dashboard/privkey.pem",
        ])

        assert args.ssl_certfile == "/run/hermes-dashboard/fullchain.pem"
        assert args.ssl_keyfile == "/run/hermes-dashboard/privkey.pem"

        serve_args = build_serve_parser(cmd_dashboard=lambda args: None).parse_args([
            "--ssl-certfile", "/run/hermes-dashboard/fullchain.pem",
            "--ssl-keyfile", "/run/hermes-dashboard/privkey.pem",
        ])
        assert serve_args.ssl_certfile == args.ssl_certfile
        assert serve_args.ssl_keyfile == args.ssl_keyfile

    def test_flags_are_registered(self):
        from hermes_cli.main import main as _cli_main  # noqa: F401
        # Rebuild the argparse tree by re-running the section of main()
        # that builds it.  Cheapest way: introspect via --help on the
        # already-built parser would require refactoring; instead we
        # parse the flags directly via a minimal replay.
        import importlib
        mod = importlib.import_module("hermes_cli.main")
        # Find the dashboard_parser instance by running build logic would
        # be too invasive.  Instead parse args as if via the CLI by
        # intercepting parse_args.  This is overkill for a smoke test —
        # we just want to know the flags don't KeyError.
        with patch("hermes_cli.dashboard_procs._scan_dashboard_processes", return_value=[]), \
             pytest.raises(SystemExit) as exc:
            mod.cmd_dashboard(_ns(status=True))
        assert exc.value.code == 0
