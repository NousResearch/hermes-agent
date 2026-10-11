"""Dashboard identity where argv can't say "dashboard": ledger backends and shell-wrapped plists.

``--status`` re-parsed the spawn ledger's recorded argv and dropped what it could not classify. That
argv is ``sys.argv``, which for a backend started through the PM launcher (``python -I -c <bootstrap>
dashboard …``, what ``~/.local/bin/hermes`` and LaunchAgents run) is ``-c dashboard …``: a LaunchAgent
dashboard serving on 9119 was reported as "No hermes dashboard or serve processes running", while
``--stop`` (which trusts the ledger) still targeted it. Related: #126887 (the macOS ``ps`` half).

The launchd inventory and the uninstall sweep classify plist ``ProgramArguments`` with the same parser;
since the canonical matcher keeps every ``-c`` operand as data, a ``/bin/sh -c 'exec hermes dashboard'``
plist stopped being found.
"""

from __future__ import annotations

import argparse
import shlex
import socket

import pytest

from hermes_cli import dashboard_procs, process_identity
from hermes_cli.main import cmd_dashboard
from hermes_cli.main_dashboard import _find_stale_dashboard_pids, _parse_dashboard_runtime


def test_status_lists_a_ledger_backend_whose_recorded_argv_has_no_entry_token(capsys, monkeypatch):
    pid = 434343
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        monkeypatch.setattr(dashboard_procs, "_iter_process_table", lambda: [])
        monkeypatch.setattr(process_identity, "ledger_entries", lambda *a, **k: [
            {"pid": pid, "purpose": "dashboard", "argv": "-c dashboard --no-open --skip-build",
             "host": "127.0.0.1", "port": port}])
        monkeypatch.setattr("gateway.status._pid_exists", lambda p: p == pid)

        assert _find_stale_dashboard_pids() == [pid]       # --stop sees it ...
        with pytest.raises(SystemExit):
            cmd_dashboard(argparse.Namespace(port=9119, host="127.0.0.1", no_open=False, insecure=False,
                                             stop=False, status=True))
    assert f"PID {pid} [dashboard]" in capsys.readouterr().out   # ... so --status must list it


def test_shell_wrapped_plist_arguments_classify_by_the_command_the_shell_runs():
    parse = lambda argv: _parse_dashboard_runtime(shlex.join(argv))
    assert parse(["/bin/sh", "-c", "exec /opt/hermes/bin/hermes dashboard --port 9200"]) == (
        "dashboard", "127.0.0.1", 9200)
    assert parse(["/bin/bash", "-c", "exec /opt/hermes/bin/hermes --profile work serve"])[0] == "serve"
    assert parse(["/bin/sh", "-c", "exec mytool dashboard"]) is None
    # python -c keeps its operand as data (#107002): a program CARRYING the words is not a backend.
    assert parse(["python3", "-c", "import sys", "hermes", "dashboard"]) is None
