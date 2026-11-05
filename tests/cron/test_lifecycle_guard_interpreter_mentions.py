"""Paths inside a non-shell interpreter script are mentions, not executions.

The referenced-script walk reads an executed script's text and tokenizes it as shell. For a
``#!/usr/bin/env python3`` script that text is Python: ``os.path.expanduser("~/.hermes/state.db")``
splits at the parenthesis and leaves the quoted path in command position, so the walk took a
data file the script merely names for a script it executes, and failed the whole command closed
when that file could not be read (a live SQLite database, an oversized file). The walk's own
contract is that a mere data mention must never trip a fail-closed verdict.

Mentions are still read and scanned, so an interpreter script that hands a restart command or a
restart script to ``os.system``/``subprocess`` is still blocked.
"""

from __future__ import annotations

import os

from cron.lifecycle_guard import _MAX_REFERENCED_SCRIPT_BYTES, scan_gateway_lifecycle
from hermes_cli.sqlite_safe_read import connect_tracked


def _script(path, body, shebang="#!/usr/bin/env python3"):
    path.write_text((shebang + "\n" if shebang else "") + body, encoding="utf-8")
    os.chmod(path, 0o755)
    return path


def _oversized(path):
    path.write_bytes(b"x" * (_MAX_REFERENCED_SCRIPT_BYTES + 1))
    return path


def test_python_script_naming_a_live_sqlite_db_is_allowed(tmp_path):
    db = tmp_path / "state.db"
    conn = connect_tracked(db)
    try:
        script = _script(tmp_path / "report.py", f'import os\nDB = os.path.expanduser("{db}")\n')
        assert scan_gateway_lifecycle(f"{script} --send") == (False, None)
    finally:
        conn.close()


def test_python_script_naming_an_oversized_file_is_allowed(tmp_path):
    data = _oversized(tmp_path / "history.jsonl")
    script = _script(tmp_path / "report.py", f'open("{data}").read()\n')
    assert scan_gateway_lifecycle(str(script)) == (False, None)


def test_env_dash_s_python_shebang_counts_as_an_interpreter(tmp_path):
    data = _oversized(tmp_path / "history.jsonl")
    script = _script(tmp_path / "report.py", f'open("{data}").read()\n',
                     shebang="#!/usr/bin/env -S python3 -u")
    assert scan_gateway_lifecycle(str(script)) == (False, None)


def test_python_script_that_restarts_the_gateway_is_still_blocked(tmp_path):
    script = _script(tmp_path / "bounce.py",
                     'import os\nos.system("systemctl --user restart hermes-gateway")\n')
    assert scan_gateway_lifecycle(str(script))[0] is True


def test_python_script_handing_off_to_a_restart_script_is_still_blocked(tmp_path):
    restart = _script(tmp_path / "restart.sh", "hermes gateway restart\n", shebang="#!/bin/sh")
    script = _script(tmp_path / "bounce.py",
                     f'import os\nos.system("{restart}")\n')
    assert scan_gateway_lifecycle(str(script))[0] is True


def test_shell_script_executing_an_oversized_file_still_fails_closed(tmp_path):
    big = _oversized(tmp_path / "big.sh")
    script = _script(tmp_path / "run.sh", f"{big}\n", shebang="#!/bin/bash")
    unsafe, refusal = scan_gateway_lifecycle(str(script))
    assert unsafe is True and refusal


def test_script_without_a_shebang_is_still_read_as_shell(tmp_path):
    # execve() hands a shebang-less script to /bin/sh (ENOEXEC), so it stays shell.
    big = _oversized(tmp_path / "big.sh")
    script = _script(tmp_path / "run", f"{big}\n", shebang=None)
    unsafe, refusal = scan_gateway_lifecycle(str(script))
    assert unsafe is True and refusal
