"""A stopped backend's captured argv must be the kernel's argv, token for token.

Regression for #136159: on macOS the capture went through ``ps -o command=`` (argv joined by
spaces, no quoting) and ``shlex.split``, so a backend launched through the managed interpreter
with ``-I -c "<python code>"`` was shredded into one token per word and the post-update respawn
died with ``SyntaxError: Expected one or more names after 'import'`` on every update.
"""

import subprocess
import sys
import time

import pytest

from hermes_cli import main_dashboard

# The shape ``hermes_cli.venv_sync.relaunch_command`` builds: spaces, commas, quotes and brackets.
_CODE = ("import sys, runpy; sys.path.insert(0, '/opt/hermes'); "
         "sys.argv = ['/opt/hermes/venv/bin/hermes', 'serve', '--host', '127.0.0.1', '--port', '9119']; "
         "import time; time.sleep(60)")


@pytest.fixture
def interpreter_child():
    proc = subprocess.Popen([sys.executable, "-I", "-c", _CODE], stdin=subprocess.DEVNULL,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(0.3)
    try:
        yield proc
    finally:
        proc.kill()
        proc.wait(timeout=10)


@pytest.mark.platforms("posix")
def test_captured_argv_round_trips_an_interpreter_code_argument(interpreter_child):
    """The captured argv equals what was spawned, so replaying it re-enters the same code."""
    argv = main_dashboard._dashboard_cmdline_for_pid(interpreter_child.pid)
    assert argv is not None
    assert argv[-3:] == ["-I", "-c", _CODE], argv


@pytest.mark.platforms("posix")
def test_captured_argv_survives_the_respawn_rebuild(interpreter_child):
    """An interpreter-relaunch shape is not a ``[python, hermes, ...]`` launcher argv, so the
    respawn rebuild replays it unchanged — the ``-c`` code must still be one token there."""
    argv = main_dashboard._dashboard_cmdline_for_pid(interpreter_child.pid)
    rebuilt = main_dashboard._respawnable_command_for_current_install(argv)
    assert rebuilt.count("-c") == 1
    assert rebuilt[rebuilt.index("-c") + 1] == _CODE
