"""The Windows launcher gateway form must be recognised as a gateway.

Windows spawns gateways two ways:

- ``hermes gateway start`` -> ``<python> -m hermes_cli.main gateway run``
- launcher scripts / the updater's relaunch (``_launchers.runtime_command``) ->
  ``<python> -I -c "<bootstrap; runpy.run_module('hermes_cli.main', alter_sys=True)>" gateway run --replace``

``_gateway_command_subcommand`` used to answer ``None`` for *every* ``-c`` form — a guard
(#107002) written for the restart watcher, whose ``-c`` really does carry a command it will
spawn later. That made the shim-launched gateway invisible to ``find_gateway_pids``,
``_wait_for_gateway_ready`` and the update preflight, so a live gateway read as "no gateway
process detected" and the updater refused to continue.

The fix recognises only *our own* launcher bootstrap (the ``alter_sys=True)`` tail marker)
and keeps rejecting foreign inline source, so #107002 stays intact.
"""

import sys

import pytest

from gateway.status import _gateway_command_subcommand

# Verbatim shape of the injected launcher bootstrap (hermes_cli/_launchers.py::runtime_command).
LAUNCHER_BOOTSTRAP = (
    "import os, sys, runpy; "
    "os.environ.pop('PYTHONHOME', None); "
    "sys.path.insert(0, 'C:\\\\hermes\\\\hermes-agent'); "
    "import hermes_bootstrap; "
    "runpy.run_module('hermes_cli.main', run_name='__main__', alter_sys=True)"
)

pytestmark = pytest.mark.skipif(
    sys.platform != "win32", reason="the launcher form only exists on Windows"
)


def _launcher_cmdline(*args: str) -> str:
    """The cmdline Windows reports for a launcher-spawned gateway.

    ``_read_process_cmdline`` returns an unquoted, space-joined argv, so the bootstrap source
    arrives split across many tokens — which is exactly why the matcher scans for the entry
    marker instead of trusting a single token.
    """
    return " ".join([r"C:\tools\python.exe", "-I", "-c", LAUNCHER_BOOTSTRAP, *args])


def _console_script_cmdline(*args: str) -> str:
    return " ".join([r"C:\tools\python.exe", "-m", "hermes_cli.main", *args])


@pytest.mark.skipif(sys.platform != "win32", reason="windows launcher form")
@pytest.mark.parametrize("subcommand", ["run", "status", "restart"])
def test_launcher_form_is_recognised(subcommand):
    argv = ["gateway", subcommand]
    if subcommand == "run":
        argv.append("--replace")
    assert _gateway_command_subcommand(_launcher_cmdline(*argv)) == subcommand


@pytest.mark.skipif(sys.platform != "win32", reason="windows launcher form")
def test_launcher_form_served_by_legacy_module_form():
    """The plain console-script form keeps working (regression guard)."""
    cmdline = r"C:\tools\python.exe -m hermes_cli.main gateway run"
    assert _gateway_command_subcommand(cmdline) == "run"


def test_watcher_inline_source_still_rejected():
    """#107002: the restart watcher's own ``-c`` must never look like a gateway."""
    watcher = 'python -c "import os, sys, time; pass" 40688 gateway run --replace'
    assert _gateway_command_subcommand(watcher) is None


def test_foreign_inline_source_still_rejected():
    """Any other third-party ``-c`` payload stays unrecognised."""
    assert _gateway_command_subcommand('python -c "print(1)" gateway run') is None
