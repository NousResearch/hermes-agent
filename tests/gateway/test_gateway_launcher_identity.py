"""Identity of a gateway started through Hermes' own CLI launcher (``bin/hermes.cmd``).

The Desktop runs every ``hermes …`` through that launcher, so a gateway it starts carries an
INLINE-SOURCE command line — ``python -I -c "<bootstrap source>" gateway run --replace`` — which the
identity matchers refuse outright (#107002). That refusal is right for FOREIGN inline programs, whose
trailing argv is data for something they will spawn later, but the launcher's source imports
``hermes_bootstrap`` and hands ``sys.argv`` to ``hermes_cli.main``: its trailing tokens ARE this
process's own identity.

Refusing it made every identity probe blind to a running gateway — ``gateway.pid`` adoption, the
wmic/CIM process-table scan behind ``find_gateway_pids``, and ``hermes update``'s post-relaunch
liveness poll. The update then failed its own verification gate ("no stable gateway process appeared
after relaunch") and reported a failed update (exit 1) on a healthy install, while the recovery
re-entry sent ``--replace`` to the gateway the first attempt had just brought up.

Invariants: the launcher form is unwrapped and recognized in BOTH command-line spellings real probes
produce; the restart watcher and unrelated inline programs stay refused.
"""
from __future__ import annotations

import shlex

from gateway.status import (
    hermes_cli_launcher_argv,
    inline_source_flag_index,
    looks_like_gateway_command_line,
    looks_like_gateway_runtime_command_line,
)

# The bootstrap source ``hermes_cli._launchers`` renders.
_LAUNCHER_SOURCE = (
    "import os, sys, runpy; "
    "os.environ.pop('PYTHONHOME', None); "
    "sys.path.insert(0, 'C:\\\\hermes\\\\hermes-agent'); "
    "import hermes_bootstrap; "
    "runpy.run_module('hermes_cli.main', run_name='__main__', alter_sys=True)"
)
_ENTRYPOINT = r"C:\hermes\tools\python-3.14.7\python.exe"
# wmic/CIM listings keep the shell's quoting: the source stays ONE argv element.
_QUOTED = f'{_ENTRYPOINT} -I -c "{_LAUNCHER_SOURCE}" gateway run --replace'
# psutil's ``cmdline()`` re-joins argv WITHOUT quoting: the source arrives as many tokens.
_UNQUOTED = f"{_ENTRYPOINT} -I -c {_LAUNCHER_SOURCE} gateway run --replace"
# ``gateway._spawn_gateway_restart_watcher``: its own inline source is the watcher, and the gateway
# argv it will spawn later rides along as plain data — never this process's identity.
_WATCHER = (
    f'{_ENTRYPOINT} -c "import sys; from gateway.status import _pid_exists" '
    f"4242 {_ENTRYPOINT} -I -c \"{_LAUNCHER_SOURCE}\" gateway run --replace"
)
_FOREIGN_INLINE = f'{_ENTRYPOINT} -c "print(1)" gateway run'


def _tokens(command: str) -> list[str]:
    return [t.strip("\"'").replace("\\", "/") for t in shlex.split(command, posix=False)]


def test_launcher_started_gateway_is_recognized_in_both_probe_spellings():
    assert inline_source_flag_index(_tokens(_QUOTED)) is not None
    for command in (_QUOTED, _UNQUOTED):
        assert hermes_cli_launcher_argv(_tokens(command)) == [
            "hermes_cli.main", "gateway", "run", "--replace",
        ]
        assert looks_like_gateway_command_line(command) is True
        assert looks_like_gateway_runtime_command_line(command) is True


def test_restart_watcher_and_foreign_inline_sources_stay_refused():
    assert hermes_cli_launcher_argv(_tokens(_WATCHER)) is None
    for command in (_WATCHER, _FOREIGN_INLINE):
        assert looks_like_gateway_command_line(command) is False
        assert looks_like_gateway_runtime_command_line(command) is False
