"""Tests for the strict gateway command-line matcher.

Regression guard for the Windows ``hermes gateway restart`` silent-outage bug:
the previous loose substring match (``"... gateway" in cmdline``) false-matched
``gateway status``/``dashboard`` siblings and unrelated processes such as
``python -m tui_gateway``, which let ``restart()`` race a still-draining old
process and ``status``/``start`` report false positives.
"""

from __future__ import annotations

import pytest

from gateway.status import (
    _gateway_identity_from_argv as argv_identity,
    gateway_spawn_intent_subcommand as spawn_intent,
    looks_like_gateway_command_line as matches,
    looks_like_gateway_runtime_command_line as matches_runtime,
)


ACCEPT = [
    "pythonw.exe -m hermes_cli.main gateway run",
    r"C:\Users\me\hermes\venv\Scripts\pythonw.exe -m hermes_cli.main gateway run",
    "python -m hermes_cli.main --profile work gateway run",
    "python -m hermes_cli.main gateway run --replace",
    "python -m hermes_cli/main.py gateway run",
    "python gateway/run.py",
    "hermes-gateway.exe",
    "hermes gateway",          # bare `hermes gateway` defaults to run
    "hermes gateway run",
    # profile selector AFTER the `gateway` token (argv is profile-position
    # agnostic — _apply_profile_override strips --profile/-p anywhere)
    "hermes gateway --profile work run",
    "python -m hermes_cli.main gateway -p work run",
    "hermes gateway --profile=work run",
    # a profile literally NAMED "gateway"
    "hermes -p gateway gateway run",
    "python -m hermes_cli.main --profile gateway gateway run",
    # quoted Windows paths with spaces (shlex-aware tokenization)
    r'"C:\Program Files\Hermes\hermes-gateway.exe"',
    r'"C:\Program Files\Hermes\gateway\run.py" run',
    r'"C:\Program Files\Py\pythonw.exe" -m hermes_cli.main gateway run',
]

REJECT = [
    "python -m tui_gateway",                              # unrelated module
    "python -m hermes_cli.main gateway status",           # other subcommand
    "python -m hermes_cli.main gateway restart",
    "python -m hermes_cli.main gateway stop",
    "python -m hermes_cli.main --profile x dashboard",    # non-gateway subcommand
    "some random python -m mygateway thing",
    "",
    None,
]


@pytest.mark.parametrize("cmd", ACCEPT)
def test_accepts_real_gateway_run(cmd):
    assert matches(cmd) is True


@pytest.mark.parametrize("cmd", REJECT)
def test_rejects_non_gateway_run(cmd):
    assert matches(cmd) is False


# ``python -c <src> <old_pid> <gateway argv…>`` — the detached restart watcher
# (hermes_cli.gateway._spawn_gateway_restart_watcher). Its trailing argv is the command it will
# spawn LATER, so reading identity off it made the updater's post-relaunch liveness poll vouch for
# the watcher instead of a gateway (#107002).
INLINE_SOURCE_REJECT = [
    'python -c "import time; time.sleep(1)" 14980 python -m hermes_cli.main gateway run',
    r'"C:\Users\me\hermes\venv\Scripts\python.exe" -c "import os" 14980 '
    r'"C:\Users\me\hermes\venv\Scripts\python.exe" -m hermes_cli.main gateway run',
    'python -u -c "import os" 14980 python -m hermes_cli.main --profile work gateway run',
    'python -uc "import os" 14980 hermes gateway run',
    # Options that take a SEPARATE operand must not end the option walk before ``-c`` (the operand
    # is not the start of the program's own argv). The repo itself spawns ``-I -S -B -X utf8 …``
    # (hermes_cli/_old_updater.py, _update_takeover.py), so this shape is not hypothetical.
    'python -X utf8 -c "import os" 14980 python -m hermes_cli.main gateway run',
    'python -W ignore -c "import os" 14980 python -m hermes_cli.main gateway run',
    'python --check-hash-based-pycs always -c "import os" 14980 hermes gateway run',
    'python -I -S -B -X utf8 -c "import os" 14980 python -m hermes_cli.main gateway run',
    # ``-q`` (quiet) takes NO operand, unlike ``-Q``; a case-folded walk would skip past the ``-c``.
    'python -q -c "import os" 14980 python -m hermes_cli.main gateway run',
]


# Real gateways whose interpreter carries operand-taking options must STILL be recognised — the
# value-aware walk must not over-reject. Mirror image of INLINE_SOURCE_REJECT.
INTERPRETER_OPTION_ACCEPT = [
    "python -X utf8 -m hermes_cli.main gateway run",
    "python -W ignore -m hermes_cli.main gateway run",
    "python -q -m hermes_cli.main gateway run",
    "python -I -S -B -X utf8 -m hermes_cli.main gateway run",
    "python --check-hash-based-pycs always -m hermes_cli.main gateway run",
]


@pytest.mark.parametrize("cmd", INTERPRETER_OPTION_ACCEPT)
def test_accepts_gateway_behind_operand_taking_interpreter_options(cmd):
    assert matches(cmd) is True


# The repo's own non-gateway ``-X utf8`` spawn shapes must stay unmatched.
@pytest.mark.parametrize(
    "cmd",
    [
        "python -I -S -B -X utf8 /tmp/update_takeover.py",
        "python -X utf8 -E script.py",
    ],
)
def test_operand_taking_options_do_not_manufacture_a_gateway(cmd):
    assert matches(cmd) is False


@pytest.mark.parametrize("cmd", INLINE_SOURCE_REJECT)
def test_rejects_interpreter_running_inline_source(cmd):
    assert matches(cmd) is False
    assert matches_runtime(cmd) is False


# Spawn INTENT is the mirror image of process identity: the same wrapper that must not be read as a
# live gateway MUST still be recognised as "launching this eventually produces a gateway runtime".
# tests/_fixtures/live_system_guard.py relies on it — without this, the autouse guard stopped
# blocking the detached restart watcher and real gateways leaked out of the test run.
@pytest.mark.parametrize("cmd", INLINE_SOURCE_REJECT)
def test_spawn_intent_sees_through_the_inline_source_wrapper(cmd):
    assert spawn_intent(cmd) == "run"


@pytest.mark.parametrize("cmd", ACCEPT)
def test_spawn_intent_matches_plain_gateway_run(cmd):
    assert spawn_intent(cmd) == "run"


@pytest.mark.parametrize("cmd", REJECT)
def test_spawn_intent_rejects_non_gateway_commands(cmd):
    assert spawn_intent(cmd) != "run"


def test_spawn_intent_keeps_read_only_subcommands_spawnable():
    """The guard only blocks run/start/restart; a ``-c``-wrapped ``gateway status`` must stay
    launchable (tests/test_live_system_guard_self_test.py asserts it passes through)."""
    cmd = 'python -c "import sys; print(sys.argv[1:])" -m hermes_cli.main gateway status'
    assert spawn_intent(cmd) == "status"


def test_spawn_intent_ignores_inline_source_without_a_gateway_argv():
    assert spawn_intent('python -c "import time; time.sleep(1)" 14980') is None


# Atomic Hermes' bundled desktop runner (regression for #22418): it shares
# HERMES_HOME with the CLI and must be recognised as a gateway so
# ``gateway run --replace`` enters the replace/lock-handoff path instead of
# colliding with the desktop runner's still-held scoped locks.
ATOMIC_DESKTOP = (
    "/Applications/Atomic Hermes.app/Contents/Resources/python-server/python "
    "/Applications/Atomic Hermes.app/Contents/Resources/python-server/desktop-gateway.py"
)


def test_accepts_atomic_desktop_gateway():
    assert matches(ATOMIC_DESKTOP) is True
    assert matches_runtime(ATOMIC_DESKTOP) is True


# The launcher the live gateway actually runs under. Its operand IS the hermes CLI entrypoint, so
# identity must come from argv: ``_read_process_cmdline`` space-joins the operand into the trailing
# argv and the string matcher refuses every ``-c`` wrapper on purpose (#107002). Only argv can tell
# "this process IS hermes" apart from "this process will spawn one later".
LAUNCHER_OPERAND = (
    "import os, sys, runpy; sys.path.insert(0, '/repo'); import hermes_bootstrap; "
    "from hermes_cli.main import main\n"
    "sys.argv[0] = re.sub(r'(-script\\.pyw|\\.exe)?$', '', sys.argv[0])\n"
    "sys.exit(main())"
)
RUNPY_OPERAND = (
    "import os, sys, runpy; os.environ.pop('PYTHONPATH', None); import hermes_bootstrap; "
    "runpy.run_module('hermes_cli.main', run_name='__main__', alter_sys=True)"
)
# ``_spawn_gateway_restart_watcher``'s own operand: polls ``_pid_exists`` then Popen()s sys.argv[2:].
WATCHER_OPERAND = "import time, subprocess, sys; time.sleep(1); subprocess.Popen(sys.argv[2:])"


def test_launcher_inline_source_is_gateway_identity():
    """The installed launcher (``.hermes/bin/hermes``) runs the CLI from the operand itself."""
    assert argv_identity(["/usr/bin/python3", "-I", "-c", LAUNCHER_OPERAND, "gateway", "run"]) == "run"
    # bare ``hermes gateway`` defaults to run, as in the string matcher
    assert argv_identity(["/usr/bin/python3", "-I", "-c", LAUNCHER_OPERAND, "gateway"]) == "run"
    # ...and the string path still refuses it: this recognition is argv-only, not a loosened matcher
    assert matches('python -I -c "{}" gateway run'.format(LAUNCHER_OPERAND)) is False


def test_launcher_runpy_form_is_gateway_identity():
    """``hermes_cli._launchers.runtime_command()`` builds the same process with runpy."""
    argv = ["/usr/bin/python3", "-I", "-c", RUNPY_OPERAND, "gateway", "run"]
    assert argv_identity(argv) == "run"
    assert matches('python -I -c "{}" gateway run'.format(RUNPY_OPERAND)) is False


def test_restart_watcher_operand_is_not_identity():
    """The watcher's own operand is the poll-and-spawn script, so it is not the hermes entrypoint."""
    assert argv_identity(["/usr/bin/python3", "-c", WATCHER_OPERAND, "4242"]) is None
    assert argv_identity(["/usr/bin/python3", "-c", WATCHER_OPERAND]) is None


def test_nested_watcher_with_launcher_trailing_argv_is_not_identity():
    """A correct launcher bootstrap sitting in the watcher's TRAILING argv proves nothing about the
    watcher: that argv is the command the watcher will spawn LATER. Reading it made the updater's
    post-relaunch liveness poll vouch for the watcher instead of a gateway (#107002)."""
    watcher = [
        "/usr/bin/python3", "-c", WATCHER_OPERAND, "4242",
        "/usr/bin/python3", "-I", "-c", LAUNCHER_OPERAND, "gateway", "run",
    ]
    assert argv_identity(watcher) is None
    assert argv_identity([*watcher[:4], "/usr/bin/python3", "-I", "-c", RUNPY_OPERAND, "gateway",
                          "restart"]) is None


def test_plain_python_dash_c_is_not_identity():
    """An inline program that names no hermes entrypoint is not a gateway, whatever its argv."""
    assert argv_identity(["/usr/bin/python3", "-c", "import os; print(os.getpid())"]) is None
    assert argv_identity(["/usr/bin/python3", "-c", "import os", "gateway", "run"]) is None
    # no inline source at all -> argv has no verdict (the string path decides those)
    assert argv_identity(["/usr/bin/python3", "-m", "hermes_cli.main", "gateway", "run"]) is None
    assert argv_identity(None) is None


@pytest.mark.parametrize(
    "selector",
    [
        ["-p", "work"],
        ["--profile", "work"],
        ["--profile=work"],
    ],
)
def test_named_profile_launcher_is_gateway_identity(selector):
    """``hermes_cli.gateway._gateway_run_args_for_profile()`` appends the profile selector BEFORE
    ``gateway run``, so a named-profile launcher's trailing argv starts with ``-p`` / ``--profile``
    rather than ``gateway``. Those selectors are stripped anywhere in argv, exactly as the string
    matcher does."""
    argv = ["/usr/bin/python3", "-I", "-c", LAUNCHER_OPERAND, *selector, "gateway", "run"]
    assert argv_identity(argv) == "run"
    # the runpy launcher shape too, and the selector is value-consuming: a profile NAMED "gateway"
    runpy = ["/usr/bin/python3", "-I", "-c", RUNPY_OPERAND, *selector, "gateway", "run"]
    assert argv_identity(runpy) == "run"
    named_gateway = ["/usr/bin/python3", "-I", "-c", LAUNCHER_OPERAND, "-p", "gateway", "gateway",
                     "run"]
    assert argv_identity(named_gateway) == "run"


def test_launcher_argv_without_subcommand_is_run():
    """Bare ``hermes gateway`` defaults to run, profile selector or not -- and the profile token
    must not be mistaken for the subcommand, nor a sibling such as ``gateway status`` accepted."""
    assert argv_identity(["/usr/bin/python3", "-I", "-c", LAUNCHER_OPERAND, "gateway"]) == "run"
    assert argv_identity(["/usr/bin/python3", "-I", "-c", LAUNCHER_OPERAND, "-p", "work",
                          "gateway"]) == "run"
    assert argv_identity(["/usr/bin/python3", "-I", "-c", LAUNCHER_OPERAND, "--profile=work",
                          "gateway"]) == "run"
    assert argv_identity(["/usr/bin/python3", "-I", "-c", LAUNCHER_OPERAND, "-p", "work",
                          "gateway", "status"]) is None
    assert argv_identity(["/usr/bin/python3", "-I", "-c", LAUNCHER_OPERAND, "-p", "work"]) is None


