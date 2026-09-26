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


# ``hermes_cli/venv_sync.py::relaunch_command`` re-execs a Hermes entry point as
# ``python -I -c "import sys, runpy; …; sys.argv = ['…/hermes_cli/main.py', 'gateway', 'run'];
# …"`` (hermes_bootstrap's source-update completion path). That in-source assignment re-binds the
# process to that argv, so the process IS the gateway it names — unlike the #107002 watcher above,
# whose trailing argv is data for a child it spawns LATER. Without recognition, a PM-install
# gateway is invisible to every liveness surface: ``hermes gateway status``, the dashboard's
# ``/api/status`` and the update fleet verification all report it stopped while it runs, and the
# stale-PID cleanup then unlinks its gateway.pid/gateway.lock.
RELAUNCH_ACCEPT = [
    # real /proc shape: the -c body is an unquoted argv element, so shlex(posix=False)
    # fragments the embedded list — recognition must work on the command string, not tokens
    "python -I -c import sys, runpy; sys.path.insert(0, '/opt/hermes-agent'); "
    "sys.argv = ['/opt/hermes-agent/hermes_cli/main.py', 'gateway', 'run']; "
    "runpy.run_module('hermes_cli.main', run_name='__main__', alter_sys=True)",
    # quoted-body spelling (ps / log output)
    'python -I -c "import sys, runpy; sys.path.insert(0, \'/opt/hermes-agent\'); '
    "sys.argv = ['/opt/hermes-agent/hermes_cli/main.py', 'gateway', 'run']; "
    "runpy.run_module('hermes_cli.main', run_name='__main__', alter_sys=True)\"",
    # run_path variant (distlib .exe launchers) carrying a profile selector
    "python -I -c import sys, runpy; sys.argv = ['/opt/hermes-agent/hermes_cli/main.py', "
    "'--profile', 'work', 'gateway', 'run']; "
    "runpy.run_path('/opt/hermes-agent/hermes_cli/main.py', run_name='__main__')",
    # Windows: relaunch_command's argv!r double-escapes the backslash separators
    'python.exe -I -c "import sys, runpy; sys.argv = [\'C:\\\\hermes-agent\\\\hermes_cli\\\\main.py\', \'gateway\', \'run\']; '
    "runpy.run_module('hermes_cli.main', run_name='__main__', alter_sys=True)\"",
]


@pytest.mark.parametrize("cmd", RELAUNCH_ACCEPT)
def test_accepts_venv_sync_relaunched_gateway(cmd):
    assert matches(cmd) is True
    assert matches_runtime(cmd) is True
    # Identity and spawn-intent agree on the self-rebinding shape: the direct rung now
    # answers, so gateway_spawn_intent_subcommand no longer mis-reads it either.
    assert spawn_intent(cmd) == "run"


def test_inline_source_assigning_a_non_gateway_argv_stays_anonymous():
    cmd = (
        "python -I -c import sys; sys.argv = ['/opt/hermes-agent/hermes_cli/main.py', "
        "'chat', '-q']; runpy.run_module('hermes_cli.main')"
    )
    assert matches(cmd) is False
    assert matches_runtime(cmd) is False


def test_inline_source_only_spawning_a_gateway_later_stays_anonymous():
    # The #107002 rule enforced against THIS change: a child's argv mentioned without a
    # sys.argv self-assignment is spawn data, never identity.
    cmd = (
        "python -I -c import subprocess; subprocess.run(['python', '-m', "
        "'hermes_cli.main', 'gateway', 'run'])"
    )
    assert matches(cmd) is False
    assert matches_runtime(cmd) is False


def test_watcher_replaying_a_relaunch_shaped_argv_stays_anonymous():
    """The #107002 rule vs this fix's own side effect: once a relaunched gateway is
    identifiable, ``_capture_gateway_argv`` can capture it and the restart machinery replays
    its argv as the watcher's TRAILING data (``gateway.py::_spawn_gateway_restart_watcher`` /
    ``update_cmd_windows`` unmapped relaunch) — so that trailing data can itself be
    relaunch-shaped. The watcher is not the gateway: an assignment inside a second ``-c``
    region names the child the watcher will spawn, not the watcher's own argv."""
    cmd = (
        "/usr/bin/python3 -c import sys, time; pid = int(sys.argv[1]); cmd = sys.argv[2:]; "
        "time.sleep(30) 4242 /usr/bin/python3 -I -c import sys, runpy; "
        "sys.path.insert(0, '/opt/hermes-agent'); "
        "sys.argv = ['/opt/hermes-agent/hermes_cli/main.py', 'gateway', 'run']; "
        "runpy.run_module('hermes_cli.main', run_name='__main__', alter_sys=True)"
    )
    assert matches(cmd) is False
    assert matches_runtime(cmd) is False
    # Identity anonymous, intent preserved: the watcher WILL start that gateway after the old
    # PID exits, so the update's spawn-intent verification keeps answering "run".
    assert spawn_intent(cmd) == "run"


def test_shell_wrapper_embedding_a_relaunch_command_stays_anonymous():
    # Same rule for generic ``sh -c`` wrappers (e.g. launchd reload scripts): the assignment
    # belongs to a command the shell will run, not to the shell process itself.
    cmd = (
        "/bin/bash -c launchctl kickstart -k system/hermes.gateway; /usr/bin/python3 -I -c "
        "import sys, runpy; sys.argv = ['/opt/hermes-agent/hermes_cli/main.py', 'gateway', "
        "'run']; runpy.run_module('hermes_cli.main')"
    )
    assert matches(cmd) is False
    assert matches_runtime(cmd) is False


@pytest.mark.parametrize(
    "assignment",
    [
        "sys.argv = []",                                   # degenerate: empty list
        "sys.argv = [sys.executable, 'gateway', 'run']",   # computed, not a literal
        "sys.argv = [42, 'gateway', 'run']",               # non-string element
        "sys.argv",                                        # not even an assignment
    ],
)
def test_non_literal_or_degenerate_sys_argv_assignment_does_not_count(assignment):
    cmd = f"python -I -c import sys; {assignment}; import time; time.sleep(1)"
    assert matches(cmd) is False


@pytest.mark.platforms("linux")
@pytest.mark.spawns_gateway_lookalike
def test_live_relaunched_process_is_a_gateway_and_a_trailing_argv_spawner_is_not():
    """E2E on a real ``/proc/<pid>/cmdline``: a relaunch-shaped live process IS the gateway it
    re-binds to; the same interpreter shape with the gateway argv as TRAILING data (#107002
    watcher) is not. Pins behavior against the real unquoted-argv /proc spelling shlex mangles."""
    import subprocess
    import sys
    import time

    from gateway.status import _looks_like_gateway_process, _read_process_cmdline

    relaunch_inline = (
        "import time\n"
        "import sys; sys.argv = ['/opt/hermes-agent/hermes_cli/main.py', 'gateway', 'run']\n"
        "time.sleep(300)\n"
    )
    watcher_inline = "import sys, time\npid = int(sys.argv[1])\ncmd = sys.argv[2:]\ntime.sleep(300)\n"
    procs = [
        subprocess.Popen([sys.executable, "-I", "-c", relaunch_inline]),
        subprocess.Popen(
            [sys.executable, "-I", "-c", watcher_inline, "14980",
             sys.executable, "-m", "hermes_cli.main", "gateway", "run"]),
    ]
    try:
        deadline = time.time() + 15
        while time.time() < deadline:
            if all(_read_process_cmdline(p.pid) for p in procs):
                break
            time.sleep(0.05)
        assert _looks_like_gateway_process(procs[0].pid) is True
        assert _looks_like_gateway_process(procs[1].pid) is False
    finally:
        for proc in procs:
            proc.kill()
            proc.wait()


