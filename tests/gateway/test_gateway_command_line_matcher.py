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


# The installation's OWN published launcher (``hermes_cli._launchers.runtime_command``):
# ``python -I -c <bootstrap source> gateway run``. The bootstrap source IS the install's identity —
# it clears the ambient interpreter environment, puts the checkout on ``sys.path`` and loads
# ``hermes_bootstrap`` (which selects and leases a dependency generation). ``hermes update``'s
# self-relaunch respawns exactly this argv, so rejecting every inline source rejected the install's
# own gateway: the updater deleted ``gateway.pid`` for a process that was alive, reported "no stable
# gateway process appeared", aborted the update, and left a gateway ``gateway stop --all`` could not
# stop. #107002 only ever meant the restart WATCHER, whose source carries neither marker.
def _install_launcher_source() -> str:
    from pathlib import Path

    from hermes_cli._launchers import runtime_command

    argv = runtime_command(Path(__file__).resolve().parents[2], ["gateway", "run"])
    return argv[argv.index("-c") + 1]


def test_accepts_the_installations_own_published_launcher():
    src = _install_launcher_source()
    assert matches(f'python -I -c "{src}" gateway run') is True
    assert matches_runtime(f'python -I -c "{src}" gateway run --replace') is True
    # profile selector anywhere in argv, and the bare `gateway` that defaults to run
    assert matches(f'python -I -c "{src}" --profile work gateway run') is True
    assert matches(f'python -I -c "{src}" gateway') is True


def test_installations_own_launcher_wrapping_a_sibling_subcommand_is_not_a_gateway():
    src = _install_launcher_source()
    assert matches(f'python -I -c "{src}" gateway status') is False
    assert matches(f'python -I -c "{src}" --profile x dashboard') is False


def test_accepts_the_launcher_as_a_live_process_readback():
    """A live process is read back through psutil, which joins argv with spaces: the inline source
    arrives UNQUOTED and re-tokenized on its own spaces (the real launcher is ~29 tokens), so the
    token after ``-c`` is just ``import``. Matching markers against that single token never fires."""
    src = _install_launcher_source()
    assert matches_runtime(f"python -I -c {src} gateway run --replace") is True


# Both markers are required, so an unrelated inline source that merely mentions one of them stays
# rejected — the launcher is identified by its whole source, not a keyword.
INLINE_SOURCE_WITH_ONE_MARKER = [
    'python -c "import hermes_bootstrap" 14980 python -m hermes_cli.main gateway run',
    'python -c "import runpy; runpy.run_module(\'x\')" 14980 python -m hermes_cli.main gateway run',
]


@pytest.mark.parametrize("cmd", INLINE_SOURCE_WITH_ONE_MARKER)
def test_partial_launcher_markers_do_not_manufacture_a_gateway(cmd):
    assert matches(cmd) is False
    assert matches_runtime(cmd) is False


