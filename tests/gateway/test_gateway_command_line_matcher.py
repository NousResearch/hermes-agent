"""Tests for the strict gateway command-line matcher.

Regression guard for the Windows ``hermes gateway restart`` silent-outage bug:
the previous loose substring match (``"... gateway" in cmdline``) false-matched
``gateway status``/``dashboard`` siblings and unrelated processes such as
``python -m tui_gateway``, which let ``restart()`` race a still-draining old
process and ``status``/``start`` report false positives.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from gateway.status import (
    gateway_spawn_intent_subcommand as spawn_intent,
    looks_like_gateway_command_line as matches,
    looks_like_gateway_runtime_command_line as matches_runtime,
)
from hermes_cli._launchers import _launcher_script, runtime_command


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


# The bootstrap launcher (hermes_cli/_launchers.py) is ALSO ``python -I -c <src> …``: the
# ``.hermes/bin/hermes`` script every systemd/launchd unit execs, and ``runtime_command`` behind
# ``gateway start`` and the dashboard's restart. Its tail is the argv the source hands to
# ``hermes_cli.main`` in-process, so it names THIS process. Refusing it along with the watcher made
# every status probe unlink a live gateway's gateway.pid/gateway.lock (#123109). Built from the real
# builders and joined with spaces, exactly as /proc and psutil render argv.
REPO = Path("/opt/hermes-agent")


def _bootstrap_one_liner(*args: str, repo: Path = REPO) -> str:
    return " ".join(runtime_command(repo, args, python="python3"))


def _shell_launcher(*args: str, repo: Path = REPO) -> str:
    return " ".join(["python3", "-I", "-c", _launcher_script("hermes", repo, None), *args])


@pytest.mark.parametrize(
    ("cmd", "subcommand"),
    [
        (_bootstrap_one_liner("gateway", "run"), "run"),
        (_bootstrap_one_liner("gateway", "run", "--replace"), "run"),
        (_bootstrap_one_liner("-p", "work", "gateway", "run"), "run"),
        (_bootstrap_one_liner("gateway", "restart"), "restart"),
        (_bootstrap_one_liner("gateway", "status"), "status"),
        (_bootstrap_one_liner("dashboard"), None),
        (_shell_launcher("gateway", "run"), "run"),
        (_shell_launcher("gateway", "--profile", "work", "run"), "run"),
        (_shell_launcher("gateway", "stop"), "stop"),
    ],
)
def test_bootstrap_launcher_tail_identifies_this_process(cmd, subcommand):
    assert matches(cmd) is (subcommand == "run")
    assert matches_runtime(cmd) is (subcommand in {"run", "restart"})
    assert spawn_intent(cmd) == subcommand


# Mirror image: the discriminator is the tail naming a program, not the ``-c`` itself, so the
# bootstrap-wrapped Windows restart watcher (gateway/run_shutdown.py) — whose tail is a whole
# bootstrap-launched gateway command line — must stay unrecognised as a live gateway while its
# spawn intent still resolves.
WATCHER_WRAPPING_BOOTSTRAP = " ".join(runtime_command(
    REPO,
    ["14980", "30", *runtime_command(REPO, ["gateway", "restart"], python="python3")],
    code="import time; time.sleep(30)",
    python="python3",
))


def test_watcher_wrapping_a_bootstrap_launched_gateway_is_not_a_gateway():
    assert matches(WATCHER_WRAPPING_BOOTSTRAP) is False
    assert matches_runtime(WATCHER_WRAPPING_BOOTSTRAP) is False
    assert spawn_intent(WATCHER_WRAPPING_BOOTSTRAP) == "restart"


# The launcher embeds the install root in its source as string literals, and the source's words are
# scanned along with the tail. A folder can be named anything — a checkout at ``~/hermes``, or a
# root with spaces that /proc splits into separate words — and none of it names a program or a
# gateway entrypoint.
@pytest.mark.parametrize(
    "root",
    [
        "/opt/hermes",
        "/opt/hermes-gateway",
        "/opt/python3",
        "/opt/Python Projects/hermes-agent",
        "/opt/my hermes install/hermes-agent",
    ],
)
@pytest.mark.parametrize("launcher", [_bootstrap_one_liner, _shell_launcher])
def test_launcher_recognised_whatever_its_install_root_is_named(launcher, root):
    assert matches(launcher("gateway", "run", repo=Path(root))) is True
    assert spawn_intent(launcher("gateway", "status", repo=Path(root))) == "status"


# The mirror image: words inside the source's string literals never speak for the process, so a
# script that shells out to a gateway command line is not that gateway — though its spawn intent
# still is.
def test_inline_source_shelling_out_to_a_gateway_is_not_a_gateway():
    source = "import os; os.system('/usr/local/bin/hermes gateway run --replace')"
    cmd = " ".join(["python3", "-c", source])
    assert matches(cmd) is False
    assert matches_runtime(cmd) is False
    assert spawn_intent(cmd) == "run"


# A named profile puts its name in the tail and — through the systemd unit's
# ``installation_command(home=…)`` on installs without a store Python — its HERMES_HOME in the
# source. Either can read like a program.
@pytest.mark.parametrize("profile", ["hermes", "python-bot"])
def test_profile_named_like_a_program_is_still_this_process(profile):
    home = f"/opt/state/.hermes/profiles/{profile}"
    unit = " ".join(runtime_command(
        REPO, ["--profile", profile, "gateway", "run"], python="python3", home=home,
    ))
    assert matches(unit) is True
    assert matches(_shell_launcher("-p", profile, "gateway", "run")) is True


# Windows process listings (Get-CimInstance) keep the ``-c`` operand as ONE quoted argument and
# quote every other argument that contains a space. The operand is source, never a program; a
# quoted interpreter in the tail still names one, with or without a ``-m`` after it.
WINDOWS_PYTHON = r"C:\Users\Jo Doe\AppData\Local\hermes\python\python.exe"
APOSTROPHE_PYTHON = r"C:\Users\O'Brien\python.exe"


def test_windows_listing_of_a_launcher_is_this_process():
    argv = runtime_command(Path("/opt/python3"), ["gateway", "run"], python=WINDOWS_PYTHON)
    assert matches(subprocess.list2cmdline(argv)) is True


@pytest.mark.parametrize(
    "cmd",
    [
        subprocess.list2cmdline([
            WINDOWS_PYTHON, "-c", "import time; time.sleep(30)", "14980",
            *runtime_command(REPO, ["gateway", "run", "--replace"], python=WINDOWS_PYTHON),
        ]),
        # /proc join: a quote inside a word (``O'Brien``) does not open a string literal.
        " ".join([
            APOSTROPHE_PYTHON, "-c", "import time; time.sleep(30)", "14980",
            *runtime_command(REPO, ["gateway", "run", "--replace"], python=APOSTROPHE_PYTHON),
        ]),
    ],
    ids=["windows-listing", "proc-join-apostrophe"],
)
def test_watcher_replaying_a_bootstrap_launched_gateway_is_not_a_gateway(cmd):
    assert matches(cmd) is False
    assert matches_runtime(cmd) is False
    assert spawn_intent(cmd) == "run"


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


