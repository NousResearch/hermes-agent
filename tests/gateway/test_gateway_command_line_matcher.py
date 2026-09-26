"""Tests for the strict gateway command-line matcher.

Regression guard for the Windows ``hermes gateway restart`` silent-outage bug:
the previous loose substring match (``"... gateway" in cmdline``) false-matched
``gateway status``/``dashboard`` siblings and unrelated processes such as
``python -m tui_gateway``, which let ``restart()`` race a still-draining old
process and ``status``/``start`` report false positives.
"""

from __future__ import annotations

import base64
from pathlib import Path

import pytest

from gateway.status import (
    _published_launcher_source_matches as published_source_matches,
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


def _flatten(argv: list[str]) -> str:
    """Match _read_process_cmdline(): psutil argv parts are joined with spaces."""
    return " ".join(argv)


_RUNTIME_BOOTSTRAP = _flatten(
    runtime_command(Path("/tmp/hermes-agent"), ["gateway", "run", "--replace"], python="python")
)
_PUBLISHED_LAUNCHER = _flatten(
    ["python", "-I", "-c", _launcher_script("hermes", Path("/tmp/hermes-agent"), None), "gateway", "run"]
)
_WINDOWS_REDIRECTOR = (
    r"C:\Users\me\hermes\venv\Scripts\python.exe -I -c "
    "import sys, runpy; "
    r"sys.argv = ['C:\\Users\\me\\hermes\\venv\\Scripts\\hermes.exe', "
    "'-p', 'worker', 'gateway', 'run']; "
    "runpy.run_module('hermes_cli.main', run_name='__main__')"
)
HERMES_INLINE_GATEWAYS = [_RUNTIME_BOOTSTRAP, _PUBLISHED_LAUNCHER, _WINDOWS_REDIRECTOR]

HERMES_INLINE_GATEWAYS_BY_ROOT = [
    _flatten(runtime_command(Path(root), ["gateway", "run", "--replace"], python="python"))
    for root in ("/opt/hermes", "/opt/hermes-gateway", "/srv/hermes")
]


def _windows_cmd_fallback_gateway() -> str:
    """Reproduce mint_launcher's real base64/exec .cmd fallback without requiring Windows."""
    source = _launcher_script("hermes", Path("/opt/My Hermes"), None)
    encoded = base64.b64encode(source.encode("utf-8")).decode("ascii")
    code = f"import base64; exec(base64.b64decode('{encoded}'))"
    return _flatten(
        [
            r"C:\Program Files\Python\python.exe",
            "-I",
            "-c",
            code,
            "gateway",
            "run",
        ]
    )


def test_accepts_windows_command_file_fallback_launcher():
    """The supported .cmd fallback executes the generated launcher source in-process."""
    cmd = _windows_cmd_fallback_gateway()
    assert matches(cmd) is True
    assert matches_runtime(cmd) is True
    assert spawn_intent(cmd) == "run"


def test_non_python_inline_runtime_cannot_donate_gateway_identity():
    """Once inline launchers are acceptance-capable, a foreign -c runtime must stay anonymous."""
    cmd = (
        "bash -c import sys, runpy; "
        "sys.argv = ['/opt/hermes/hermes_cli/main.py', 'gateway', 'run']; "
        "runpy.run_module('hermes_cli.main', run_name='__main__')"
    )
    assert matches(cmd) is False
    assert matches_runtime(cmd) is False


def test_inert_full_published_launcher_fingerprint_stays_anonymous():
    """All marker strings as data are not equivalent to executing the generated launcher."""
    cmd = (
        "python -c import sys; "
        "note = \"import os, re, sys; os.environ.pop('PYTHONHOME', None); "
        "from hermes_constants import get_default_hermes_root; import hermes_bootstrap; "
        "from hermes_cli.main import main; sys.argv[0] = re.sub; sys.exit(main())\"; "
        "main = lambda: 0; sys.exit(main()) gateway run"
    )
    assert matches(cmd) is False
    assert matches_runtime(cmd) is False


def test_flattened_real_launcher_text_inside_string_stays_anonymous():
    """Flattening must not turn an inert copy of the real producer source into identity."""
    producer_source = _launcher_script("hermes", Path("/opt/hermes"), None).strip()
    producer_lines = producer_source.splitlines()
    flattened_producer = " ".join(producer_lines)
    inert_source = (
        producer_lines[0]
        + " note = \"\"\""
        + " ".join(producer_lines[1:])
        + "\"\"\" main = lambda: 0; sys.exit(main())"
    )

    # The real flattened producer is still recognized. Keeping its genuine first import outside
    # the string makes this exercise the lexical fallback rather than only the prefix guard.
    assert published_source_matches(flattened_producer) is True
    assert published_source_matches(inert_source) is False


def test_flattened_launcher_unicode_root_keeps_marker_offsets_stable():
    """Case-fold expansion in a real install path must not move lexical marker positions."""
    root = Path("/opt") / ("İ" * 32) / "hermes"
    producer_source = _launcher_script("hermes", root, None).strip()
    assert published_source_matches(" ".join(producer_source.splitlines())) is True


def test_spaced_python_path_watcher_stays_anonymous_but_keeps_intent():
    """psutil argv joining must not reopen #107002 when the outer interpreter path has spaces."""
    future = _flatten(
        runtime_command(
            Path("/opt/hermes"),
            ["gateway", "run", "--replace"],
            python="python",
        )
    )
    watcher = (
        r"C:\Program Files\Python\python.exe -c "
        "import time; time.sleep(1) 14980 "
        + future
    )
    assert matches(watcher) is False
    assert matches_runtime(watcher) is False
    assert spawn_intent(watcher) == "run"


@pytest.mark.parametrize("cmd", HERMES_INLINE_GATEWAYS)
def test_accepts_hermes_owned_inline_gateway_launchers(cmd):
    """Both canonical bootstrap shapes and the Windows redirector are this process, not a future one."""
    assert matches(cmd) is True
    assert matches_runtime(cmd) is True
    assert spawn_intent(cmd) == "run"


@pytest.mark.parametrize("cmd", HERMES_INLINE_GATEWAYS_BY_ROOT)
def test_inline_bootstrap_install_root_name_does_not_change_identity(cmd):
    """Install-path basenames are source data, not nested-program boundaries."""
    assert matches(cmd) is True
    assert matches_runtime(cmd) is True
    assert spawn_intent(cmd) == "run"


def _venv_sync_relaunch(module: str | None) -> str:
    from hermes_cli.venv_sync import relaunch_command

    root = Path("/opt/hermes")
    argv = [str(root / "hermes_cli" / "main.py"), "gateway", "run"]
    original = ["python", "-m", "hermes_cli.main", "gateway", "run"]
    return _flatten(relaunch_command(Path("python"), root, argv, original, module))


@pytest.mark.parametrize("module", ["hermes_cli.main", None])
def test_accepts_real_venv_sync_relaunch_shapes(module):
    """Both run_module and run_path producer branches rebind this process to gateway argv."""
    cmd = _venv_sync_relaunch(module)
    assert matches(cmd) is True
    assert matches_runtime(cmd) is True
    assert spawn_intent(cmd) == "run"


@pytest.mark.parametrize(
    "original",
    [
        runtime_command(
            Path("/opt/hermes"),
            ["gateway", "run", "--replace"],
            python="python",
        ),
        [
            "python",
            "-I",
            "-c",
            _launcher_script("hermes", Path("/opt/hermes"), None),
            "gateway",
            "run",
        ],
        [
            "python",
            "-I",
            "-c",
            "import base64; exec(base64.b64decode('"
            + base64.b64encode(
                _launcher_script("hermes", Path("/opt/hermes"), None).encode("utf-8")
            ).decode("ascii")
            + "'))",
            "gateway",
            "run",
        ],
    ],
)
def test_accepts_inline_launcher_after_venv_sync_reexec(original):
    """Lazy sync re-execs -c launchers through relaunch_command's literal exec branch."""
    from hermes_cli.venv_sync import relaunch_command

    flag_index = original.index("-c")
    argv = ["-c", *original[flag_index + 2 :]]
    cmd = _flatten(
        relaunch_command(Path("python"), Path("/opt/hermes"), argv, original, None)
    )
    assert matches(cmd) is True
    assert matches_runtime(cmd) is True
    assert spawn_intent(cmd) == "run"


def test_restart_watcher_with_unquoted_relaunch_data_preserves_spawn_intent():
    future = _venv_sync_relaunch("hermes_cli.main")
    watcher = (
        "python -c import sys, time; pid = int(sys.argv[1]); cmd = sys.argv[2:]; "
        f"time.sleep(30) 4242 {future}"
    )
    assert matches(watcher) is False
    assert matches_runtime(watcher) is False
    assert spawn_intent(watcher) == "run"


def test_unreachable_relaunch_entrypoint_does_not_grant_identity():
    """A syntactic runpy call after an unconditional exit is not executed gateway identity."""
    cmd = (
        "python -c import sys, runpy; "
        "sys.argv = ['/opt/hermes/hermes_cli/main.py', 'gateway', 'run']; "
        "__import__('time').sleep(3600); raise SystemExit; "
        "runpy.run_module('hermes_cli.main', run_name='__main__')"
    )
    assert matches(cmd) is False
    assert matches_runtime(cmd) is False


def test_marker_text_inside_inline_source_is_not_gateway_identity():
    cmd = (
        "python -c import sys; "
        "sys.argv = ['/opt/hermes/hermes_cli/main.py', 'gateway', 'run']; "
        "note = \"runpy.run_module('hermes_cli.main')\""
    )
    assert matches(cmd) is False
    assert matches_runtime(cmd) is False


def test_bootstrap_marker_text_is_not_gateway_identity():
    cmd = (
        "python -c import hermes_bootstrap; "
        "note = \"runpy.run_module('hermes_cli.main', alter_sys=True)\" gateway run"
    )
    assert matches(cmd) is False
    assert matches_runtime(cmd) is False


@pytest.mark.parametrize(
    "launch",
    [
        "runpy.run_module('hermes_cli.main')",
        "runpy.run_module('hermes_cli.main', run_name='probe')",
        "runpy.run_path('/opt/hermes/hermes_cli/main.py')",
        "runpy.run_path('/opt/hermes/hermes_cli/main.py', run_name='probe')",
    ],
)
def test_relaunch_target_without_main_execution_is_not_gateway_identity(launch):
    """Naming Hermes in runpy is insufficient unless the CLI is actually executed as __main__."""
    cmd = (
        "python -c import sys, runpy; "
        "sys.argv = ['/opt/hermes/hermes_cli/main.py', 'gateway', 'run']; "
        + launch
    )
    assert matches(cmd) is False
    assert matches_runtime(cmd) is False


def test_runtime_bootstrap_wrong_run_name_is_not_gateway_identity():
    """The runtime bootstrap must execute hermes_cli.main's __main__ guard, not only load it."""
    cmd = _RUNTIME_BOOTSTRAP.replace("run_name='__main__'", "run_name='probe'")
    assert cmd != _RUNTIME_BOOTSTRAP
    assert matches(cmd) is False
    assert matches_runtime(cmd) is False


@pytest.mark.parametrize("future_gateway", HERMES_INLINE_GATEWAYS)
def test_restart_watcher_wrapping_inline_gateway_stays_non_gateway(future_gateway):
    """#107002: a watcher's nested future gateway must never become the watcher's identity."""
    watcher = f'python -c "import time; time.sleep(1)" 14980 {future_gateway}'
    assert matches(watcher) is False
    assert matches_runtime(watcher) is False
    assert spawn_intent(watcher) == "run"


def test_foreign_inline_source_cannot_borrow_plain_gateway_tail():
    command = 'python -c "print(\'hermes_cli.main\')" gateway run'
    assert matches(command) is False
    assert matches_runtime(command) is False


@pytest.mark.parametrize("subcommand", ["status", "stop"])
def test_redirector_non_runtime_subcommands_stay_non_gateway(subcommand):
    command = _WINDOWS_REDIRECTOR.replace("'run']", f"'{subcommand}']")
    assert matches(command) is False
    assert matches_runtime(command) is False


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


