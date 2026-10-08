"""Reader-boundary unescape of macOS ps's literal backslash-octal control characters (#126887).

macOS ``ps`` prints an argv-embedded newline as the four literal characters ``\\012`` (a tab as
``\\011``) so each process stays on one output line. Every matcher downstream of a ps text read
(``_hermes_holder_subcommand``, ``_gateway_command_subcommand``, ``_parse_dashboard_runtime``,
the dashboard/orphan scans) then receives a corrupted inline bootstrap source — the
bootstrap regexes anchor on real whitespace — and goes blind on POSIX-launcher processes.

These tests build ps lines in the exact shape of the live launcher-started gateway and assert
the ps read boundaries this PR touches hand the matchers unescaped text. No source-shape or
source-text assertions: every assertion goes through the real reader + matcher pipeline.

The gateway sweep's ps reads (``_parse_ps_line``, the ``_read_process_cmdline`` ps arm) are
deliberately NOT unescaped here: #126887 asks for ``_get_service_pids()`` to cover a launchd
job's descendants first, otherwise a decoded gateway command line makes ``hermes update``'s
manual-sweep mistake a launchd-managed gateway (the macOS job PID is the osascript wrapper,
not the gateway child) for drain/SIGTERM. The dashboard respawn path is unaffected: it reads
exact argv via ``/proc``/psutil, and its ps arm only feeds the runtime matcher.
"""

import shlex
import subprocess
import sys

import pytest

import hermes_cli.gateway as gateway_cli
from gateway import status as gateway_status
from hermes_cli import dashboard_procs, main_dashboard

_INTERPRETER = "/Users/forge/.hermes/tools/python-3.14.7-darwin-arm64/bin/python3"

# Faithful shape of the published POSIX launcher's inline bootstrap (hermes_cli._launchers):
# one argv element containing REAL newlines — what the kernel handes ps before ps escaped them.
_LAUNCHER_SOURCE = (
    "import os, re, sys\n"
    "os.environ.pop('PYTHONHOME', None)\n"
    "os.environ.pop('PYTHONPATH', None)\n"
    "sys.path.insert(0, '/Users/forge/workspace/hermes-agent')\n"
    "import hermes_bootstrap\n"
    "from hermes_cli.main import main\n"
    "sys.argv[0] = re.sub(r'(-script\\.pyw|\\.exe)?$', '', sys.argv[0])\n"
    "sys.exit(main())\n"
)


def _ps_escaped(command: str) -> str:
    """What macOS ``ps`` prints for *command*: newline/tab as literal backslash-octal."""
    return command.replace("\n", "\\012").replace("\t", "\\011")


def _gateway_command() -> str:
    return f"{_INTERPRETER} -I -c {_LAUNCHER_SOURCE} gateway run --external-supervisor"


def _dashboard_command() -> str:
    return f"{_INTERPRETER} -I -c {_LAUNCHER_SOURCE} dashboard --host 127.0.0.1 --port 9119"


@pytest.fixture(autouse=True)
def _darwin(monkeypatch):
    # The unescape is darwin-only (procps ps does not octal-escape); pin the platform so the
    # pipeline tests exercise the translation on every CI host.
    monkeypatch.setattr(sys, "platform", "darwin")


def test_helper_unescapes_only_on_darwin(monkeypatch):
    from hermes_cli._subprocess_compat import unescape_ps_command

    assert unescape_ps_command("a\\012b\\011c") == "a\nb\tc"
    monkeypatch.setattr(sys, "platform", "linux")
    assert unescape_ps_command("a\\012b") == "a\\012b"



def test_process_table_reader_feeds_holder_matcher_unescaped(monkeypatch):
    from hermes_cli.update_cmd_windows import _hermes_holder_subcommand

    escaped = _ps_escaped(_gateway_command())
    stdout = f"  93780 {escaped}\n  12345 /usr/bin/python3 -m hermes_cli.main gateway run\n"
    fake = subprocess.CompletedProcess(args=["ps"], returncode=0, stdout=stdout, stderr="")
    monkeypatch.setattr(dashboard_procs.subprocess, "run", lambda *a, **k: fake)

    table = dict(dashboard_procs._iter_process_table())

    assert "\\012" not in table[93780]
    assert "\n" in table[93780]
    assert _hermes_holder_subcommand(table[93780]) == "gateway"
    # Plain argv (no embedded newlines) is byte-identical through the boundary.
    assert _hermes_holder_subcommand(table[12345]) == "gateway"


def test_dashboard_cmdline_roundtrip_parses_runtime(monkeypatch):
    escaped = _ps_escaped(_dashboard_command())
    fake = subprocess.CompletedProcess(args=["ps"], returncode=0, stdout=escaped + "\n", stderr="")
    monkeypatch.setattr(main_dashboard, "_run_probe", lambda *a, **k: fake)

    argv = main_dashboard._dashboard_cmdline_for_pid(987654321)

    assert argv is not None
    assert main_dashboard._parse_dashboard_runtime(shlex.join(argv)) == ("dashboard", "127.0.0.1", 9119)


def test_dashboard_cmdline_respawn_argv_replayable(monkeypatch):
    """The respawn path must get exact argv elements, not a shlex split of ps text.

    ps text is space-joined and cannot express argv element boundaries, so shlex.split of even
    fully unescaped text still shreds the inline bootstrap source into dozens of tokens
    (#126887). psutil returns real argv elements; the ps arm remains only as the fallback for
    processes psutil cannot read (another user's process on macOS), where the runtime matcher
    still works but respawn replay is best-effort.
    """
    import psutil

    argv_elements = [_INTERPRETER, "-I", "-c", _LAUNCHER_SOURCE, "dashboard", "--host", "127.0.0.1", "--port", "9119"]

    class _FakeProcess:
        def __init__(self, pid):
            self.pid = pid

        def cmdline(self):
            return list(argv_elements)

    monkeypatch.setattr(psutil, "Process", _FakeProcess)
    escaped = _ps_escaped(_dashboard_command())
    fake = subprocess.CompletedProcess(args=["ps"], returncode=0, stdout=escaped + "\n", stderr="")
    monkeypatch.setattr(main_dashboard, "_run_probe", lambda *a, **k: fake)

    argv = main_dashboard._dashboard_cmdline_for_pid(987654321)

    assert argv is not None
    assert argv[argv.index(_LAUNCHER_SOURCE)] == _LAUNCHER_SOURCE
    assert "dashboard" in argv
    assert "9119" in argv


def test_unescape_is_lossy_for_literal_octal_text():
    """Documented lossiness, pinned so a future change re-decides it deliberately.

    BSD ps does not escape backslashes, so a literal ``\\012`` typed into an argv element is
    byte-identical to an escaped newline and the helper cannot tell them apart — it always
    restores the newline. Only matcher/respawn readers consume this text, and they anchor on
    real whitespace, so the trade favors restoring.
    """
    from hermes_cli._subprocess_compat import unescape_ps_command

    assert unescape_ps_command("sed \\012 pattern") == "sed \n pattern"
    assert unescape_ps_command("cut \\011 field") == "cut \t field"


