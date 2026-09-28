"""Reader-boundary unescape of macOS ps's literal backslash-octal control characters (#126887).

macOS ``ps`` prints an argv-embedded newline as the four literal characters ``\\012`` (a tab as
``\\011``) so each process stays on one output line. Every matcher downstream of a ps text read
(``_hermes_holder_subcommand``, ``_gateway_command_subcommand``, ``_parse_dashboard_runtime``,
the dashboard/orphan scans) then receives a corrupted inline bootstrap source — the
bootstrap regexes anchor on real whitespace — and goes blind on POSIX-launcher processes.

These tests build ps lines in the exact shape of the live launcher-started gateway and assert
each of the four ps read boundaries hands the matchers unescaped text. No source-shape or
source-text assertions: every assertion goes through the real reader + matcher pipeline.
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


def test_status_ps_fallback_feeds_matchers_unescaped(monkeypatch):
    escaped = _ps_escaped(_gateway_command())
    fake = subprocess.CompletedProcess(args=["ps"], returncode=0, stdout=escaped + "\n", stderr="")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: fake)

    # A pid with no /proc entry and no psutil process, so the ps fallback is exercised.
    command = gateway_status._read_process_cmdline(987654321)

    assert command is not None
    assert "\\012" not in command
    assert "\n" in command
    assert gateway_status._gateway_command_subcommand(command) == "run"


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


def test_gateway_ps_line_parser_feeds_matcher_unescaped():
    escaped = _ps_escaped(_gateway_command())

    parsed = gateway_cli._parse_ps_line(f"  93780 {escaped}")

    assert parsed is not None
    pid, command = parsed
    assert pid == 93780
    assert "\\012" not in command
    assert "\n" in command
    assert gateway_status._gateway_command_subcommand(command) == "run"
