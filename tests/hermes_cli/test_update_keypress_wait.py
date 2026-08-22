"""Regression tests for /update's post-run keypress pause."""

from __future__ import annotations

import argparse
import io
import sys
from types import SimpleNamespace

import pytest

from hermes_cli import main
from hermes_cli.subcommands.update import build_update_parser


class _TtyInput(io.StringIO):
    def isatty(self) -> bool:
        return True


class _NonTtyInput(io.StringIO):
    def isatty(self) -> bool:
        return False


class _BrokenOutput:
    def write(self, _text):
        raise OSError("terminal closed")

    def flush(self):
        raise OSError("terminal closed")


def _parse_update(*argv: str):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    build_update_parser(subparsers, cmd_update=lambda _args: None)
    return parser.parse_args(["update", *argv])


def test_update_parser_accepts_internal_wait_for_keypress_flag():
    args = _parse_update("--wait-for-keypress")

    assert args.wait_for_keypress is True


def test_wait_for_update_keypress_reads_one_key_and_prints_prompt(monkeypatch):
    stdin = _TtyInput()
    stdout = io.StringIO()
    reads = []
    monkeypatch.setattr(main, "_read_single_key", lambda stream: reads.append(stream))

    main._wait_for_update_keypress(stdin=stdin, stdout=stdout)

    assert reads == [stdin]
    assert "Press any key to close this window" in stdout.getvalue()


def test_wait_for_update_keypress_skips_noninteractive_stdin(monkeypatch):
    stdin = _NonTtyInput()
    stdout = io.StringIO()
    monkeypatch.setattr(
        main,
        "_read_single_key",
        lambda _stream: pytest.fail("non-interactive input must not be read"),
    )

    main._wait_for_update_keypress(stdin=stdin, stdout=stdout)

    assert stdout.getvalue() == ""


def test_wait_for_update_keypress_preserves_result_when_output_is_closed(monkeypatch):
    monkeypatch.setattr(
        main,
        "_read_single_key",
        lambda _stream: pytest.fail("a closed terminal cannot be dismissed"),
    )

    main._wait_for_update_keypress(stdin=_TtyInput(), stdout=_BrokenOutput())


def test_read_single_key_restores_posix_terminal_after_read_failure(monkeypatch):
    calls = []
    previous = object()

    def fail_read(_size):
        raise RuntimeError("read failed")

    stream = SimpleNamespace(fileno=lambda: 23, read=fail_read)
    termios = SimpleNamespace(
        TCSADRAIN=99,
        tcgetattr=lambda fd: calls.append(("get", fd)) or previous,
        tcsetattr=lambda fd, when, state: calls.append(("restore", fd, when, state)),
    )
    tty = SimpleNamespace(setraw=lambda fd: calls.append(("raw", fd)))
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setitem(sys.modules, "termios", termios)
    monkeypatch.setitem(sys.modules, "tty", tty)

    with pytest.raises(RuntimeError, match="read failed"):
        main._read_single_key(stream)

    assert calls == [("get", 23), ("raw", 23), ("restore", 23, 99, previous)]


def test_read_single_key_uses_msvcrt_getwch_on_windows(monkeypatch):
    calls = []
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setitem(
        sys.modules,
        "msvcrt",
        SimpleNamespace(getwch=lambda: calls.append("getwch")),
    )
    stream = SimpleNamespace(fileno=lambda: pytest.fail("Windows must not use fileno"))

    main._read_single_key(stream)

    assert calls == ["getwch"]


def test_cmd_update_waits_after_update_completion(monkeypatch):
    calls = []
    monkeypatch.setattr(main, "_cmd_update_without_keypress_wait", lambda args: calls.append(("update", args)))
    monkeypatch.setattr(main, "_wait_for_update_keypress", lambda: calls.append(("wait", None)))
    args = SimpleNamespace(wait_for_keypress=True)

    main.cmd_update(args)

    assert calls == [("update", args), ("wait", None)]


@pytest.mark.parametrize(
    "pause_error",
    [ValueError("no fileno"), RuntimeError("pause failed"), SystemExit(9)],
)
def test_cmd_update_preserves_successful_result_when_pause_fails(monkeypatch, pause_error):
    expected = object()
    monkeypatch.setattr(main, "_cmd_update_without_keypress_wait", lambda _args: expected)

    def fail_pause():
        raise pause_error

    monkeypatch.setattr(main, "_wait_for_update_keypress", fail_pause)

    result = main.cmd_update(SimpleNamespace(wait_for_keypress=True))

    assert result is expected


def test_cmd_update_waits_after_update_failure(monkeypatch):
    calls = []

    def fail(_args):
        calls.append(("update", None))
        raise SystemExit(7)

    monkeypatch.setattr(main, "_cmd_update_without_keypress_wait", fail)
    monkeypatch.setattr(main, "_wait_for_update_keypress", lambda: calls.append(("wait", None)))

    with pytest.raises(SystemExit, match="7"):
        main.cmd_update(SimpleNamespace(wait_for_keypress=True))

    assert calls == [("update", None), ("wait", None)]


@pytest.mark.parametrize("pause_error", [ValueError("no fileno"), SystemExit(9)])
def test_cmd_update_preserves_original_system_exit_when_pause_fails(monkeypatch, pause_error):
    def fail_update(_args):
        raise SystemExit(7)

    def fail_pause():
        raise pause_error

    monkeypatch.setattr(main, "_cmd_update_without_keypress_wait", fail_update)
    monkeypatch.setattr(main, "_wait_for_update_keypress", fail_pause)

    with pytest.raises(SystemExit) as exc_info:
        main.cmd_update(SimpleNamespace(wait_for_keypress=True))

    assert exc_info.value.code == 7


def test_direct_cmd_update_does_not_wait(monkeypatch):
    calls = []
    monkeypatch.setattr(main, "_cmd_update_without_keypress_wait", lambda _args: calls.append("update"))
    monkeypatch.setattr(main, "_wait_for_update_keypress", lambda: calls.append("wait"))

    main.cmd_update(SimpleNamespace(wait_for_keypress=False))

    assert calls == ["update"]
