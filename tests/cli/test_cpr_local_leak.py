"""Local CPR leak reproduction + classic-CLI Application output selection.

* Deterministic local-PTY proof that delayed CPR replies leak as
  ``ESC[row;colR`` / ``^[[row;colR`` when ``enable_cpr=True`` (no SSH).
* Integration-level assertion that, with no SSH env vars, classic CLI
  output selection wires a CPR-disabled Output into Application on POSIX.
* Native Windows keeps prompt_toolkit's default output selection.
"""

from __future__ import annotations

import os
import select
import sys
import threading
import time

import pytest

from cli import (
    _build_cpr_disabled_output,
    _select_classic_cli_pt_output,
    _terminal_may_leak_cpr,
)


@pytest.fixture(autouse=True)
def _clear_cpr_env(monkeypatch):
    for var in ("SSH_CONNECTION", "SSH_CLIENT", "SSH_TTY", "PROMPT_TOOLKIT_NO_CPR"):
        monkeypatch.delenv(var, raising=False)


class TestClassicCliOutputSelection:
    def test_posix_local_without_ssh_selects_cpr_disabled_output(self, monkeypatch):
        """Changed contract: no SSH vars, still CPR-disabled on POSIX."""
        monkeypatch.setattr(sys, "platform", "linux")
        assert _terminal_may_leak_cpr() is True
        out = _select_classic_cli_pt_output(sys.stdout)
        assert out is not None
        assert out.enable_cpr is False

    def test_application_receives_cpr_not_supported_without_ssh(self, monkeypatch):
        """Classic-CLI Application construction must get CPR-disabled output."""
        from prompt_toolkit.application import Application
        from prompt_toolkit.layout import FormattedTextControl, Layout, Window
        from prompt_toolkit.renderer import CPR_Support

        monkeypatch.setattr(sys, "platform", "linux")
        out = _select_classic_cli_pt_output(sys.stdout)
        assert out is not None

        app = Application(
            layout=Layout(Window(FormattedTextControl("x"))),
            output=out,
            full_screen=False,
        )
        assert app.renderer.cpr_support == CPR_Support.NOT_SUPPORTED

    def test_windows_preserves_default_output_selection(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        assert _terminal_may_leak_cpr() is False
        assert _select_classic_cli_pt_output(sys.stdout) is None

    def test_windows_honors_explicit_no_cpr(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setenv("PROMPT_TOOLKIT_NO_CPR", "1")
        assert _terminal_may_leak_cpr() is True
        out = _select_classic_cli_pt_output(sys.stdout)
        # Build may return None if stdout is not a real tty in CI; if it
        # succeeds it must be CPR-disabled.
        assert out is None or out.enable_cpr is False


def _openpty_or_skip():
    import pty

    try:
        return pty.openpty()
    except OSError as exc:
        pytest.skip(f"no PTY devices available: {exc}")


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX PTY harness")
class TestDelayedCprLocalPtyLeak:
    def test_delayed_cpr_reply_leaks_when_enable_cpr_true(self):
        """Local (no SSH) delayed ESC[6n reply lands as ESC[39;1R on stdin."""
        import tty

        from prompt_toolkit.data_structures import Size
        from prompt_toolkit.output.vt100 import Vt100_Output

        master, slave = _openpty_or_skip()
        try:
            tty.setraw(slave)
            slave_w = os.fdopen(os.dup(slave), "w", buffering=1)
            stop = threading.Event()
            queries = 0

            def terminal() -> None:
                nonlocal queries
                buf = b""
                while not stop.is_set():
                    try:
                        r, _, _ = select.select([master], [], [], 0.05)
                    except OSError:
                        break
                    if not r:
                        continue
                    try:
                        chunk = os.read(master, 4096)
                    except OSError:
                        break
                    if not chunk:
                        break
                    buf += chunk
                    while True:
                        idx = buf.find(b"\x1b[6n")
                        if idx < 0:
                            buf = buf[-8:] if len(buf) > 8 else buf
                            break
                        buf = buf[idx + 4 :]
                        queries += 1
                        time.sleep(0.12)
                        try:
                            os.write(master, b"\x1b[39;1R")
                        except OSError:
                            pass

            threading.Thread(target=terminal, daemon=True).start()
            out = Vt100_Output(
                slave_w, lambda: Size(rows=40, columns=80), enable_cpr=True
            )
            out.ask_for_cpr()
            out.flush()
            for i in range(4):
                slave_w.write(f"\rgpt-5.6-sol Q {i}\n")
                slave_w.flush()
                time.sleep(0.02)
            time.sleep(0.3)

            data = b""
            while True:
                r, _, _ = select.select([slave], [], [], 0.05)
                if not r:
                    break
                data += os.read(slave, 4096)

            stop.set()
            slave_w.close()

            assert queries >= 1
            assert b"\x1b[39;1R" in data
        finally:
            try:
                os.close(slave)
            except OSError:
                pass
            try:
                os.close(master)
            except OSError:
                pass

    def test_cpr_disabled_output_sends_no_query(self):
        """Hermes CPR-disabled builder must not emit ESC[6n."""
        master, slave = _openpty_or_skip()
        try:
            slave_w = os.fdopen(slave, "w", buffering=1)
            out = _build_cpr_disabled_output(slave_w)
            assert out is not None
            assert out.enable_cpr is False

            seen = b""

            def reader() -> None:
                nonlocal seen
                r, _, _ = select.select([master], [], [], 0.25)
                if r:
                    seen = os.read(master, 4096)

            threading.Thread(target=reader, daemon=True).start()
            slave_w.write("status ok\n")
            slave_w.flush()
            # Do not call ask_for_cpr — renderer skips it when NOT_SUPPORTED.
            time.sleep(0.3)
            slave_w.close()
            assert b"\x1b[6n" not in seen
        finally:
            try:
                os.close(master)
            except OSError:
                pass
