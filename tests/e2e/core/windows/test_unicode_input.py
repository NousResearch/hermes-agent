"""Non-ASCII / emoji user input on native Windows reaches the model intact.

Three real input paths a Windows user hits:

* ``hermes chat -q "<text>"`` — argv (CreateProcessW) into the oneshot turn; the reply,
  with its own emoji, is printed back through a pipe (Windows code-page territory).
* the TUI gateway's stdio JSON-RPC — UTF-8 frames from the Ink TUI (Node writes literal
  non-ASCII, never ``\\u`` escapes).
* the classic interactive CLI (prompt_toolkit) typed into a real console: a ConPTY driven
  through pywinpty, the same console layer Windows Terminal uses (#120776).
"""

from __future__ import annotations

import re
import subprocess
import threading
import time
from pathlib import Path

import pytest

from tests.e2e.core.windows._helpers import (
    hermes,
    hermes_argv,
    kill_tree,
    last_user,
    make_home,
    nonce,
    persisted_messages,
    process_tree,
    wait_until,
)
from tests.e2e.core.windows._rpc import StdioGateway
from tests.fakes.fake_llm_provider import FakeLLMServer, Text

pytestmark = [pytest.mark.windows_only, pytest.mark.integration]

TEXT = "Grüße, 日本語 und Emoji 😂👍🏽"
REPLY = "Réponse ✓ 😂"
BMP = "Grüße, 日本語"


def test_chat_q_argv_non_ascii_reaches_wire_and_state_db(tmp_path: Path) -> None:
    tag = nonce("ARGV")
    with FakeLLMServer([Text(f"{REPLY} {tag}")]) as srv:
        home = make_home(tmp_path, srv.base_url)
        res = hermes(home, "chat", "-q", f"{TEXT} {tag}", "-Q")
        assert res.returncode == 0, res.tail()
        user = last_user(srv.main_requests()[0])
    assert f"{TEXT} {tag}" in user, f"prompt altered before the wire: {user!r}"
    stored = [r["content"] for r in persisted_messages(home) if r["role"] == "user"]
    assert any(f"{TEXT} {tag}" in (c or "") for c in stored), f"prompt altered in state.db: {stored!r}"


def test_chat_q_non_ascii_reply_printed_intact(tmp_path: Path) -> None:
    tag = nonce("REPLY")
    with FakeLLMServer([Text(f"{REPLY} {tag}")]) as srv:
        home = make_home(tmp_path, srv.base_url)
        res = hermes(home, "chat", "-q", "Reply with the code.", "-Q")
    assert res.returncode == 0, res.tail()
    assert tag in res.stdout, f"reply not printed at all:\n{res.tail()}"
    assert f"{REPLY} {tag}" in res.stdout, f"reply printed with its non-ASCII mangled: {res.stdout[-500:]!r}"


def test_tui_gateway_utf8_frames_reach_wire(tmp_path: Path) -> None:
    tag = nonce("TUI")
    with FakeLLMServer([Text(f"{REPLY} {tag}")]) as srv:
        home = make_home(tmp_path, srv.base_url)
        gw = StdioGateway(home)
        try:
            answer = gw.turn(f"{TEXT} {tag}")
        finally:
            gw.close()
        user = last_user(srv.main_requests()[0])
    assert tag in user, f"prompt never reached the provider: {user!r}"
    assert f"{TEXT} {tag}" in user, f"TUI gateway altered the prompt before the wire: {user!r}"
    assert f"{REPLY} {tag}" in answer, f"reply altered on the way back to the TUI: {answer!r}"


class _Console:
    """A real ConPTY (pywinpty) running the classic CLI; output is pumped on a thread."""

    def __init__(self, argv: list[str], cwd: Path, env: dict[str, str]) -> None:
        from winpty import PtyProcess

        self.proc = PtyProcess.spawn(subprocess.list2cmdline(argv), cwd=str(cwd), env=env,
                                     dimensions=(50, 200))
        self._chunks: list[str] = []
        self._last = time.monotonic()
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self) -> None:
        while True:
            try:
                data = self.proc.read(4096)
            except (EOFError, OSError):
                return
            if data:
                self._chunks.append(data)
                self._last = time.monotonic()

    @property
    def screen(self) -> str:
        return "".join(self._chunks)

    def quiet_for(self, seconds: float) -> bool:
        return bool(self._chunks) and time.monotonic() - self._last >= seconds

    def close(self) -> None:
        tree = process_tree(self.proc.pid)
        try:
            self.proc.terminate(force=True)
        except Exception:  # noqa: BLE001 - teardown of an already-dead console
            pass
        kill_tree(tree)


_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07]*\x07")


def _plain(screen: str) -> str:
    return _ANSI.sub("", screen)


def test_classic_cli_console_non_ascii_reaches_wire(tmp_path: Path) -> None:
    """Typed into the classic CLI composer through a real ConPTY: Latin-1 + CJK letters.
    Symbols (✓, emoji) typed this way never reached the composer (not echoed before Enter);
    without a harness control proving pywinpty delivers them, that loss can't be pinned on
    Hermes, so they are left out here (#120776 is the submit-time half)."""
    tag = nonce("CONPTY")
    with FakeLLMServer([Text(f"ack {tag}")]) as srv:
        home = make_home(tmp_path, srv.base_url)
        console = _Console(hermes_argv("chat"), home.project, home.env({"TERM": "xterm-256color"}))
        try:
            wait_until(lambda: console.quiet_for(3.0), 120, "the classic CLI to finish painting its prompt")
            console.proc.write(f"{BMP} {tag}")
            wait_until(lambda: tag in console.screen, 30, "the composer to echo the typed text")
            echoed = _plain(console.screen)
            # An Enter within 50 ms of the last composer change is read as a pasted newline, and the
            # echo can paint inside that window: give the Enter real time after the last typed key.
            time.sleep(0.5)
            console.proc.write("\r")
            wait_until(lambda: srv.main_requests(), 90, "the typed turn to reach the provider")
            user = last_user(srv.main_requests()[0])
        finally:
            screen = console.screen
            console.close()
    assert tag in user, f"typed text never reached the provider: {user!r}\n{_plain(screen)[-2000:]}"
    assert f"{BMP} {tag}" in user, (
        f"non-ASCII lost between the console and the wire: {ascii(user)}\n"
        f"composer echoed it before Enter: {BMP in echoed}\nconsole tail:\n{ascii(_plain(screen)[-600:])}")
