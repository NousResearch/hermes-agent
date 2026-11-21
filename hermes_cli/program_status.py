"""Program Status Protocol (OSC 7501) reports for the terminal Hermes is drawn in.

Herdr.app treats a terminal as an agent only when its program reports state this
way; the Herdr TUI detects agents from the process/screen instead. Reports are
zero-width OSC sequences written to the tty, so they never touch the rendered
output.

Only a Herdr pane gets them: Herdr.app exports ``HC_TERMINAL_SOCKET`` and the TUI
exports ``HERDR_ENV=1``. Set ``HERMES_PROGRAM_STATUS=0`` to silence the reports,
or ``HERMES_PROGRAM_STATUS=1`` to send them to any terminal that wants them.

The agent runs in the local ``hermes serve`` process, which has no terminal, so
the reports are emitted by the client view that owns the tty
(``hermes_cli.gateway_chat_view``) at the points where it knows a turn began or
ended. A failed write never affects the turn.
"""

from __future__ import annotations

import os

_APP = "hermes"
_ENV = "HERMES_PROGRAM_STATUS"
_last: tuple[str, str | None] | None = None


def enabled() -> bool:
    """True when reports should be sent from this terminal."""
    value = os.environ.get(_ENV, "").strip().lower()
    if value in ("0", "false", "off", "no"):
        return False
    if value in ("1", "true", "on", "yes"):
        return True
    return bool(os.environ.get("HC_TERMINAL_SOCKET")) or os.environ.get("HERDR_ENV") == "1"


def report(state: str, kind: str | None = None) -> None:
    """Send one report; repeated identical states are skipped.

    Every report replaces the terminal's record whole, so ``app`` rides along
    each time. ``kind`` only means something with ``blocked``.
    """
    global _last
    try:
        if not enabled():
            return
        key = (state, kind)
        if key == _last:
            return
        pairs = f"state={state}:app={_APP}"
        if kind:
            pairs += f":kind={kind}"
        payload = f"\x1b]7501;{pairs}\x1b\\".encode("ascii", "ignore")
        fd = os.open("/dev/tty", os.O_WRONLY | os.O_NOCTTY)
        try:
            os.write(fd, payload)
        finally:
            os.close(fd)
        _last = key
    except Exception:
        pass


def idle() -> None:
    report("idle")


def working() -> None:
    report("working")


def done() -> None:
    report("done")


def blocked(kind: str | None = "permission") -> None:
    report("blocked", kind)
