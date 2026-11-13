"""OSC 7501 (Program Status Protocol) emitter for terminal agent monitors.

Reports the CLI's live state (idle / working / blocked / done) to any attached
terminal multiplexer or agent monitor (Herdr, etc.) through the OSC 7501
escape sequence, so the agent shows up in agent lists without screen scraping.

The protocol: ``ESC ] 7501 ; state=<state>:app=<app> BEL`` — where ``<state>``
is one of ``idle`` / ``working`` / ``blocked`` / ``done``.

Writes go straight to the controlling terminal (``/dev/tty``), NOT
``sys.stdout``: prompt_toolkit's ``patch_stdout`` StdoutProxy sanitizes
unknown OSC sequences out of the stream, so a stdout write never reaches
the pty. With no ``/dev/tty`` (pipes, files, detached CI, API servers)
there is nowhere to report and the emitter stays silent. It must never
raise into the CLI loop.

Hermes reports ``app=hermes``. Reference: Program Status Protocol (OSC 7501).
"""

from __future__ import annotations

import os
import threading

APP_ID = "hermes"

_LOCK = threading.Lock()
_LAST_STATE: str | None = None

# Valid protocol states, per the OSC 7501 spec.
_VALID_STATES = frozenset({"idle", "working", "blocked", "done"})


def _emit(state: str) -> None:
    """Write the OSC 7501 sequence for ``state`` to the controlling terminal."""
    global _LAST_STATE
    if state not in _VALID_STATES:
        return
    try:
        with _LOCK:
            if _LAST_STATE == state:
                return  # avoid duplicate sequences between turns
            _LAST_STATE = state
            payload = f"\x1b]7501;state={state}:app={APP_ID}\x07"
            # Direct fd write: patch_stdout's StdoutProxy strips OSC 7501 from
            # sys.stdout before it reaches the pty, so bypass it for /dev/tty —
            # the controlling terminal of the pty the monitor reads. BEL
            # terminator (ST also valid; BEL is safest across emulators).
            fd = os.open("/dev/tty", os.O_WRONLY)
            try:
                os.write(fd, payload.encode("utf-8"))
            finally:
                os.close(fd)
    except OSError:
        pass  # no controlling terminal: nowhere to report, stay silent
    except Exception:
        pass  # never let status reporting break the CLI


def report_working() -> None:
    """A turn started: the agent is working."""
    _emit("working")


def report_blocked() -> None:
    """The agent is waiting on the user (approval/question/clarification)."""
    _emit("blocked")


def report_idle() -> None:
    """The turn finished cleanly and the prompt is open."""
    _emit("idle")


def report_done() -> None:
    """The agent finished its task and has nothing pending."""
    _emit("done")