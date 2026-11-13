"""OSC 7501 (Program Status Protocol) emitter for terminal agent monitors.

Reports the CLI's live state (idle / working / blocked) to any attached
terminal multiplexer or agent monitor (Herdr, etc.) through the OSC 7501
escape sequence, so the agent shows up in agent lists without screen scraping.

The protocol: ``ESC ] 7501 ; state=<state>:app=<app> ESC \\`` — where
``<state>`` is one of ``idle`` / ``working`` / ``blocked`` / ``done``.
Emitters must be silent on non-TTY stdout (pipes, files, API servers) and
must never raise into the CLI loop.

Hermes reports ``app=hermes``. Reference:
https://gist.github.com/agent-status/osc-7501 (Program Status Protocol).
"""

from __future__ import annotations

import sys
import threading

APP_ID = "hermes"

_LOCK = threading.Lock()
_LAST_STATE: str | None = None

# Valid protocol states, per the OSC 7501 spec.
_VALID_STATES = frozenset({"idle", "working", "blocked", "done"})


def _emit(state: str) -> None:
    """Write the OSC 7501 sequence for ``state`` to the real stdout TTY."""
    global _LAST_STATE
    if state not in _VALID_STATES:
        return
    try:
        stdout = sys.stdout
        if not hasattr(stdout, "isatty") or not stdout.isatty():
            return  # piped/file/API-server stdout: monitors read the pty, not us
        with _LOCK:
            if _LAST_STATE == state:
                return  # avoid duplicate sequences between turns
            _LAST_STATE = state
            # OSC: ESC ] 7501 ; payload BEL (ST also valid; BEL is safest)
            stdout.write(f"\x1b]7501;state={state}:app={APP_ID}\x07")
            stdout.flush()
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