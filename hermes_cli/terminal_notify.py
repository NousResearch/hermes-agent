"""Terminal-native desktop notifications: OSC 9 and Warp's OSC 777 CLI-agent protocol.

OSC 9 (``ESC ] 9 ; <body> BEL``): Foot, Ghostty, iTerm2, Kitty and WezTerm raise an OS notification;
others drop it. OSC 777 (``ESC ] 777 ; notify ; warp://cli-agent ; <json> BEL``): Warp's
structured CLI-agent protocol (tab status + notification mailbox).

Sequences are written to ``/dev/tty`` because prompt_toolkit's stdout wrapper can buffer or strip
raw escapes; when ``/dev/tty`` can't be opened (Windows, no controlling terminal) they fall back to
``sys.stdout``. Never raises.
"""

from __future__ import annotations

import json
import os
import re
import sys

_C0_AND_DEL = re.compile(r"[\x00-\x1f\x7f]")
_WARP_PROTOCOL_VERSION = 1
# Last Warp release per channel that set WARP_CLI_AGENT_PROTOCOL_VERSION but could not render
# structured payloads (Warp's should-use-structured.sh). Bash compares lexicographically; so do we.
_WARP_LAST_BROKEN = {"stable": "v0.2026.03.25.08.24.stable_05", "preview": "v0.2026.03.25.08.24.preview_05"}


def _write_tty(seq: str) -> None:
    """Write raw escapes to /dev/tty, falling back to sys.stdout. Never raises."""
    try:
        with open("/dev/tty", "w", encoding="utf-8") as tty:
            tty.write(seq)
        return
    except OSError:
        pass
    try:
        sys.stdout.write(seq)
        sys.stdout.flush()
    except Exception:
        pass


_BODY_LIMIT = 200  # matches the Warp payload's existing detail[:200] cap so both surfaces
                   # carry the same text


def prompt_body(kind: str, detail: str = "") -> str:
    """Notification body for a blocking prompt: "<kind> — <detail>", detail collapsed to a
    single line (whitespace runs, including newlines, become one space) and capped at
    _BODY_LIMIT codepoints. Falls back to just <kind> when detail is empty/whitespace."""
    collapsed = " ".join(detail.split())
    if not collapsed:
        return kind
    return f"{kind} — {collapsed[:_BODY_LIMIT]}"


def osc9(body: str) -> str:
    """OSC 9 sequence with C0 controls and DEL stripped from the body."""
    return f"\x1b]9;{_C0_AND_DEL.sub('', body)}\x07"


def warp_supported(env=None) -> bool:
    """True when running in a Warp build that can render OSC 777 agent payloads."""
    env = os.environ if env is None else env
    client = env.get("WARP_CLIENT_VERSION", "")
    if env.get("TERM_PROGRAM") != "WarpTerminal" or not env.get("WARP_CLI_AGENT_PROTOCOL_VERSION") or not client:
        return False
    return not any(channel in client and client <= last_broken for channel, last_broken in _WARP_LAST_BROKEN.items())


def warp_osc777(event: str, detail: str, session_id: str = "") -> str:
    """OSC 777 ``warp://cli-agent`` notification; ``event`` is ``stop`` or ``permission_request``."""
    try:
        advertised = int(os.environ.get("WARP_CLI_AGENT_PROTOCOL_VERSION", "1"))
    except ValueError:
        advertised = 1
    cwd = os.getcwd()
    payload = {"v": min(advertised, _WARP_PROTOCOL_VERSION), "agent": "hermes", "event": event,
               "session_id": session_id, "cwd": cwd, "project": os.path.basename(cwd)}
    payload["summary" if event == "permission_request" else "response"] = detail[:200]
    return f"\x1b]777;notify;warp://cli-agent;{json.dumps(payload, separators=(',', ':'))}\x07"


# Terminals that raise an OS notification for OSC 9. Verified against terminal-support
# references: iTerm2, Ghostty, WezTerm, Warp (and kitty/foot, which leave TERM_PROGRAM unset and
# are matched on TERM). xterm.js-based terminals (VS Code, Cursor) are deliberately absent —
# xterm.js does not implement OSC 9 notifications.
_OSC9_TERM_PROGRAMS = {"iterm.app", "ghostty", "wezterm", "warpterminal"}
_OSC9_TERMS = ("kitty", "foot")


def osc9_capable(env=None) -> bool:
    """True when the terminal described by `env` raises an OS notification for OSC 9."""
    env = os.environ if env is None else env
    if env.get("TMUX") or env.get("STY"):
        # tmux/screen drop unknown OSC unless passthrough is configured, so the sequence never
        # reaches the terminal — treat the session as incapable and let the OS notifier cover it.
        return False
    if (env.get("TERM_PROGRAM") or "").lower() in _OSC9_TERM_PROGRAMS:
        return True
    term = (env.get("TERM") or "").lower()
    return any(name in term for name in _OSC9_TERMS)


def notify(context: str, *, prompt: bool, session_id: str = "", detail: str = "") -> None:
    """Emit OSC 9 (plus Warp OSC 777 when supported) for a blocking prompt or turn end."""
    seq = osc9(f"Hermes: {context}")
    if warp_supported():
        event = "permission_request" if prompt else "stop"
        seq += warp_osc777(event, detail or context, session_id)
    _write_tty(seq)
    if not osc9_capable():
        try:
            from hermes_cli import os_notify

            os_notify.notify("Hermes", context)
        except Exception:
            pass
