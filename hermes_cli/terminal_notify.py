"""Terminal-native desktop notifications: OSC 9, Ghostty's OSC 777, and Warp's OSC 777 CLI-agent protocol.

OSC 9 (``ESC ] 9 ; <body> BEL``): Ghostty, iTerm2, Kitty and WezTerm raise an OS notification;
others drop it. OSC 777 ``notify;<title>;<body>``: Ghostty's rxvt extension, the only notification
escape there that carries a TITLE — a bare OSC 9 makes Ghostty notify without one, so Ghostty gets
777 and everyone else keeps OSC 9. Warp's OSC 777 is a different payload on the same code
(``warp://cli-agent`` structured JSON: tab status + notification mailbox) and is sent IN ADDITION.

Inside the running CLI, ``HermesCLI._ring_bell`` sends ``notification_sequence()`` through the
prompt_toolkit output on the app loop (a second writer on the tty would splice into an in-flight kitty
pet frame). ``write_tty`` is the no-app path: ``/dev/tty`` because ``patch_stdout``'s wrapper strips raw
escapes, falling back to ``sys.stdout`` when ``/dev/tty`` can't be opened (Windows, no controlling
terminal). Never raises.
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


def write_tty(seq: str) -> None:
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


def osc9(body: str) -> str:
    """OSC 9 sequence with C0 controls and DEL stripped from the body."""
    return f"\x1b]9;{_C0_AND_DEL.sub('', body)}\x07"


def ghostty_supported(env=None) -> bool:
    """True inside Ghostty, whose OSC 9 has no title field (OSC 777 does)."""
    env = os.environ if env is None else env
    return (
        (env.get("TERM_PROGRAM") or "").strip().lower() == "ghostty"
        or (env.get("TERM") or "").strip().lower() == "xterm-ghostty"
    )


def _osc777_field(text: str) -> str:
    """One OSC 777 argument: controls stripped, the field separator neutralized.

    Ghostty splits ``notify;<title>;<body>`` on the first semicolons, so an unescaped ``;`` in
    either field shifts the body into the title instead of truncating it.
    """
    return _C0_AND_DEL.sub("", text).replace(";", ",")


def ghostty_osc777(title: str, body: str) -> str:
    """Ghostty's rxvt notify extension (``OSC 777 ; notify ; <title> ; <body>``), title required."""
    return f"\x1b]777;notify;{_osc777_field(title)};{_osc777_field(body)}\x07"


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


def notification_sequence(context: str, *, prompt: bool, session_id: str = "", detail: str = "",
                          env=None) -> str:
    """OSC 777 (Ghostty) / OSC 9 (everyone else) for a blocking prompt or turn end.

    Ghostty raises the titled OSC 777 notification. Sending OSC 9 there too would double-notify
    (Ghostty honours both), so the two are mutually exclusive; Warp's Cli-agent 777 is additive.
    """
    env = os.environ if env is None else env
    body = f"Hermes: {context}"
    seq = ghostty_osc777("Hermes", body) if ghostty_supported(env) else osc9(body)
    if warp_supported(env):
        event = "permission_request" if prompt else "stop"
        seq += warp_osc777(event, detail or context, session_id)
    return seq
