"""Terminal-native desktop notifications: OSC 9 and Warp's OSC 777 CLI-agent protocol.

Both emitters ride on the existing ``display.bell_on_prompt`` /
``display.bell_on_complete`` flags (see ``cli._ring_bell``) — no extra config.

- **OSC 9** (``ESC ] 9 ; <body> BEL``): Ghostty, iTerm2, Kitty and WezTerm
  raise an OS notification; terminals that don't know the sequence drop it.
- **OSC 777** (``ESC ] 777 ; notify ; warp://cli-agent ; <json> BEL``): Warp's
  structured CLI-agent protocol (tab status + notification mailbox). Only sent
  when Warp advertises support and the build is newer than the last release
  that set the protocol var without being able to render the payload.

Sequences are written to ``/dev/tty`` because prompt_toolkit's stdout wrapper
can buffer or strip raw escapes; when ``/dev/tty`` can't be opened (Windows,
no controlling terminal) they fall back to ``sys.stdout``. Never raises.
"""

from __future__ import annotations

import json
import os
import re
import sys

_C0_AND_DEL = re.compile(r"[\x00-\x1f\x7f]")
_WARP_PROTOCOL_VERSION = 1
# Last Warp release per channel that set WARP_CLI_AGENT_PROTOCOL_VERSION but
# could not render structured payloads (Warp's reference agent plugin,
# should-use-structured.sh). Bash compares these lexicographically; so do we.
_WARP_LAST_BROKEN = {
    "stable": "v0.2026.03.25.08.24.stable_05",
    "preview": "v0.2026.03.25.08.24.preview_05",
}


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


def osc9(body: str) -> str:
    """OSC 9 sequence with C0 controls and DEL stripped from the body."""
    return f"\x1b]9;{_C0_AND_DEL.sub('', body)}\x07"


def warp_supported(env=None) -> bool:
    """True when running in a Warp build that can render OSC 777 agent payloads."""
    env = os.environ if env is None else env
    if env.get("TERM_PROGRAM") != "WarpTerminal" or not env.get("WARP_CLI_AGENT_PROTOCOL_VERSION"):
        return False
    client = env.get("WARP_CLIENT_VERSION", "")
    if not client:
        return False
    for channel, last_broken in _WARP_LAST_BROKEN.items():
        if channel in client and client <= last_broken:
            return False
    return True


def warp_osc777(event: str, detail: str, session_id: str = "") -> str:
    """OSC 777 ``warp://cli-agent`` notification; ``event`` is ``stop`` or ``permission_request``.

    Payload mirrors the reference plugin's build-payload.sh: common fields plus
    ``summary`` (permission_request) or ``response`` (stop), truncated to 200.
    """
    try:
        advertised = int(os.environ.get("WARP_CLI_AGENT_PROTOCOL_VERSION", "1"))
    except ValueError:
        advertised = 1
    cwd = os.getcwd()
    payload = {
        "v": min(advertised, _WARP_PROTOCOL_VERSION),
        "agent": "hermes",
        "event": event,
        "session_id": session_id,
        "cwd": cwd,
        "project": os.path.basename(cwd),
    }
    payload["summary" if event == "permission_request" else "response"] = detail[:200]
    return f"\x1b]777;notify;warp://cli-agent;{json.dumps(payload, separators=(',', ':'))}\x07"


def notify(context: str, *, prompt: bool, session_id: str = "", detail: str = "") -> None:
    """Emit OSC 9 (plus Warp OSC 777 when supported) for a blocking prompt or turn end."""
    seq = osc9(f"Hermes: {context}")
    if warp_supported():
        event = "permission_request" if prompt else "stop"
        seq += warp_osc777(event, detail or context, session_id)
    _write_tty(seq)
