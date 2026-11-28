"""Terminal-native desktop notifications and protocol reporters (OSC 9, OSC 777, OSC 7501, OSC 7577).

OSC 9 (``ESC ] 9 ; <body> BEL``): Ghostty, iTerm2, Kitty and WezTerm raise an OS notification;
others drop it. OSC 777 (``ESC ] 777 ; notify ; warp://cli-agent ; <json> BEL``): Warp's
structured CLI-agent protocol (tab status + notification mailbox). OSC 7501 reports live
turn/prompt status. OSC 7577 reports an authenticated resume command so supporting terminals
can bring Hermes back across terminal/server restarts.

Inside the running CLI, ``HermesCLI._ring_bell`` sends ``notification_sequence()`` through the
prompt_toolkit output on the app loop (a second writer on the tty would splice into an in-flight kitty
pet frame). ``write_tty`` is the no-app path: ``/dev/tty`` because ``patch_stdout``'s wrapper strips raw
escapes, falling back to ``sys.stdout`` when ``/dev/tty`` can't be opened (Windows, no controlling
terminal). Never raises.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from base64 import b64encode
from collections.abc import Mapping, Sequence
from contextlib import suppress

_C0_AND_DEL = re.compile(r"[\x00-\x1f\x7f]")
_WARP_PROTOCOL_VERSION = 1
# Last Warp release per channel that set WARP_CLI_AGENT_PROTOCOL_VERSION but could not render
# structured payloads (Warp's should-use-structured.sh). Bash compares lexicographically; so do we.
_WARP_LAST_BROKEN = {"stable": "v0.2026.03.25.08.24.stable_05", "preview": "v0.2026.03.25.08.24.preview_05"}
_PROGRAM_STATES = frozenset({"idle", "working", "done", "blocked", "error"})
_PROGRAM_KINDS = frozenset({"permission", "question", "auth"})
_PROGRAM_MESSAGE_LIMIT = 2048
# OSC 7501 discards a whole report whose decoded msg holds C0, DEL or C1 controls.
_PROGRAM_MSG_CONTROLS = re.compile(r"[\x00-\x1f\x7f-\x9f]+")

# OSC 7577 Program Resume Protocol (rev 0.5)
_OSC7577_TOKEN_RE = re.compile(r"[A-Za-z0-9_-]{22,64}")
_OSC7577_CONTROLS_RE = re.compile(r"[\x00-\x1f\x7f-\x9f]")
_OSC7577_WIN_DRIVE_RE = re.compile(r"[A-Za-z]:[\\/]")
_OSC7577_MAX_BYTES = 2048
_OSC7577_MAX_ARGV = 64


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


def program_status_sequence(state: str, *, kind: str = "", message: str = "") -> str:
    """Encode one OSC 7501 Program Status Protocol report for Hermes' root record."""
    if state not in _PROGRAM_STATES:
        raise ValueError(f"unsupported program status state: {state}")
    # The spec treats a kind on any other state, or an unknown kind, as absent; the TUI encoder agrees.
    if state != "blocked" or kind not in _PROGRAM_KINDS:
        kind = ""
    pairs = [f"state={state}", "app=hermes"]
    if kind:
        pairs.append(f"kind={kind}")
    clean_message = " ".join(_PROGRAM_MSG_CONTROLS.sub(" ", message).split())
    if clean_message:
        message_bytes = clean_message.encode("utf-8")[:_PROGRAM_MESSAGE_LIMIT]
        clean_message = message_bytes.decode("utf-8", "ignore")
        encoded = b64encode(clean_message.encode("utf-8")).decode("ascii")
        pairs.append(f"msg={encoded}")
    return f"\x1b]7501;{':'.join(pairs)}\x1b\\"


def program_status_clear_sequence() -> str:
    """OSC 7501 ``state=clear`` with no id: removes every record this program left on the terminal."""
    return "\x1b]7501;state=clear\x1b\\"


_claimed_osc7577_token: str | None = None


def claim_osc7577_token() -> None:
    """Move ``OSC7577_TOKEN`` out of ``os.environ`` into this module. Its holder can register a command the
    terminal runs later, and OSC 7577 asks a reporting program to keep it from its children: every
    subprocess inherits ``os.environ`` unless handed another env, so the CLI entry point claims it before
    starting any. The TUI launcher and ``restore_osc7577_token`` hand it on explicitly."""
    global _claimed_osc7577_token
    token = os.environ.pop("OSC7577_TOKEN", None)
    if token is not None:
        _claimed_osc7577_token = token


def restore_osc7577_token() -> None:
    """Put a claimed token back into ``os.environ`` for an in-place relaunch, which reports for the same pane."""
    if _claimed_osc7577_token is not None:
        os.environ["OSC7577_TOKEN"] = _claimed_osc7577_token


def osc7577_token(env: Mapping[str, str] | None = None) -> str | None:
    """Return the pane's validated ``OSC7577_TOKEN`` (from *env*, else ``os.environ`` or the claimed one)
    or ``None`` if absent/malformed."""
    if env is not None:
        token = env.get("OSC7577_TOKEN")
    else:
        token = os.environ.get("OSC7577_TOKEN") or _claimed_osc7577_token
    if not isinstance(token, str) or not _OSC7577_TOKEN_RE.fullmatch(token):
        return None
    return token


def _is_valid_osc7577_abs_path(path: str) -> bool:
    """True when *path* is absolute (POSIX root, drive letter plus separator, or UNC) without C0, DEL, or
    C1 control characters. Same rule as the TUI's ``isValidAbsPath``; ``ntpath.isabs`` varies by Python
    version (3.12 accepts a rooted path with no drive)."""
    if not isinstance(path, str) or not path or _OSC7577_CONTROLS_RE.search(path):
        return False
    return path.startswith(("/", "\\\\")) or _OSC7577_WIN_DRIVE_RE.match(path) is not None


def _is_valid_osc7577_argv0(arg: str) -> bool:
    """True when *arg* satisfies OSC 7577 ``argv[0]`` rules (absolute path or bare command)."""
    if not isinstance(arg, str) or not arg or _OSC7577_CONTROLS_RE.search(arg):
        return False
    return _is_valid_osc7577_abs_path(arg) or ("/" not in arg and "\\" not in arg and not arg.startswith("-"))


def _resolve_osc7577_params(token: str | None, pid: int | None, now: int | None) -> tuple[str, int, int]:
    """Validate common OSC 7577 header fields (`token`, `pid`, `time`)."""
    resolved_token = osc7577_token() if token is None else token
    if not isinstance(resolved_token, str) or not _OSC7577_TOKEN_RE.fullmatch(resolved_token):
        raise ValueError("missing or invalid OSC7577_TOKEN")
    resolved_pid = os.getpid() if pid is None else pid
    resolved_time = int(time.time()) if now is None else now
    if isinstance(resolved_pid, bool) or not isinstance(resolved_pid, int) or resolved_pid < 1:
        raise ValueError("invalid OSC 7577 pid")
    if isinstance(resolved_time, bool) or not isinstance(resolved_time, int) or resolved_time < 1:
        raise ValueError("invalid OSC 7577 time")
    return resolved_token, resolved_pid, resolved_time


def _finalize_osc7577(body: str, resolved_token: str) -> str:
    """Append plain SHA-256 ``mac`` trailer and enforce the 2048-byte sequence ceiling."""
    mac = hashlib.sha256(f"{body}:{resolved_token}".encode("ascii")).hexdigest()
    seq = f"\x1b]7577;{body}:mac={mac}\x1b\\"
    if len(seq.encode("utf-8")) > _OSC7577_MAX_BYTES:
        raise ValueError("OSC 7577 sequence exceeds 2048 bytes")
    return seq


def program_resume_sequence(
    argv: Sequence[str],
    *,
    token: str | None = None,
    pid: int | None = None,
    now: int | None = None,
    cwd: str | None = None,
) -> str:
    """Encode an authenticated OSC 7577 Program Resume report (``argv`` form)."""
    resolved_token, resolved_pid, resolved_time = _resolve_osc7577_params(token, pid, now)
    if isinstance(argv, (str, bytes)) or not isinstance(argv, Sequence):
        raise ValueError("argv must be a sequence of strings")
    if not 1 <= len(argv) <= _OSC7577_MAX_ARGV:
        raise ValueError("OSC 7577 argv must contain between 1 and 64 elements")
    for idx, arg in enumerate(argv):
        if not isinstance(arg, str) or _OSC7577_CONTROLS_RE.search(arg):
            raise ValueError("OSC 7577 argv element contains unsupported control characters")
        if idx == 0 and not _is_valid_osc7577_argv0(arg):
            raise ValueError("OSC 7577 argv[0] must be an absolute path or bare command name")
    argv_b64 = ",".join(b64encode(arg.encode("utf-8")).decode("ascii") for arg in argv)
    body = f"pid={resolved_pid}:time={resolved_time}:argv={argv_b64}"
    if cwd is not None:
        if not _is_valid_osc7577_abs_path(cwd):
            raise ValueError("OSC 7577 cwd must be an absolute path without control characters")
        cwd_b64 = b64encode(cwd.encode("utf-8")).decode("ascii")
        body = f"{body}:cwd={cwd_b64}"
    return _finalize_osc7577(body, resolved_token)


def program_resume_clear_sequence(
    *,
    token: str | None = None,
    pid: int | None = None,
    now: int | None = None,
) -> str:
    """Encode an authenticated OSC 7577 ``clear=1`` report removing the record of owner *pid* (default: this process)."""
    resolved_token, resolved_pid, resolved_time = _resolve_osc7577_params(token, pid, now)
    body = f"pid={resolved_pid}:time={resolved_time}:clear=1"
    return _finalize_osc7577(body, resolved_token)


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


def notification_sequence(context: str, *, prompt: bool, session_id: str = "", detail: str = "") -> str:
    """OSC 9 (plus Warp OSC 777 when supported) for a blocking prompt or turn end."""
    seq = osc9(f"Hermes: {context}")
    if warp_supported():
        event = "permission_request" if prompt else "stop"
        seq += warp_osc777(event, detail or context, session_id)
    return seq
