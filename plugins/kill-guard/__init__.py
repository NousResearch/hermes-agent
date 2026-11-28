"""kill-guard plugin: veto ``pkill`` / ``killall`` commands that would match far more than intended.

macOS/BSD ``pkill`` stops reading options at the first pattern, so in
``pkill -f "serve.py" -u me --signal TERM`` the tokens ``-u``, ``me``, ``--signal`` and ``TERM`` become four more
patterns; with ``-f``, a user name that appears in every command line (``/Users/me/...``) matches, and the
SIGTERM lands on the gateway, dev servers and every cron job. This hook blocks, before it runs, any terminal
``pkill``/``killall`` segment where an option follows the pattern, several patterns are given, a pattern is the user
name / home path / ``/Users``, or ``pkill -u <user>`` has no pattern at all. It fails open on any internal error.

Opt-in (list it under ``plugins.enabled``); ``KILL_GUARD_DISABLE=1`` bypasses it. Blocks are appended to
``<HERMES_HOME>/logs/kill_guard.log``.
"""
from __future__ import annotations

import getpass
import logging
import os
import re
import shlex
from datetime import datetime
from pathlib import Path

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)
ARG_OPTS = set("FGgPUutsJ")  # BSD pkill options that take a value
SPLIT = re.compile(r"\|\||&&|[;|&\n()`]|\$\(")
MESSAGE = (
    "kill-guard: this pkill/killall would match far more than you mean. macOS (BSD) pkill stops reading options "
    "at the pattern, so anything after it (-u, a user name, --signal, TERM) becomes another pattern, and with -f "
    "the user name matches every process of the user. Put every option BEFORE one pattern, e.g. "
    "pkill -TERM -u <user> -f 'my/server.py', and never use the user name or home path as the pattern. "
    "Better: launchctl kickstart -k gui/$(id -u)/<label> for a launchd service."
)


def _home() -> Path:
    return Path(get_hermes_home())


def _bad_names():
    u = getpass.getuser()
    h = str(Path.home())
    return {u, h, h + "/", "/Users", "/Users/", "-u", "TERM", "KILL", "--signal"}


def _segment_bad(toks):
    if not toks:
        return None
    exe = os.path.basename(toks[0])
    if exe == "sudo" and len(toks) > 1:
        toks = toks[1:]
        exe = os.path.basename(toks[0])
    if exe not in ("pkill", "killall"):
        return None
    pats, opts, i = [], [], 1
    while i < len(toks):
        t = toks[i]
        if pats and t.startswith("-") and t != "-":
            return f"option {t!r} after pattern {pats[0]!r}"
        if not pats and t.startswith("-") and len(t) > 1:
            opts.append(t)
            if exe == "pkill" and len(t) == 2 and t[1] in ARG_OPTS:
                i += 2
                continue
            i += 1
            continue
        pats.append(t)
        i += 1
    if exe == "pkill" and len(pats) > 1:
        return f"several patterns {pats!r} (BSD pkill ORs them)"
    bad = _bad_names()
    for p in pats:
        if p.strip() in bad or p.strip().rstrip("/") == str(Path.home()):
            return f"pattern {p!r} matches every process of the user"
    if exe == "pkill" and not pats and any(o.startswith("-u") or o.startswith("-U") for o in opts):
        return "pkill -u with no pattern kills every process of the user"
    return None


def check_command(cmd: str):
    """Return a reason string if the command would run a dangerous pkill/killall, else None."""
    if not cmd or not re.search(r"\b(pkill|killall)\b", cmd):
        return None
    for seg in SPLIT.split(cmd):
        try:
            toks = shlex.split(seg, posix=True)
        except ValueError:
            toks = seg.split()
        while toks and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", toks[0]):
            toks = toks[1:]
        why = _segment_bad(toks)
        if why:
            return why
    return None


def _pre_tool_call(tool_name=None, args=None, session_id="", **_kw):
    if os.environ.get("KILL_GUARD_DISABLE", "").lower() in ("1", "true", "yes", "on"):
        return None
    try:
        args = args if isinstance(args, dict) else {}
        calls = [(tool_name, args)]
        if tool_name == "tool_call":
            calls = [(c.get("name"), c.get("arguments") or {}) for c in args.get("calls", []) if isinstance(c, dict)]
        for n, a in calls:
            if not isinstance(a, dict) or str(n).split("__")[-1] != "terminal":
                continue
            why = check_command(str(a.get("command") or ""))
            if why:
                try:
                    log = _home() / "logs" / "kill_guard.log"
                    log.parent.mkdir(parents=True, exist_ok=True)
                    with log.open("a") as f:
                        f.write(f"{datetime.now().isoformat(timespec='seconds')} BLOCK session={session_id} {why}\n")
                except OSError:
                    logger.debug("kill-guard: could not write block log", exc_info=True)
                return {"action": "block", "message": f"{MESSAGE} ({why})"}
        return None
    except Exception:
        logger.warning("kill-guard failed open", exc_info=True)
        return None


def register(ctx) -> None:
    ctx.register_hook("pre_tool_call", _pre_tool_call)
