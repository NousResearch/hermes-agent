"""Built-in per-turn footer line for the interactive CLI.

Renders the closing line::

    HH:MM:SS | <turn mm:ss> | m: <real-model> | t: turn: in=X out=Y session: in=A out=B

at the end of every turn, replacing the previous skill/plugin-based approach
(``footer.py`` + ``generate_footer`` tool).  The CLI owns the rendering; the
agent no longer calls ``footer.py`` itself.

Config: ``display.turn_footer.enabled`` (default true for interactive), distinct
from ``display.runtime_footer``.
"""

from __future__ import annotations

import json
import os
import socket
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

BERLIN = ZoneInfo("Europe/Berlin")
VPS_HOST = "gk@72.60.32.82"
VPS_9R_DB = "file:/home/gk/.9router/db/data.sqlite?mode=ro"
LOCAL_9R_DB = "/home/gk/.9router/db/data.sqlite"
VPS_HOSTNAME = "srv979961"


def _home_dir() -> str:
    h = os.environ.get("HERMES_HOME", "")
    if h and os.path.exists(os.path.join(os.path.expanduser(h), "state.db")):
        return os.path.expanduser(h)
    return os.path.expanduser("~/.hermes")


HOME_DIR = _home_dir()
STATE_FILE = os.path.join(HOME_DIR, "state", "footer-tok.json")
SESSION_DB = os.path.join(HOME_DIR, "state.db")


def _resolve_clock(sys_now, auth_epoch, tol_s, tz):
    if not auth_epoch:
        return sys_now, sys_now.strftime("%H:%M:%S"), ""
    off = auth_epoch - int(sys_now.timestamp())
    if abs(off) <= tol_s:
        return sys_now, sys_now.strftime("%H:%M:%S"), ""
    corrected = datetime.fromtimestamp(auth_epoch, tz)
    sign = "+" if off > 0 else "-"
    a = abs(off)
    note = (f"  ⚠host-clock {sign}{a // 3600:d}:{(a % 3600) // 60:02d}:{a % 60:02d}"
            f" (host clock {sign}{a}s off; time shown from VPS)")
    return corrected, corrected.strftime("%H:%M:%S"), note


def _fmt(n):
    if n >= 1000:
        v = int(n / 100) / 10
        return f"{v:.1f}k"
    return str(n)


def _resolve_session_id(conn, session_id=None):
    if session_id:
        try:
            r = conn.execute("SELECT id FROM sessions WHERE id=?", (session_id,)).fetchone()
            if r and r[0]:
                return r[0]
        except Exception:
            pass
    try:
        r = conn.execute("SELECT id FROM sessions ORDER BY last_activity_at DESC LIMIT 1").fetchone()
        if r and r[0]:
            return r[0]
    except Exception:
        pass
    return None


def _fmt_dur(secs):
    if secs is None:
        return ""
    secs = max(0, int(secs))
    if secs >= 3600:
        return f"{secs // 3600}:{(secs % 3600) // 60:02d}:{secs % 60:02d}"
    return f"{secs // 60:02d}:{secs % 60:02d}"


def is_enabled(config=None) -> bool:
    """Resolve ``display.turn_footer.enabled`` (default True for interactive)."""
    try:
        if config is None:
            from hermes_cli.config import load_config
            config = load_config() or {}
    except Exception:
        config = config or {}
    try:
        display = (config or {}).get("display") or {}
        tf = display.get("turn_footer")
        if isinstance(tf, dict):
            if "enabled" in tf:
                return bool(tf.get("enabled"))
        # fall through to default True
    except Exception:
        pass
    return True


def compute_turn_footer_line(config=None, session_id=None) -> str:
    """Compute the footer line string, or ``""`` if disabled / unavailable."""
    if not is_enabled(config):
        return ""

    sys_now = datetime.now(BERLIN)
    tol_s = int(os.environ.get("FOOTER_CLOCK_TOL_S", "30"))

    # 1. Real model from 9r usageHistory (newest request with promptTokens > 1000).
    model = None
    auth_epoch = None
    if socket.gethostname() == VPS_HOSTNAME:
        auth_epoch = int(time.time())
        try:
            c = sqlite3.connect(f"file:{LOCAL_9R_DB}?mode=ro", uri=True)
            r = c.execute(
                "SELECT model FROM usageHistory WHERE promptTokens > 1000 ORDER BY timestamp DESC LIMIT 1"
            ).fetchone()
            if r and r[0]:
                model = r[0]
        except Exception:
            pass
    else:
        try:
            r = subprocess.run(
                ["ssh", "-o", "ConnectTimeout=5", VPS_HOST, "python3 -"],
                input=(
                    "import sqlite3, time\n"
                    f'c=sqlite3.connect("{VPS_9R_DB}",uri=True)\n'
                    'r=c.execute("SELECT model FROM usageHistory '
                    'WHERE promptTokens > 1000 ORDER BY timestamp DESC LIMIT 1").fetchone()\n'
                    'print("MODEL\\t" + (r[0] if r and r[0] else "?"))\n'
                    'print("EPOCH\\t" + str(int(time.time())))\n'
                ),
                capture_output=True, text=True, timeout=10, check=False,
            )
            if r.returncode == 0:
                for line in r.stdout.splitlines():
                    key, _sep, val = line.partition("\t")
                    if key == "MODEL" and val and val != "?":
                        model = val
                    elif key == "EPOCH":
                        try:
                            auth_epoch = int(val)
                        except ValueError:
                            pass
        except Exception:
            pass

    _, clock, drift_note = _resolve_clock(sys_now, auth_epoch, tol_s, BERLIN)

    # Fallback: combo name from session DB.
    if not model:
        try:
            c = sqlite3.connect(f"file:{SESSION_DB}?mode=ro", uri=True)
            sid = _resolve_session_id(c, session_id)
            if sid:
                r = c.execute(
                    "SELECT model FROM session_model_usage WHERE session_id=?", (sid,)
                ).fetchone()
                if r and r[0]:
                    model = r[0]
        except Exception:
            pass
    model = model or "?"

    # Cumulative session token counts.
    in_tok = out_tok = 0
    sess_id = None
    try:
        c = sqlite3.connect(f"file:{SESSION_DB}?mode=ro", uri=True)
        sess_id = _resolve_session_id(c, session_id)
        if sess_id:
            r = c.execute(
                "SELECT input_tokens, output_tokens FROM session_model_usage WHERE session_id=?",
                (sess_id,),
            ).fetchone()
            if r:
                in_tok = r[0] or 0
                out_tok = r[1] or 0
    except Exception:
        pass

    # Per-turn delta via state file.
    prev_in = prev_out = 0
    prev_sid = None
    try:
        with open(STATE_FILE) as f:
            d = json.load(f)
            prev_in = d.get("in", 0)
            prev_out = d.get("out", 0)
            prev_sid = d.get("sid")
    except Exception:
        pass
    if prev_sid is not None and sess_id is not None and prev_sid != sess_id:
        prev_in, prev_out = in_tok, out_tok
    turn_in = in_tok - prev_in
    turn_out = out_tok - prev_out
    if turn_in < 0 or turn_out < 0:
        turn_in = 0
        turn_out = 0
    try:
        os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
        with open(STATE_FILE, "w") as f:
            json.dump({"in": in_tok, "out": out_tok, "sid": sess_id}, f)
    except Exception:
        pass

    # Turn duration: last user message in session -> now (host clock).
    turn_secs = None
    try:
        if sess_id:
            c = sqlite3.connect(f"file:{SESSION_DB}?mode=ro", uri=True)
            u = c.execute(
                "SELECT timestamp FROM messages WHERE session_id=? AND role='user' "
                "ORDER BY rowid DESC LIMIT 1",
                (sess_id,),
            ).fetchone()
            if u and u[0]:
                turn_secs = int(sys_now.timestamp() - u[0])
    except Exception:
        turn_secs = None

    # Assemble.
    tok_parts = []
    if turn_in > 0 or turn_out > 0:
        tok_parts.append(f"turn: in={_fmt(turn_in)} out={_fmt(turn_out)}")
    tok_parts.append(f"session: in={_fmt(in_tok)} out={_fmt(out_tok)}")

    parts = [clock]
    d = _fmt_dur(turn_secs)
    if d:
        parts.append(d)
    parts.append(f"m: {model}")
    parts.append(f"t: {' '.join(tok_parts)}")
    line = " | ".join(parts)
    if drift_note:
        line += drift_note
    return line


# Backwards/alternate names for callers/tests.
compute_turn_footer = compute_turn_footer_line
render_turn_footer = compute_turn_footer_line
turn_footer_line = compute_turn_footer_line
get_turn_footer = compute_turn_footer_line
format_turn_footer = compute_turn_footer_line


def render_turn_footer_for_cli(cli=None, config=None, session_id=None) -> str:
    """CLI-facing helper: respects ``cli.config`` and prints nothing if disabled."""
    if config is None and cli is not None:
        config = getattr(cli, "config", None)
    if session_id is None and cli is not None:
        session_id = getattr(cli, "session_id", None)
    return compute_turn_footer_line(config=config, session_id=session_id)


def print_turn_footer(cli=None, config=None, session_id=None) -> str:
    """Compute and print the footer line (if enabled) to stdout."""
    line = render_turn_footer_for_cli(cli=cli, config=config, session_id=session_id)
    if line:
        try:
            print(line)
        except Exception:
            pass
    return line
