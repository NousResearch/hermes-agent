# -*- coding: utf-8 -*-
"""Pure agent-state derivation for the Hermes Windows tray icon.

No UI imports (pystray/ctypes/PIL) so it is testable on any host and on CI:
every reader takes explicit paths/clock. The tray script calls ``compute_state``
every poll; ``agent_state`` here maps Hermes' own signals onto the four dot
states.

Signals:
- active turn: ``session_turn_leases`` rows in state.db with ``expires_at > now``
  (authoritative; a dead backend's leases expire on their own). Falls back to
  log freshness when the DB is unreadable.
- needs input: ``tray-needs-input.json`` written by the bundled
  tray-needs-input plugin while a clarify question or approval prompt is open.
- last-turn error: the newest ``tui turn finished: ... status=`` line in
  gui.log (sticky error until a later completed turn).
"""
import json
import os
import re
import sqlite3
import time

ACTIVE_SECS = 90        # log-freshness fallback window
NI_STALE_SECS = 900     # ignore a pending marker older than this (backend died)
TAIL_BYTES = 131072     # read only the tail of logs/DB side files

FINISH_RE = re.compile(r"tui turn finished: .*?status=([a-z_]+)")


def hermes_home(environ=None):
    """Resolve the Hermes home the same precedence the CLI uses on Windows."""
    env = environ if environ is not None else os.environ
    h = env.get("HERMES_HOME")
    if h:
        return h
    lad = env.get("LOCALAPPDATA")
    if lad and os.path.isdir(os.path.join(lad, "hermes")):
        return os.path.join(lad, "hermes")
    return os.path.join(os.path.expanduser("~"), ".hermes")


def tail_text(path, nbytes=TAIL_BYTES):
    """Last ``nbytes`` of a text file as lines ('' when missing/unreadable)."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            f.seek(max(0, os.path.getsize(path) - nbytes))
            return f.read()
    except OSError:
        return ""


def turn_active(state_db, now=None):
    """True if any session has an unexpired turn lease. None when unreadable."""
    if now is None:
        now = time.time()
    try:
        con = sqlite3.connect("file:%s?mode=ro" % str(state_db).replace("\\", "/"),
                              uri=True, timeout=1)
        try:
            return con.execute(
                "SELECT COUNT(*) FROM session_turn_leases WHERE expires_at > ?",
                (now,),
            ).fetchone()[0] > 0
        finally:
            con.close()
    except sqlite3.Error:
        return None


def log_age(log_dir, now=None):
    """Seconds since the newest write to agent.log / gui.log (None if neither)."""
    if now is None:
        now = time.time()
    newest = None
    for name in ("agent.log", "gui.log"):
        try:
            a = now - os.path.getmtime(os.path.join(log_dir, name))
        except OSError:
            continue
        newest = a if newest is None else min(newest, a)
    return newest


def last_turn_status(gui_log):
    """status= of the most recent 'tui turn finished' line, or None."""
    status = None
    for line in tail_text(gui_log).splitlines():
        m = FINISH_RE.search(line)
        if m:
            status = m.group(1)
    return status


def needs_input(ni_file, now=None):
    """The plugin marker is pending and fresh (not a dead backend's leftover)."""
    if now is None:
        now = time.time()
    try:
        with open(ni_file, encoding="utf-8") as f:
            d = json.load(f)
        return bool(d.get("pending")) and (now - float(d.get("ts") or 0)) < NI_STALE_SECS
    except (OSError, ValueError, TypeError):
        return False


def compute_state(home, now=None):
    """The dot state for a Hermes home: needs_input|active|idle|error."""
    if now is None:
        now = time.time()
    gui_log = os.path.join(home, "logs", "gui.log")
    active = turn_active(os.path.join(home, "state.db"), now)
    if active is None:  # DB unreadable -> log-freshness fallback
        age = log_age(os.path.join(home, "logs"), now)
        active = age is not None and age < ACTIVE_SECS
    if active:
        return "needs_input" if needs_input(os.path.join(home, "tray-needs-input.json"), now) else "active"
    return "error" if last_turn_status(gui_log) == "error" else "idle"


def icon_visibility(pid, prev_pid, quit_flag_exists):
    """Tray-icon visibility policy keyed on the desktop process PID.

    The tray process is resident (single autostart entry, one pythonw); only
    the ICON tracks the desktop session, which replaces the old standalone
    watchdog. Returns ``(visible, clear_quit_flag)``.

    - PID changed to a desktop  -> new session: show, honour no stale flag
      from the previous session (fast relaunch / update restart gaps).
    - PID changed to None       -> session over: hide, clear the flag so the
      next desktop launch shows again.
    - no change, desktop up     -> hidden only while the user's
      "hide icon until next session" flag stands.
    - no change, no desktop     -> stay hidden.
    """
    if pid != prev_pid:
        return (pid is not None), True
    if pid is None:
        return False, False
    return (not quit_flag_exists), False
