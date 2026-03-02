"""
hermes_cli/stats.py — Session analytics for the /stats slash command.

Reads from ~/.hermes/state.db using the existing SessionDB schema and
renders a compact summary. Zero new dependencies.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

from hermes_cli.banner import cprint, _DIM, _RST


def _db_path() -> Path:
    import os
    return Path(os.getenv("HERMES_HOME", Path.home() / ".hermes")) / "state.db"


def _connect(db: Path) -> sqlite3.Connection:
    con = sqlite3.connect(str(db), check_same_thread=False, timeout=5.0)
    con.row_factory = sqlite3.Row
    return con


def _gather(con: sqlite3.Connection) -> dict:
    s = {}

    row = con.execute(
        "SELECT COUNT(*) as n, "
        "SUM(message_count) as msgs, "
        "SUM(tool_call_count) as tools, "
        "SUM(input_tokens) as itok, "
        "SUM(output_tokens) as otok, "
        "MIN(started_at) as first, "
        "MAX(started_at) as last "
        "FROM sessions"
    ).fetchone()

    s["total_sessions"]   = row["n"] or 0
    s["total_messages"]   = row["msgs"] or 0
    s["total_tool_calls"] = row["tools"] or 0
    s["input_tokens"]     = row["itok"] or 0
    s["output_tokens"]    = row["otok"] or 0
    s["first_session"]    = row["first"]
    s["last_session"]     = row["last"]

    if s["total_sessions"] == 0:
        return s

    row2 = con.execute(
        "SELECT COUNT(*) as n FROM messages WHERE role = 'user'"
    ).fetchone()
    s["total_turns"] = row2["n"] or 0
    s["avg_turns"] = round(s["total_turns"] / s["total_sessions"], 1)

    rows = con.execute(
        "SELECT source, COUNT(*) as n FROM sessions GROUP BY source ORDER BY n DESC"
    ).fetchall()
    s["by_source"] = [(r["source"] or "unknown", r["n"]) for r in rows]

    rows = con.execute(
        "SELECT model, COUNT(*) as n FROM sessions "
        "WHERE model IS NOT NULL GROUP BY model ORDER BY n DESC LIMIT 5"
    ).fetchall()
    s["by_model"] = [(r["model"], r["n"]) for r in rows]

    rows = con.execute(
        "SELECT tool_name, COUNT(*) as n FROM messages "
        "WHERE tool_name IS NOT NULL "
        "GROUP BY tool_name ORDER BY n DESC LIMIT 10"
    ).fetchall()
    s["top_tools"] = [(r["tool_name"], r["n"]) for r in rows]

    rows = con.execute("SELECT started_at FROM sessions").fetchall()
    dow_map = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
    dow_counts: dict = {v: 0 for v in dow_map.values()}
    for r in rows:
        if r["started_at"]:
            try:
                dt = datetime.fromtimestamp(float(r["started_at"]))
                dow_counts[dow_map[dt.weekday()]] += 1
            except Exception:
                pass
    s["by_dow"] = dow_counts

    return s


_CYAN = "\033[96m"
_BOLD = "\033[1m"
_GRN  = "\033[92m"
_YLW  = "\033[93m"
_RST2 = "\033[0m"
_W    = 52


def _fmt_ts(ts) -> str:
    if not ts:
        return "—"
    try:
        return datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d")
    except Exception:
        return str(ts)


def _fmt_tokens(n: int) -> str:
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}K"
    return str(n)


def _bar(count: int, max_count: int, width: int = 16) -> str:
    if max_count == 0:
        return "░" * width
    filled = round(count / max_count * width)
    return "█" * filled + "░" * (width - filled)


def print_stats() -> None:
    """Render session statistics to stdout. Called by /stats handler."""

    db = _db_path()
    if not db.exists():
        cprint(f"\n{_YLW}No session database found.{_RST2}")
        cprint(f"{_DIM}Start a conversation first to generate stats.{_RST}\n")
        return

    try:
        con = _connect(db)
        data = _gather(con)
        con.close()
    except sqlite3.Error as e:
        cprint(f"\n{_YLW}Could not read session database: {e}{_RST2}\n")
        return

    if data["total_sessions"] == 0:
        cprint(f"\n{_DIM}No sessions recorded yet.{_RST}\n")
        return

    total_tok = data["input_tokens"] + data["output_tokens"]
    first     = _fmt_ts(data["first_session"])
    last      = _fmt_ts(data["last_session"])

    print(f"\n{_CYAN}{_BOLD}{'─'*_W}{_RST2}")
    print(f"{_CYAN}{_BOLD}  ⚕  Hermes Stats{_RST2}")
    print(f"{_CYAN}{'─'*_W}{_RST2}")
    print(f"  {_BOLD}Sessions  {_RST2} {_CYAN}{data['total_sessions']}{_RST2}")
    print(f"  {_BOLD}Turns     {_RST2} {_CYAN}{data['total_turns']}{_RST2}  "
          f"{_DIM}(avg {data['avg_turns']}/session){_RST}")
    print(f"  {_BOLD}Tool calls{_RST2} {_CYAN}{data['total_tool_calls']}{_RST2}")
    if total_tok:
        print(f"  {_BOLD}Tokens    {_RST2} {_CYAN}{_fmt_tokens(total_tok)}{_RST2}  "
              f"{_DIM}(in {_fmt_tokens(data['input_tokens'])} / "
              f"out {_fmt_tokens(data['output_tokens'])}){_RST}")
    print(f"  {_BOLD}Active    {_RST2} {_DIM}{first} → {last}{_RST}")
    print(f"{_CYAN}{'─'*_W}{_RST2}")

    if data["by_source"]:
        print(f"\n  {_BOLD}Platform{_RST2}")
        for src, cnt in data["by_source"]:
            print(f"    {_CYAN}{src:<14}{_RST2} {cnt}")

    if data["by_model"]:
        print(f"\n  {_BOLD}Model{_RST2}")
        for mdl, cnt in data["by_model"]:
            short = mdl.split("/")[-1] if "/" in mdl else mdl
            print(f"    {_CYAN}{short:<26}{_RST2} {cnt}")

    if data["top_tools"]:
        print(f"\n  {_BOLD}Top Tools{_RST2}")
        max_t = data["top_tools"][0][1]
        for tool, cnt in data["top_tools"]:
            bar = _bar(cnt, max_t)
            print(f"    {_CYAN}{tool:<20}{_RST2} {_GRN}{bar}{_RST2} {cnt}")

    dow = data["by_dow"]
    if any(dow.values()):
        print(f"\n  {_BOLD}Activity by Day{_RST2}")
        max_d = max(dow.values()) or 1
        for day, cnt in dow.items():
            bar = _bar(cnt, max_d)
            print(f"    {_DIM}{day}{_RST}  {_GRN}{bar}{_RST2}  {cnt}")

    print(f"\n{_CYAN}{'─'*_W}{_RST2}")
    print(f"{_DIM}  Database: {db}{_RST}\n")
```
