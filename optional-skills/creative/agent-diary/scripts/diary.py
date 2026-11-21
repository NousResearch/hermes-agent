#!/usr/bin/env python3
"""agent-diary helper: the facts of one day, entry storage and the share card.

Standard library only. state.db is opened read only. Nothing is sent anywhere: the
script reads local files, writes into <HERMES_HOME>/diary/ and, for the card, starts a
local headless Chrome when one is installed. Every command prints JSON.

Commands:
  facts  [--date YYYY-MM-DD] [--tz Area/City]   facts of the day for the entry
  save   --date D --line TEXT [--title T] [--from FILE | stdin]   store the entry
  card   --date D [--lang en|de] [--theme night|paper]           render the share card
  recent [--n 3]                                 last entries (to avoid repeating jokes)
  check  --line TEXT [--date D]                  privacy check of a share line
"""
from __future__ import annotations

import argparse
import datetime as _dt
import glob
import html
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Optional

try:
    from itertools import pairwise
except ImportError:  # Python 3.9, still the system python3 on some Macs

    def pairwise(items):
        it = iter(items)
        prev = next(it, _END)
        for cur in it:
            yield prev, cur
            prev = cur

_END = object()


VERSION = "1.0.3"

# Background work is not "the day with the user". oneshot = scripted hermes chat -q runs.
EXCLUDED_SOURCES = tuple(x.strip() for x in os.environ.get(
    "AGENT_DIARY_EXCLUDE", "cron,subagent,kanban,batch,webhook,tool,oneshot").split(",") if x.strip())
LINE_MAX = 220
SNIPPET_MAX = 220
MAX_SNIPPETS = 14
MAX_TITLES = 20
MAX_TYPED_CHARS = 4000

# ---------------------------------------------------------------------------
# Paths and time
# ---------------------------------------------------------------------------


def hermes_home() -> Path:
    return Path(os.environ.get("HERMES_HOME") or Path.home() / ".hermes")


def diary_dir() -> Path:
    custom = os.environ.get("AGENT_DIARY_DIR", "").strip()
    return Path(custom).expanduser() if custom else hermes_home() / "diary"


def _config_timezone() -> str:
    """Top level 'timezone:' of the Hermes config.yaml, read as text (no YAML library needed)."""
    try:
        raw = (hermes_home() / "config.yaml").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    m = re.search(r"^timezone:[ \t]{0,8}['\"]?([A-Za-z_]{1,32}(?:/[A-Za-z0-9_+\-]{1,32}){0,2})", raw, re.MULTILINE)
    return m.group(1) if m else ""


def _tz(name: str):
    name = (name or os.environ.get("AGENT_DIARY_TZ", "") or _config_timezone() or os.environ.get("TZ", "")).strip()
    if name:
        try:
            from zoneinfo import ZoneInfo
            return ZoneInfo(name), name
        except (ImportError, KeyError, ValueError) as exc:  # unknown zone: fall back to local time
            sys.stderr.write(f"agent-diary: unknown timezone {name!r} ({type(exc).__name__}), using local time\n")
    local = _dt.datetime.now().astimezone().tzinfo
    return local, str(local or "local")


def _day(date_s: str, tz) -> _dt.date:
    if date_s in ("", "today"):
        return _dt.datetime.now(tz).date()
    if date_s == "yesterday":
        return _dt.datetime.now(tz).date() - _dt.timedelta(days=1)
    return _dt.date.fromisoformat(date_s)


def _window(day: _dt.date, tz) -> tuple[float, float]:
    start = _dt.datetime(day.year, day.month, day.day, tzinfo=tz)
    return start.timestamp(), (start + _dt.timedelta(days=1)).timestamp()


def _clock(ts: float, tz) -> str:
    return _dt.datetime.fromtimestamp(ts, tz).strftime("%H:%M")


def _ro(db: Path) -> sqlite3.Connection:
    uri = "file:" + urllib.request.pathname2url(str(db.resolve())) + "?mode=ro"
    con = sqlite3.connect(uri, uri=True, timeout=10)
    con.execute("PRAGMA query_only = 1")
    return con


def _cols(con: sqlite3.Connection, table: str) -> set[str]:
    return {r[1] for r in con.execute(f"PRAGMA table_info({table})")}


# ---------------------------------------------------------------------------
# Text cleaning: only what the user really typed counts
# ---------------------------------------------------------------------------

_FENCE = re.compile("\x60{3}.*?(?:\x60{3}|$)", re.DOTALL)
_QUOTE = re.compile(r"^[ \t]{0,8}>.*$", re.MULTILINE)
_DROP_MESSAGE = ("[system", "[skill", "[context", "<system", "[important:", "[async delegation",
                 "[your active task list", "[ide")
_DROP_LINE = _DROP_MESSAGE + ("[workspace::", "[out-of-band", "[/out-of-band", "[attached", "[the user sent",
                              "[group chat:", "[image", "[replying to:")
_REPLY_QUOTE = re.compile(r'^[ \t\n]{0,8}\[Replying to: "[\s\S]{0,2000}?"\]')


def clean_user_text(text) -> str:
    if not isinstance(text, str) or not text:
        return ""
    if text.lstrip()[:24].lower().startswith(_DROP_MESSAGE):
        return ""
    text = text[:MAX_TYPED_CHARS * 3]
    text = _REPLY_QUOTE.sub(" ", text)
    text = _FENCE.sub(" ", text)
    text = _QUOTE.sub(" ", text)
    lines = [ln for ln in text.splitlines() if not ln.lstrip()[:24].lower().startswith(_DROP_LINE)]
    return "\n".join(lines)[:MAX_TYPED_CHARS].strip()


def _one_line(text: str, limit: int = SNIPPET_MAX) -> str:
    s = " ".join(text.split())
    return s if len(s) <= limit else s[:limit - 1].rstrip() + "\u2026"


_W = r"(?<![\w])"
POLITE = re.compile(_W + r"(?:bitte|danke|dankesch(?:ö|oe)n|please|pls|plz|thanks|thank\s{1,3}you|thx|"
                    r"merci|gracias|por\s{1,3}favor)(?![\w])", re.IGNORECASE)
SWEAR = re.compile(_W + r"(?:fuck(?:ing|ed|er|s)?|wtf|ffs|shit(?:ty|s)?|bullshit|damn(?:it)?|crap(?:py)?|"
                   r"schei(?:ß|ss)(?:e|dreck|ding)?|verdammt(?:e|er|es)?|mist|kacke)(?![\w])", re.IGNORECASE)
# A message that opens like a correction or a complaint.
CORRECTION = re.compile(r"^[ \t]{0,4}(?:nein|ne[,!.]|nope|no[,!.]|no\s|falsch|wrong|again|nochmal|noch\s{1,3}mal|"
                        r"immer\s{1,3}noch|still\s{1,3}(?:not|doesn|broken|wrong)|that'?s\s{1,3}not|"
                        r"das\s{1,3}(?:ist|war)\s{1,3}(?:nicht|falsch)|doesn'?t\s{1,3}work|geht\s{1,3}nicht|"
                        r"funktioniert\s{1,3}nicht|klappt\s{1,3}nicht|stop|halt|warum\s{1,3}hast\s{1,3}du|"
                        r"why\s{1,3}did\s{1,3}you|wieso|you\s{1,3}forgot|du\s{1,3}hast\s{1,3}vergessen)", re.IGNORECASE)
# The agent admitting something, first 600 characters of a reply only.
ADMISSION = re.compile(_W + r"(?:you'?re\s{1,3}right|you\s{1,3}are\s{1,3}right|du\s{1,3}hast\s{1,3}recht|"
                       r"my\s{1,3}mistake|my\s{1,3}bad|mein\s{1,3}fehler|i\s{1,3}apologi[sz]e|good\s{1,3}catch|"
                       r"guter\s{1,3}punkt|stimmt,|da\s{1,3}hast\s{1,3}du\s{1,3}recht|sorry)(?![\w])", re.IGNORECASE)
_LETTER = re.compile(r"[A-Za-zÄÖÜäöüß]")


def _is_shout(text: str) -> bool:
    letters = _LETTER.findall(text)
    if len(letters) < 12:
        return False
    upper = sum(1 for c in letters if c.isupper())
    return upper / len(letters) > 0.7


_DE = re.compile(r"(?<![\w])(?:und|nicht|ich|das|ist|bitte|danke|der|die|mit|hab|mach|wir|du|auch|noch)(?![\w])", re.IGNORECASE)
_EN = re.compile(r"(?<![\w])(?:and|not|the|is|please|thanks|with|you|this|what|can|it|do)(?![\w])", re.IGNORECASE)


def _guess_lang(texts: list[str]) -> str:
    """One vote per message, so a long pasted English document does not outvote the user."""
    votes = Counter()
    for t in texts[:400]:
        de, en = len(_DE.findall(t[:600])), len(_EN.findall(t[:600]))
        if de != en:
            votes["de" if de > en else "en"] += 1
    return "de" if votes["de"] > votes["en"] else "en"


# ---------------------------------------------------------------------------
# facts
# ---------------------------------------------------------------------------


def collect_facts(day: _dt.date, tz, tz_name: str, home: Optional[Path] = None) -> dict:
    home = home or hermes_home()
    start, end = _window(day, tz)
    out: dict = {"ok": True, "version": VERSION, "date": day.isoformat(), "weekday": day.strftime("%A"),
                 "timezone": tz_name, "is_today": day == _dt.datetime.now(tz).date(),
                 "generated_at": _dt.datetime.now(tz).strftime("%H:%M")}
    db = Path(os.environ.get("AGENT_DIARY_STATE_DB", "").strip() or home / "state.db").expanduser()
    if not db.is_file():
        out.update(ok=False, error="state.db not found", next_step="Check HERMES_HOME.")
        return out
    con = _ro(db)
    try:
        scols, mcols = _cols(con, "sessions"), _cols(con, "messages")
        excl = ",".join("?" * len(EXCLUDED_SOURCES))
        last = "COALESCE(last_activity_at, ended_at, started_at)" if "last_activity_at" in scols else \
            "COALESCE(ended_at, started_at)"
        title = "title" if "title" in scols else "NULL"
        rows = con.execute(
            "SELECT id, source, " + title + ", started_at FROM sessions WHERE started_at < ? AND "
            + last + " >= ? AND COALESCE(source,'') NOT IN (" + excl + ")",
            (end, start, *EXCLUDED_SOURCES)).fetchall()
        sessions = {sid: {"source": src or "", "title": t or "", "started": st, "user": 0, "all": 0}
                    for sid, src, t, st in rows}
        first_ever = con.execute("SELECT MIN(started_at) FROM sessions WHERE COALESCE(source,'') NOT IN ("
                                 + excl + ")", EXCLUDED_SOURCES).fetchone()[0]
        cron_runs = con.execute("SELECT COUNT(*) FROM sessions WHERE source = 'cron' AND started_at >= ? "
                                "AND started_at < ?", (start, end)).fetchone()[0]
        sub_runs = con.execute("SELECT COUNT(*) FROM sessions WHERE source = 'subagent' AND started_at >= ? "
                               "AND started_at < ?", (start, end)).fetchone()[0]

        user_msgs: list[tuple[float, str, str]] = []  # (ts, sid, cleaned text)
        replies = admissions = tool_calls = 0
        admission_words: Counter = Counter()
        tools: Counter = Counter()
        has_tool = "tool_name" in mcols
        ids = list(sessions)
        for i in range(0, len(ids), 400):
            chunk = ids[i:i + 400]
            q = ("SELECT session_id, role, CASE WHEN role = 'user' THEN substr(content, 1, ?) "
                 "WHEN role = 'assistant' THEN substr(content, 1, 600) ELSE NULL END, "
                 + ("CASE WHEN role = 'tool' THEN tool_name ELSE NULL END" if has_tool else "NULL")
                 + ", timestamp FROM messages WHERE session_id IN (" + ",".join("?" * len(chunk)) + ") "
                 "AND timestamp >= ? AND timestamp < ? ORDER BY timestamp, id")
            for sid, role, content, tname, ts in con.execute(q, (MAX_TYPED_CHARS * 3, *chunk, start, end)):
                info = sessions.get(sid)
                if info is None:
                    continue
                info["all"] += 1
                if role == "user":
                    text = clean_user_text(content)
                    if text:
                        user_msgs.append((ts, sid, text))
                        info["user"] += 1
                elif role == "assistant":
                    if isinstance(content, str) and content.strip():
                        replies += 1
                        for m in ADMISSION.finditer(content):
                            admissions += 1
                            admission_words[m.group(0).lower().rstrip(",")] += 1
                            break
                elif role == "tool":
                    tool_calls += 1
                    if tname:
                        tools[tname] += 1
    finally:
        con.close()

    texts = [t for _ts, _sid, t in user_msgs]
    vocab = {w.lower() for t in texts for w in re.findall(r"[\w\-]{3,}", t)}
    vocab |= {w.lower() for s in sessions.values() for w in re.findall(r"[\w\-]{3,}", s["title"])}
    out["_vocab"] = vocab
    out["_title_vocab"] = {w.lower() for s in sessions.values() for w in re.findall(r"[\w\-]{3,}", s["title"])}
    active = {sid: s for sid, s in sessions.items() if s["user"]}
    polite = sum(len(POLITE.findall(t)) for t in texts)
    swears = sum(len(SWEAR.findall(t)) for t in texts)
    corrections = [(ts, t) for ts, _sid, t in user_msgs if CORRECTION.search(t)]
    shouts = [(ts, t) for ts, _sid, t in user_msgs if _is_shout(t)]
    words = sum(len(t.split()) for t in texts)

    out["user"] = {
        "messages": len(user_msgs), "words_typed": words,
        "first_message": _clock(user_msgs[0][0], tz) if user_msgs else None,
        "last_message": _clock(user_msgs[-1][0], tz) if user_msgs else None,
        "please_thanks": polite, "swears": swears, "corrections": len(corrections),
        "all_caps_messages": len(shouts),
        "questions": sum(1 for t in texts if t.rstrip().endswith("?")),
        "longest_message_words": max((len(t.split()) for t in texts), default=0),
        "longest_message_time": (_clock(max(user_msgs, key=lambda m: len(m[2].split()))[0], tz)
                                 if user_msgs else None),
        "longest_silence_minutes": _longest_gap(user_msgs),
        "messages_after_midnight": sum(1 for ts, _s, _t in user_msgs
                                       if _dt.datetime.fromtimestamp(ts, tz).hour < 5),
    }
    out["agent"] = {
        "replies": replies, "tool_calls": tool_calls,
        "top_tools": [{"tool": k, "calls": v} for k, v in tools.most_common(6)],
        "admissions": admissions, "admission_phrases": dict(admission_words.most_common(4)),
        "background_jobs_today": cron_runs, "subagents_today": sub_runs,
    }
    by_source = Counter(s["source"] for s in active.values())
    longest = max(active.values(), key=lambda s: s["all"], default=None)
    out["sessions"] = {
        "count": len(active), "by_source": dict(by_source.most_common()),
        "titles": [s["title"] for s in sorted(active.values(), key=lambda s: s["started"]) if s["title"]][:MAX_TITLES],
        "longest": {"title": longest["title"], "messages": longest["all"]} if longest else None,
    }
    out["snippets"] = _snippets(user_msgs, corrections, shouts, tz)
    out["language_guess"] = _guess_lang(texts)
    if first_ever:
        first_day = _dt.datetime.fromtimestamp(first_ever, tz).date()
        out["day_number"] = (day - first_day).days + 1
        out["together_since"] = first_day.isoformat()
    out["streak_days"] = _streak(day)
    out["quiet_day"] = len(user_msgs) == 0
    out["diary_dir"] = str(diary_dir())
    out["entry_exists"] = (diary_dir() / (day.isoformat() + ".md")).is_file()
    out["previous"] = recent_entries(3, before=day)
    return out


def _longest_gap(user_msgs) -> int:
    gaps = [b[0] - a[0] for a, b in pairwise(user_msgs)]
    return int(max(gaps) // 60) if gaps else 0


def _snippets(user_msgs, corrections, shouts, tz) -> list[dict]:
    """A small, varied sample of what the user typed: first, last, corrections, shouting, a spread."""
    if not user_msgs:
        return []
    picked: dict[float, str] = {}

    def add(ts, tag):
        if len(picked) < MAX_SNIPPETS and ts not in picked:
            picked[ts] = tag

    add(user_msgs[0][0], "first")
    add(user_msgs[-1][0], "last")
    for ts, _t in corrections[:4]:
        add(ts, "correction")
    for ts, _t in shouts[:2]:
        add(ts, "all_caps")
    step = max(1, len(user_msgs) // 8)
    for ts, _sid, _t in user_msgs[::step]:
        add(ts, "sample")
    text_by_ts = {ts: t for ts, _sid, t in user_msgs}
    return [{"time": _clock(ts, tz), "tag": tag, "text": _one_line(text_by_ts[ts])}
            for ts, tag in sorted(picked.items())]


def _streak(day: _dt.date) -> int:
    d, n = day - _dt.timedelta(days=1), 0
    while (diary_dir() / (d.isoformat() + ".md")).is_file():
        n += 1
        d -= _dt.timedelta(days=1)
    return n


# ---------------------------------------------------------------------------
# Entries
# ---------------------------------------------------------------------------

_FRONT = re.compile(r"\A---\n(.*?)\n---\n?", re.DOTALL)


def read_entry(path: Path) -> dict:
    raw = path.read_text(encoding="utf-8")
    meta: dict = {}
    m = _FRONT.match(raw)
    if m:
        for ln in m.group(1).splitlines():
            if ":" in ln:
                k, v = ln.split(":", 1)
                v = v.strip()
                try:
                    meta[k.strip()] = json.loads(v)
                except ValueError:
                    meta[k.strip()] = v
        raw = raw[m.end():]
    meta["body"] = raw.strip()
    return meta


def recent_entries(n: int = 3, before: Optional[_dt.date] = None) -> list[dict]:
    files = sorted(diary_dir().glob("[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9].md"))
    if before:
        files = [f for f in files if f.stem < before.isoformat()]
    out = []
    for f in files[-n:]:
        try:
            e = read_entry(f)
        except OSError:
            continue
        out.append({"date": f.stem, "title": e.get("title", ""), "line": e.get("line", ""),
                    "body": _one_line(e.get("body", ""), 600)})
    return out


_CAP_WORD = re.compile(r"(?<![\w])([A-ZÄÖÜ][\w\-]{2,})")
_COMMON = {"the", "this", "that", "today", "and", "but", "his", "her", "him", "she", "they", "nobody", "zero",
           "heute", "der", "die", "das", "ich", "und", "aber", "wir", "sie", "mein", "meine", "niemand", "null",
           "hermes", "agent", "diary", "tagebuch", "mensch", "human"}
_COMMON |= {d.lower() for d in ("Monday Tuesday Wednesday Thursday Friday Saturday Sunday January February March "
                                "April May June July August September October November December Montag Dienstag "
                                "Mittwoch Donnerstag Freitag Samstag Sonntag Januar Februar März Mai Juni Juli "
                                "Oktober Dezember Morgen Abend Nacht Mittag").split()}


def check_line(line: str, facts: Optional[dict] = None) -> list[str]:
    """Warnings for a share line. It goes on a public card, so it must not identify anyone."""
    warns = []
    if len(line) > LINE_MAX:
        warns.append(f"too_long: {len(line)} characters, keep it under {LINE_MAX}")
    if re.search(r"https?://|www\.|\S+@\S+\.\w", line):
        warns.append("contains_link_or_email")
    if re.search(r"\d{5,}", line):
        warns.append("contains_long_number")
    if "\u2014" in line or "\u2013" in line:
        warns.append("contains_dash: use a comma, a colon or a full stop")
    vocab = (facts or {}).get("_vocab") or set()
    if vocab:
        hits = sorted({w for w in _CAP_WORD.findall(line)
                       if w.lower() in vocab and w.lower() not in _COMMON and not w.isdigit()})
        if hits:
            warns.append("check_names: " + ", ".join(hits[:8]) + ". These words also appear in today's chats. "
                         "If any of them is a person, client, company, product or project, rewrite the line. "
                         "Ordinary words are fine.")
    return warns


_BRANDISH = re.compile(r"(?<![\w@])(?:[A-Za-zÄÖÜäöü]{1,30}[a-zäöü][A-Z][\w]{1,30}|[A-Za-z][\w\-]{1,40}\.(?:com|de|io|ai|net|org|co|app)\b|"
                       r"[A-Z]{2,}[a-z]{2,}[\w]{0,20})")


def check_body(body: str, facts: Optional[dict] = None) -> list[str]:
    """Words in the entry that look like names of companies, clients, products or projects.

    Candidates: brand-like spellings (CamelCase, domains) anywhere, and capitalized words that also
    appear in today's session titles (titles carry project and client names)."""
    hits = set(m.group(0) for m in _BRANDISH.finditer(body))
    titles = (facts or {}).get("_title_vocab") or set()
    if titles:
        hits |= {w for w in _CAP_WORD.findall(body) if w.lower() in titles and w.lower() not in _COMMON}
    hits = sorted(h for h in hits if h.lower() not in _COMMON)
    if not hits:
        return []
    return ["check_names_in_entry: " + ", ".join(hits[:12]) + ". Names of companies, clients, people, products, "
            "brands, websites and projects never appear in the diary. Replace each one that is such a name with a "
            "general description (a client, a directory, a sale) and save again."]


def _stats_from_facts(facts: dict) -> dict:
    u, a = facts.get("user", {}), facts.get("agent", {})
    return {"messages": u.get("messages", 0), "sessions": facts.get("sessions", {}).get("count", 0),
            "first": u.get("first_message"), "last": u.get("last_message"),
            "tool_calls": a.get("tool_calls", 0), "please": u.get("please_thanks", 0),
            "corrections": u.get("corrections", 0), "admissions": a.get("admissions", 0),
            "day_number": facts.get("day_number"), "lang": facts.get("language_guess", "en")}


SNAPSHOT_MAX_AGE = 6 * 3600


def write_snapshot(facts: dict) -> None:
    """Keep the numbers the agent saw, so the card matches the text even if the day goes on."""
    d = diary_dir() / ".facts"
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / (facts["date"] + ".json.tmp")
    tmp.write_text(json.dumps({"taken_at": _dt.datetime.now().timestamp(), "stats": _stats_from_facts(facts)}),
                   encoding="utf-8")
    os.chmod(tmp, 0o600)
    os.replace(tmp, d / (facts["date"] + ".json"))


def read_snapshot(date_s: str) -> Optional[dict]:
    p = diary_dir() / ".facts" / (date_s + ".json")
    try:
        snap = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if _dt.datetime.now().timestamp() - float(snap.get("taken_at", 0)) > SNAPSHOT_MAX_AGE:
        return None
    return snap.get("stats")


def save_entry(date_s: str, line: str, title: str, body: str, facts: Optional[dict]) -> dict:
    if not body.strip():
        return {"ok": False, "error": "empty entry", "next_step": "Pass the entry text on stdin or with --from."}
    line = " ".join(line.split())
    warns = check_line(line, facts) + check_body(body, facts)
    d = diary_dir()
    d.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(d, 0o700)
    except OSError:
        pass
    path = d / (date_s + ".md")
    stats = read_snapshot(date_s) or (_stats_from_facts(facts) if facts else {})
    head = ["---", "date: " + json.dumps(date_s), "title: " + json.dumps(title, ensure_ascii=False),
            "line: " + json.dumps(line, ensure_ascii=False), "stats: " + json.dumps(stats, ensure_ascii=False),
            "written_at: " + json.dumps(_dt.datetime.now().astimezone().isoformat(timespec="minutes")),
            "---", ""]
    tmp = path.with_suffix(".md.tmp")
    tmp.write_text("\n".join(head) + body.strip() + "\n", encoding="utf-8")
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)
    return {"ok": True, "path": str(path), "line_warnings": warns,
            "next_step": ("Read line_warnings, fix the share line or the entry and save again." if warns
                          else "Render the card.")}


# ---------------------------------------------------------------------------
# Card
# ---------------------------------------------------------------------------

LABELS = {
    "en": {"kicker": "AGENT DIARY", "day": "Day", "msgs": "messages", "tools": "tool calls",
           "please": "please & thanks", "corr": "corrections", "adm": "× admitted I was wrong",
           "foot": "Written by my AI agent · #AgentDiary · Hermes Agent"},
    "de": {"kicker": "AGENT-TAGEBUCH", "day": "Tag", "msgs": "Nachrichten", "tools": "Tool-Aufrufe",
           "please": "Bitte & Danke", "corr": "Korrekturen", "adm": "× Fehler zugegeben",
           "foot": "Geschrieben von meinem KI-Agenten · #AgentDiary · Hermes Agent"},
}
THEMES = {
    "night": {"bg": "#0b1020", "bg2": "#141b33", "ink": "#f4f1ea", "muted": "#aab4cc", "accent": "#ffb547",
              "chip": "rgba(255,255,255,0.07)", "line": "rgba(255,255,255,0.10)",
              "line2": "rgba(255,255,255,0.22)"},
    "paper": {"bg": "#f6f2eb", "bg2": "#ece5d8", "ink": "#1b1f2a", "muted": "#4f5461", "accent": "#c2410c",
              "chip": "rgba(0,0,0,0.04)", "line": "rgba(0,0,0,0.10)",
              "line2": "rgba(0,0,0,0.20)"},
}


def card_html(entry: dict, lang: str, theme: str) -> str:
    L = LABELS.get(lang, LABELS["en"])
    T = THEMES.get(theme, THEMES["night"])
    st = entry.get("stats") or {}
    date_s = entry.get("date", "")
    try:
        d = _dt.date.fromisoformat(date_s)
        months_de = ["Januar", "Februar", "März", "April", "Mai", "Juni", "Juli", "August", "September",
                     "Oktober", "November", "Dezember"]
        nice = (f"{d.day}. {months_de[d.month - 1]} {d.year}" if lang == "de" else d.strftime("%B %-d, %Y"))
    except ValueError:
        nice = date_s
    day_no = st.get("day_number")
    kicker = L["kicker"] + (f" · {L['day']} {day_no}" if day_no else "")
    line = entry.get("line", "")
    size = 74 if len(line) <= 90 else 64 if len(line) <= 140 else 56 if len(line) <= 180 else 50
    def num(v) -> str:
        s_ = f"{int(v):,}"
        return s_.replace(",", ".") if lang == "de" else s_

    chips = []  # (value, label, highlighted)
    if st.get("messages"):
        chips.append((num(st["messages"]), L["msgs"], False))
    if st.get("first") and st.get("last"):
        chips.append((st["first"] + " → " + st["last"], "", False))
    if st.get("tool_calls"):
        chips.append((num(st["tool_calls"]), L["tools"], False))
    for key, label in (("admissions", L["adm"]), ("corrections", L["corr"]), ("please", L["please"])):
        if st.get(key) and len(chips) < 4:
            chips.append((num(st[key]), label, True))  # the human detail gets the accent
    chip_html = "".join(f'<div class="chip{" hi" if hi else ""}"><b>{html.escape(v)}</b>'
                        f'{(" " + html.escape(k)) if k else ""}</div>' for v, k, hi in chips[:4])
    words = html.escape(line).split(" ")
    if len(words) > 2:  # no single word alone on the last line
        last = words.pop()
        words[-1] = words[-1] + "&nbsp;" + last
    quote = " ".join(words)
    return f"""<!doctype html><html lang="{lang}"><head><meta charset="utf-8"><style>
html,body{{margin:0;width:1080px;height:1350px;}}
body{{background:radial-gradient(1200px 900px at 85% -10%,{T['bg2']} 0%,{T['bg']} 60%);color:{T['ink']};
font-family:Inter,"Helvetica Neue",Arial,sans-serif;display:flex;flex-direction:column;box-sizing:border-box;
padding:96px 96px 80px;position:relative;overflow:hidden;}}
body:before{{content:"";position:absolute;inset:0;background-image:radial-gradient({T['line']} 1.4px,transparent 1.4px);
background-size:36px 36px;opacity:.55;pointer-events:none;}}
.k{{font-family:"JetBrains Mono",Menlo,Consolas,monospace;font-size:26px;letter-spacing:.22em;color:{T['accent']};
position:relative;white-space:nowrap;}}
.date{{margin-top:18px;font-size:30px;color:{T['muted']};position:relative;}}
.q{{margin-top:auto;margin-bottom:auto;position:relative;font-family:Georgia,"Times New Roman",serif;
font-style:italic;font-size:{size}px;line-height:1.22;letter-spacing:-.005em;text-wrap:balance;
border-left:6px solid {T['accent']};padding-left:44px;margin-left:-50px;}}
.q:before{{content:"\\201C";position:absolute;left:30px;top:-150px;font-size:260px;color:{T['accent']};
opacity:.35;font-style:normal;}}
.chips{{display:grid;grid-template-columns:1fr 1fr;gap:16px;position:relative;margin-bottom:56px;}}
.chip{{background:{T['chip']};border:1.5px solid {T['line2']};border-radius:22px;padding:18px 26px;font-size:26px;
color:{T['muted']};white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}}
.chip b{{color:{T['ink']};font-weight:700;}}
.chip.hi{{border-color:{T['accent']};color:{T['ink']};}}
.chip.hi b{{color:{T['accent']};}}
.f{{position:relative;border-top:1px solid {T['line']};padding-top:30px;font-size:24px;color:{T['muted']};
white-space:nowrap;}}
</style></head><body>
<div class="k">{html.escape(kicker)}</div><div class="date">{html.escape(nice)}</div>
<div class="q">{quote}</div>
<div class="chips">{chip_html}</div>
<div class="f">{html.escape(L['foot'])}</div>
</body></html>"""


PLAYWRIGHT_GLOBS = ("~/.cache/ms-playwright/chromium_headless_shell-*/chrome-headless-shell-*/chrome-headless-shell",
                    "~/Library/Caches/ms-playwright/chromium_headless_shell-*/chrome-headless-shell-*/chrome-headless-shell",
                    "~/.cache/ms-playwright/chromium-*/chrome-linux*/chrome")
BROWSERS = ("chrome-headless-shell", "google-chrome", "google-chrome-stable", "chrome", "chromium",
            "chromium-browser", "microsoft-edge", "brave-browser")
MAC_APPS = ("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            "/Applications/Chromium.app/Contents/MacOS/Chromium",
            "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
            "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge")


def _is_snap(path: str) -> bool:
    if path.startswith("/snap/") or os.path.realpath(path).endswith("/usr/bin/snap"):
        return True
    try:
        with open(path, "rb") as fh:
            head = fh.read(4096)
        return head.startswith(b"#!") and b"/snap/" in head
    except OSError:
        return False


def find_browsers() -> list[str]:
    explicit = os.environ.get("AGENT_DIARY_BROWSER", "").strip()
    if explicit:
        p = shutil.which(explicit) or (explicit if os.path.isfile(explicit) else None)
        return [p] if p else []
    found: list[str] = []
    for pattern in PLAYWRIGHT_GLOBS:
        hits = sorted(glob.glob(os.path.expanduser(pattern)))
        if hits:
            found.append(hits[-1])
    found += [p for p in (shutil.which(n) for n in BROWSERS) if p]
    found += [p for p in MAC_APPS if os.path.isfile(p)]
    seen: set[str] = set()
    uniq = [p for p in found if not (os.path.realpath(p) in seen or seen.add(os.path.realpath(p)))]
    # Snap browsers cannot read hidden folders such as the Hermes home, so they come last.
    return [p for p in uniq if not _is_snap(p)] + [p for p in uniq if _is_snap(p)]


def screenshot(browser: str, html_file: Path, png_file: Path) -> bool:
    args = [browser]
    if "headless-shell" not in os.path.basename(browser):
        args.append("--headless=new")
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        args.append("--no-sandbox")
    args += ["--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=1", "--window-size=1080,1350",
             "--disable-extensions", "--no-first-run", "--virtual-time-budget=1500",
             "--screenshot=" + str(png_file.resolve()), html_file.resolve().as_uri()]
    try:
        subprocess.run(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60, check=False)
    except (OSError, subprocess.SubprocessError):
        return False
    return png_file.is_file() and png_file.stat().st_size > 1000


def render_card(date_s: str, lang: str, theme: str) -> dict:
    path = diary_dir() / (date_s + ".md")
    if not path.is_file():
        return {"ok": False, "error": "no entry for " + date_s, "next_step": "Save the entry first."}
    entry = read_entry(path)
    if not entry.get("line"):
        return {"ok": False, "error": "entry has no share line", "next_step": "Save again with --line."}
    if lang not in LABELS:
        lang = (entry.get("stats") or {}).get("lang", "en")
        lang = lang if lang in LABELS else "en"
    cards = diary_dir() / "cards"
    cards.mkdir(parents=True, exist_ok=True)
    html_file = cards / (date_s + ".html")
    png_file = cards / (date_s + ".png")
    html_file.write_text(card_html(entry, lang, theme), encoding="utf-8")
    if png_file.exists():
        png_file.unlink()
    for browser in find_browsers():
        if screenshot(browser, html_file, png_file):
            return {"ok": True, "png": str(png_file), "html": str(html_file), "lang": lang, "theme": theme}
    return {"ok": True, "png": None, "html": str(html_file), "lang": lang, "theme": theme,
            "note": "No headless Chrome or Chromium found, only the HTML card was written. "
                    "Install one, or open the HTML file and take a screenshot."}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _print(obj) -> None:
    if isinstance(obj, dict):
        obj = {k: v for k, v in obj.items() if not k.startswith("_")}
    sys.stdout.write(json.dumps(obj, ensure_ascii=False, indent=1) + "\n")


def _cmd_facts(a, tz, tz_name) -> int:
    facts = collect_facts(_day(a.date, tz), tz, tz_name)
    if facts.get("ok"):
        write_snapshot(facts)
    _print(facts)
    return 0


def _cmd_save(a, tz, tz_name) -> int:
    day = _day(a.date, tz)
    body = Path(a.src).read_text(encoding="utf-8") if a.src else sys.stdin.read()
    facts = collect_facts(day, tz, tz_name)
    _print(save_entry(day.isoformat(), a.line, a.title, body, facts if facts.get("ok") else None))
    return 0


def _cmd_card(a, tz, _tz_name) -> int:
    _print(render_card(_day(a.date, tz).isoformat(), a.lang, a.theme))
    return 0


def _cmd_recent(a, _tz, _tz_name) -> int:
    _print({"ok": True, "entries": recent_entries(max(1, a.n))})
    return 0


def _cmd_check(a, tz, tz_name) -> int:
    facts = collect_facts(_day(a.date, tz), tz, tz_name) if a.date else None
    w = check_line(a.line, facts)
    _print({"ok": not w, "warnings": w})
    return 0


COMMANDS = {"facts": _cmd_facts, "save": _cmd_save, "card": _cmd_card, "recent": _cmd_recent, "check": _cmd_check}


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="diary.py", description="agent-diary helper")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("facts")
    f.add_argument("--date", default="today")
    f.add_argument("--tz", default="")
    s = sub.add_parser("save")
    s.add_argument("--date", default="today")
    s.add_argument("--tz", default="")
    s.add_argument("--line", required=True)
    s.add_argument("--title", default="")
    s.add_argument("--from", dest="src", default="")
    c = sub.add_parser("card")
    c.add_argument("--date", default="today")
    c.add_argument("--tz", default="")
    c.add_argument("--lang", default="auto")
    c.add_argument("--theme", default="night", choices=sorted(THEMES))
    r = sub.add_parser("recent")
    r.add_argument("--n", type=int, default=3)
    k = sub.add_parser("check")
    k.add_argument("--line", required=True)
    k.add_argument("--date", default="")
    k.add_argument("--tz", default="")
    return ap


def main(argv=None) -> int:
    a = _parser().parse_args(argv)
    try:
        tz, tz_name = _tz(getattr(a, "tz", ""))
        return COMMANDS[a.cmd](a, tz, tz_name)
    except (OSError, sqlite3.Error, ValueError, KeyError, subprocess.SubprocessError) as exc:
        # The agent gets a clear JSON message instead of a traceback.
        _print({"ok": False, "error": type(exc).__name__ + ": " + str(exc)[:300],
                "next_step": "Report this error to the user. Do not invent facts."})
        return 1


if __name__ == "__main__":
    sys.exit(main())
