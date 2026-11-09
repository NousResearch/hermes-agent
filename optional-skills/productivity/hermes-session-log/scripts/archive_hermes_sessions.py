#!/usr/bin/env python3
"""Export Hermes sessions to an Obsidian vault (lossless live archive).

Primary source: state.db (sessions + messages) — live, includes the
currently-active session. Legacy backfill: request_dump_*.json files under
<sessions dir> (error/retry artifacts — only for sessions absent from the DB).

Per session:
    <YYYY-MM-DD HH-MM suffix>.md      full dialogue; tool outputs clipped
    <...>.jsonl                       raw transcript: every message, no cuts
    Session Index.md                  rebuilt when something changed

Idempotent: re-runs rewrite only changed sessions and print nothing on no-op
(cron watchdog pattern: an empty stdout delivers nothing).

Configuration: scripts/config.json next to this file (copy config.example.json),
or the HERMES_SESSION_LOG_CONFIG environment variable pointing at a JSON file.
"""
import json
import os
import re
import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
INDEX_NAME = "Session Index.md"
RAW_FIELDS = ("id", "role", "timestamp", "tool_name", "content", "tool_calls",
              "reasoning", "compacted", "active", "_compressed_summary")


def load_cfg() -> dict:
    candidates = []
    env_cfg = os.environ.get("HERMES_SESSION_LOG_CONFIG", "")
    if env_cfg:
        candidates.append(Path(env_cfg).expanduser())
    candidates += [HERE / "config.json", HERE / "config.example.json"]
    for p in candidates:
        if p.is_file():
            return json.loads(p.read_text(encoding="utf-8"))
    return {}


def auto_state_db():
    cands = []
    for env in ("HERMES_STATE_DB", "HERMES_SESSION_LOG_STATE_DB"):
        v = os.environ.get(env)
        if v:
            cands.append(Path(v).expanduser())
    if os.environ.get("HERMES_HOME"):
        cands.append(Path(os.environ["HERMES_HOME"]).expanduser() / "state.db")
    cands.append(Path.home() / ".hermes" / "state.db")
    if os.name == "nt" and os.environ.get("LOCALAPPDATA"):
        cands.append(Path(os.environ["LOCALAPPDATA"]) / "hermes" / "state.db")
    for c in cands:
        try:
            if c.exists():
                return c
        except OSError:
            pass
    return cands[0] if cands else None


def resolve_paths(cfg: dict):
    vault = Path(cfg.get("vaultPath") or os.environ.get("OBSIDIAN_VAULT_PATH")
                 or "~/Documents/Obsidian Vault").expanduser()
    folder = cfg.get("folder", "Hermes Sessions")
    state_db = Path(cfg["stateDb"]).expanduser() if cfg.get("stateDb") else auto_state_db()
    dump_dir = Path(cfg["dumpDir"]).expanduser() if cfg.get("dumpDir") else None
    if dump_dir is None and state_db is not None:
        guess = state_db.parent / "sessions"
        dump_dir = guess if guess.is_dir() else None
    return vault / folder, state_db, dump_dir


def suffix_of(session_id: str) -> str:
    if session_id.startswith("local-"):
        return session_id[6:14]
    # legacy: YYYYMMDD_HHMMSS[_hex6] -> MMDDHHMMSS[hex6], unique per session
    return session_id[4:].replace("_", "")


def clip(s, n):
    s = str(s)
    return s if len(s) <= n else s[:n] + " ...[truncated]"


def fmt_ts(ts):
    try:
        return datetime.fromtimestamp(float(ts)).strftime("%H:%M:%S")
    except Exception:
        return "??"


def load_state(state_file):
    try:
        return json.loads(state_file.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def build_index(vault: Path) -> None:
    rows = []
    for f in vault.glob("*.md"):
        if f.name == INDEX_NAME:
            continue
        txt = f.read_text(encoding="utf-8")

        def g(k: str) -> str:
            m = re.search(rf"^{k}: (.*?)\s*$", txt, re.M)
            return m.group(1).strip('"') if m else ""

        rows.append((g("session_start"), f.name, g("title"), g("project"),
                     g("status"), g("messages"), g("source")))
    rows.sort(reverse=True)
    lines = ["# Session Index\n",
             "| Started | Session | Project | Status | Messages | Source |\n|---|---|---|---|---|---|"]
    for start, name, title, project, status, msgs, source in rows:
        t = (title or name)[:60].replace("|", "/")
        lines.append(f"| {start[:16]} | [[{name[:-3]}|{t}]] | {project.replace('|', '/')} | {status} | {msgs} | {source} |")
    (vault / INDEX_NAME).write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_db_session(con, s: dict, state: dict, vault: Path, raw_clip: int, tool_clip: int, full: bool):
    """Return {note, changed} for a db session, or None if nothing to export."""
    sid = s["id"]
    sfx = suffix_of(sid)
    msgs = [dict(r) for r in con.execute(
        "SELECT id, role, content, tool_name, tool_calls, reasoning, compacted, active, timestamp, _compressed_summary"
        " FROM messages WHERE session_id=? ORDER BY id", (sid,))]
    body_rows = [m for m in msgs if m.get("active") in (1, None)]
    users = [m for m in body_rows if m["role"] == "user" and (m.get("content") or "").strip()]
    if not users or len(body_rows) < 2:
        return None
    max_id = max(m["id"] for m in msgs)
    stem = f"{datetime.fromtimestamp(float(s['started_at'])):%Y-%m-%d %H-%M} {sfx}"
    st = state.get("db:" + sid)
    if not full and st and st.get("max_id") == max_id and st.get("ended_at") == s["ended_at"] \
            and (vault / (st["note"] + ".md")).exists():
        return {"note": st["note"], "changed": False}

    first_user = next((m["content"].strip() for m in users), "")
    title = s.get("title") or re.sub(r"\s+", " ", first_user)[:80] or f"Session {sfx}"
    raw_name = stem + ".jsonl"
    compacted_n = sum(1 for m in msgs if m.get("compacted"))
    summaries = [m["_compressed_summary"] for m in msgs
                 if m.get("_compressed_summary") not in (None, "", "0")]
    status = "open" if s["ended_at"] is None else (s.get("end_reason") or "closed")
    q = chr(34)
    front = "\n".join([
        "---",
        f"title: {q}{title.replace(q, chr(39))}{q}",
        "tags: [hermes/session]",
        f"session_id: {q}{sid}{q}",
        f"session_start: {q}{datetime.fromtimestamp(float(s['started_at'])).isoformat(timespec='seconds')}{q}",
        f"source: {s.get('source', '')}",
        f"model: {s.get('model', '')}",
        f"profile: {s.get('profile_name', '')}",
        f"project: {q}{str(s.get('cwd') or '').replace(q, chr(39))}{q}",
        f"status: {q}{status}{q}",
        f"messages: {len(body_rows)}",
        f"compacted: {compacted_n}",
        f"updated: {q}{datetime.now().isoformat(timespec='seconds')}{q}",
        f"raw: {q}{raw_name}{q}",
        "---",
        "",
    ])
    parts = [
        f"# {title}\n",
        f"> Full lossless dialogue ({len(body_rows)} messages). Tool outputs clipped ({raw_clip} chars); complete version (tool_calls, reasoning) — `{raw_name}`.\n",
    ]
    if compacted_n:
        c = f"## 🗜 Context compaction: {compacted_n} earlier message(s) compacted.\n\n"
        if summaries:
            c += "Summary: " + clip(summaries[-1], 500) + "\n"
        c += f"Full pre-compaction messages — in `{raw_name}` (flag `compacted: 1`).\n"
        parts.append(c)
    for m in body_rows:
        role = m["role"]
        ts = fmt_ts(m.get("timestamp"))
        if role == "system":
            continue
        if role == "tool":
            parts.append(f"## 🔧 {m.get('tool_name') or 'Tool'} · {ts}\n\n> {clip(m.get('content') or '', raw_clip)}\n")
            continue
        icon, label = ("👤", "User") if role == "user" else ("🤖", "Assistant")
        txt = (m.get("content") or "").strip()
        calls = m.get("tool_calls")
        if role == "assistant" and not txt and not calls:
            continue
        parts.append(f"## {icon} {label} · {ts}\n")
        if txt:
            parts.append(txt + "\n")
        if role == "assistant" and calls:
            try:
                for c in json.loads(calls):
                    fn = c.get("function") or {}
                    parts.append(f"> 🔧 **{fn.get('name')}**: `{clip(fn.get('arguments') or '', tool_clip)}`\n")
            except (json.JSONDecodeError, AttributeError):
                parts.append(f"> 🔧 `{clip(calls, tool_clip)}`\n")
    (vault / (stem + ".md")).write_text(front + "\n".join(parts) + "\n", encoding="utf-8")
    with open(vault / raw_name, "w", encoding="utf-8") as fh:
        for m in msgs:
            fh.write(json.dumps({k: m.get(k) for k in RAW_FIELDS}, ensure_ascii=False) + "\n")
    state["db:" + sid] = {"note": stem, "max_id": max_id, "ended_at": s["ended_at"], "title": title}
    return {"note": stem, "changed": True}


def export_dump_session(key: str, lst, state: dict, vault: Path, raw_clip: int, full: bool, db_suffixes):
    """Legacy backfill from request_dump files (error/retry artifacts)."""
    if suffix_of(key) in db_suffixes:
        return None
    lst.sort(key=lambda x: x[0])
    oldest, newest = lst[0], lst[-1]
    sfx = suffix_of(key)
    stem = f"{datetime.fromtimestamp(oldest[0]):%Y-%m-%d %H-%M} {sfx}"
    st = state.get("dump:" + key)
    if not full and st and st.get("mtime") == newest[0] and (vault / (st["note"] + ".md")).exists():
        return {"note": st["note"], "changed": False}
    dump = newest[2]
    msgs = [m for m in (dump.get("request") or {}).get("body", {}).get("messages", []) if isinstance(m, dict)]
    users = [m for m in msgs if m.get("role") == "user"]
    assts = [m for m in msgs if m.get("role") == "assistant"]
    if not users or not any((m.get("content") or "").strip() for m in assts):
        return None
    q = chr(34)
    first_user = next((m["content"].strip() for m in users if isinstance(m.get("content"), str) and m["content"].strip()), "")
    title = re.sub(r"\s+", " ", first_user)[:80] or f"Session {sfx}"
    raw_name = stem + ".dump.json"
    front = "\n".join([
        "---",
        f"title: {q}{title.replace(q, chr(39))}{q}",
        "tags: [hermes/session]",
        f"session_id: {q}{key}{q}",
        f"session_start: {q}{datetime.fromtimestamp(oldest[0]).isoformat(timespec='seconds')}{q}",
        "source: dump",
        f"model: {(dump.get('request') or {}).get('body', {}).get('model', 'unknown')}",
        f"status: {q}{dump.get('reason', 'unknown')}{q}",
        f"messages: {len(msgs)}",
        f"updated: {q}{datetime.now().isoformat(timespec='seconds')}{q}",
        f"raw: {q}{raw_name}{q}",
        "---",
        "",
    ])
    parts = [f"# {title}\n",
             f"> Legacy backfill from request_dump (error/retry artifacts); `{raw_name}` — full snapshot at the time of the failure.\n"]
    n = 0
    for m in msgs:
        role = m.get("role")
        if role == "system":
            continue
        t = m.get("content")
        t = t if isinstance(t, str) else json.dumps(t, ensure_ascii=False)
        if not t.strip():
            continue
        n += 1
        if role == "user":
            parts.append(f"## 👤 User · #{n}\n\n{t.strip()}\n")
        elif role == "assistant":
            parts.append(f"## 🤖 Assistant · #{n}\n\n{t.strip()}\n")
        else:
            parts.append(f"## 🔧 Tool · #{n}\n\n> {clip(re.sub(chr(10), " ", t), raw_clip)}\n")
    (vault / (stem + ".md")).write_text(front + "\n".join(parts) + "\n", encoding="utf-8")
    shutil.copyfile(newest[1], vault / raw_name)
    state["dump:" + key] = {"note": stem, "mtime": newest[0], "title": title}
    return {"note": stem, "changed": True}


def main() -> None:
    full = "--full" in sys.argv[1:]
    cfg = load_cfg()
    vault, db, dump_dir = resolve_paths(cfg)
    raw_clip = int(cfg.get("rawClipChars", 300))
    tool_clip = int(cfg.get("toolClipChars", 400))
    state_file = vault / ".export-state.json"
    vault.mkdir(parents=True, exist_ok=True)
    state = {} if full else load_state(state_file)
    changed = unchanged = skipped = 0
    log: list = []
    db_owned = {}
    db_ids = set()
    if db and db.exists():
        con = sqlite3.connect("file:" + str(db).replace("\\", "/") + "?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
        for r in con.execute("SELECT * FROM sessions WHERE COALESCE(hidden, 0) = 0 ORDER BY started_at"):
            s = dict(r)
            sfx = suffix_of(s["id"])
            db_ids.add(s["id"])
            db_ids.add(s["id"][:16])
            try:
                entry = export_db_session(con, s, state, vault, raw_clip, tool_clip, full)
            except Exception as e:
                print(f"  ! db session {s['id'][:24]}: {e}")
                continue
            if entry:
                db_owned[sfx] = entry["note"]
                if entry["changed"]:
                    changed += 1
                    log.append(f"OK {entry['note']}.md")
                else:
                    unchanged += 1
            else:
                skipped += 1
        con.close()
    if dump_dir and dump_dir.is_dir():
        groups = {}
        for p in dump_dir.glob("request_dump_*.json"):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError, UnicodeDecodeError):
                continue
            key = data.get("session_id") or ""
            if not key or key in db_ids or key[:16] in db_ids:
                continue
            groups.setdefault(key, []).append((p.stat().st_mtime, p, data))
        for key, lst in sorted(groups.items()):
            entry = export_dump_session(key, lst, state, vault, raw_clip, full, set(db_owned))
            if entry is None:
                if suffix_of(key) not in db_owned:
                    skipped += 1
            elif entry["changed"]:
                changed += 1
                log.append(f"OK {entry['note']}.md (dump)")
            else:
                unchanged += 1
    # prune foreign files of db-owned suffixes (legacy dump notes superseded by db)
    pat = re.compile(r"^.* (\w{6,14})\.(?:md|jsonl|dump\.json)$")
    for f in list(vault.iterdir()):
        m = pat.match(f.name)
        if m and m.group(1) in db_owned:
            own = db_owned[m.group(1)]
            if f.name not in (own + ".md", own + ".jsonl"):
                f.unlink()
                log.append(f"pruned {f.name}")
    state_file.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")
    if log:
        build_index(vault)
        print("\n".join(log))
        print(f"hermes-session-log: {changed} updated, {unchanged} unchanged, {skipped} skipped")
    # silent on no-op (cron watchdog pattern: empty stdout sends nothing)


if __name__ == "__main__":
    main()
