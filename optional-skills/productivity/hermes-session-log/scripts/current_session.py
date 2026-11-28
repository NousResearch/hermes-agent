#!/usr/bin/env python3
"""List Hermes sessions active within the last 30 minutes + their archive notes.

Prints, per session: id, source, open/closed, title, the first user message
(pick yours from a concurrent session by recognizing it), and the paths to
the archive note (.md) and raw transcript (.jsonl).
"""
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


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


def suffix_of(session_id: str) -> str:
    if session_id.startswith("local-"):
        return session_id[6:14]
    return session_id[4:].replace("_", "")


def main() -> None:
    cfg = load_cfg()
    db = Path(cfg["stateDb"]).expanduser() if cfg.get("stateDb") else auto_state_db()
    if db is None or not db.exists():
        print(f"hermes-session-log: state.db not found (looked for: {db})")
        raise SystemExit(1)
    vault = (Path(cfg.get("vaultPath") or os.environ.get("OBSIDIAN_VAULT_PATH")
                  or "~/Documents/Obsidian Vault").expanduser()
             / cfg.get("folder", "Hermes Sessions"))
    window = 1800  # seconds
    con = sqlite3.connect("file:" + str(db).replace("\\", "/") + "?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    now = time.time()
    rows = [dict(r) for r in con.execute(
        "SELECT id, source, title, message_count, cwd, ended_at, last_activity_at"
        " FROM sessions WHERE last_activity_at > ? ORDER BY last_activity_at DESC LIMIT 6",
        (now - window,))]
    if not rows:
        print(f"hermes-session-log: no session active within the last 30 minutes ({db})")
        return
    for r in rows:
        sfx = suffix_of(r["id"])
        cands = sorted(vault.glob(f"* {sfx}.md"))
        note = cands[-1] if cands else None
        fr = con.execute(
            "SELECT content FROM messages WHERE session_id=? AND role='user' AND active=1 AND content != '' ORDER BY id LIMIT 1",
            (r["id"],)).fetchone()
        first_user = (fr[0][:120] if fr and isinstance(fr[0], str) else "")
        live = "open" if r["ended_at"] is None else "closed"
        print(f"\n== {r['id'][:24]}… ({r['source']}, {live}, {r['message_count']} msgs, cwd {r['cwd']})")
        print(f"   title: {r['title']}")
        print(f"   first user message: {first_user}")
        print(f"   note: {note if note else '(not exported yet — run archive_hermes_sessions.py)'}")
        if note:
            print(f"   raw:  {Path(str(note)).with_suffix('.jsonl')}")
    print("\nYours is the session whose first user message belongs to this conversation.")


if __name__ == "__main__":
    main()
