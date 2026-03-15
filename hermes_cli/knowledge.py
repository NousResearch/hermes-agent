"""hermes memory / hermes knowledge CLI commands."""
from __future__ import annotations
from pathlib import Path
import os


def cmd_memory(args):
    from tools.memory_tool import MemoryStore
    store = MemoryStore()
    store.load_from_disk()
    action = getattr(args, "memory_action", None)

    if action == "list" or action is None:
        for target in ("memory", "user"):
            entries = store.list_entries(target)
            print()
            print("=== {} ({} entries) ===".format(target.upper(), len(entries)))
            if not entries:
                print("  (empty)")
                continue
            for e in entries:
                ts = e["saved_at"] or "unknown time"
                sid = e["session_id"] or "unknown session"
                print("  [{} | session:{}]".format(ts, sid))
                for line in e["content"].splitlines():
                    print("    {}".format(line))
                print()
    elif action == "delete":
        targets = ["memory", "user"] if args.target == "all" else [args.target]
        deleted = False
        for t in targets:
            result = store.remove(t, args.text)
            if result["success"]:
                print("Deleted from {}: {}".format(t, args.text))
                deleted = True
        if not deleted:
            print("No entry matched: {}".format(args.text))
    elif action == "clear":
        for target in ("memory", "user"):
            store.clear_all(target)
        print("All memory entries cleared.")
    elif action == "forget-session":
        result = store.forget_session(args.session_id)
        print("Removed {} entries from session {}".format(result["removed"], args.session_id))
    else:
        print("Usage: hermes memory [list|delete|clear|forget-session]")


def cmd_knowledge(args):
    action = getattr(args, "knowledge_action", None)
    if action == "forget-session":
        from tools.memory_tool import MemoryStore
        store = MemoryStore()
        store.load_from_disk()
        result = store.forget_session(args.session_id)
        print("Memory: removed {} entries from session {}".format(result["removed"], args.session_id))
        # Also delete session from sessions DB
        try:
            from hermes_state import SessionStore
            sessions_dir = Path(os.getenv("HERMES_HOME", str(Path.home() / ".hermes"))) / "sessions"
            ss = SessionStore(sessions_dir)
            deleted = ss.delete_session(args.session_id)
            if deleted:
                print("Session history: deleted session {}".format(args.session_id))
            else:
                print("Session history: session {} not found in DB".format(args.session_id))
        except Exception as e:
            print("Session history: could not delete ({})".format(e))
        return
    print()
    print("=" * 60)
    print("  HERMES KNOWLEDGE SUMMARY")
    print("=" * 60)
    from tools.memory_tool import MemoryStore
    store = MemoryStore()
    store.load_from_disk()
    for target in ("memory", "user"):
        entries = store.list_entries(target)
        print()
        print("--- MEMORY ({}) [{} entries] ---".format(target.upper(), len(entries)))
        if not entries:
            print("  (empty)")
        else:
            for e in entries:
                ts = e["saved_at"] or "unknown"
                sid = e["session_id"] or "unknown"
                content_preview = e["content"][:120].replace("\n", " ")
                print("  [{} | {}] {}".format(ts, sid, content_preview))
    hermes_home = os.getenv("HERMES_HOME", str(Path.home() / ".hermes"))
    skills_dir = Path(hermes_home) / "skills"
    print()
    print("--- SKILLS ---")
    if skills_dir.exists():
        skill_files = list(skills_dir.rglob("SKILL.md"))
        if skill_files:
            for sf in skill_files:
                rel = sf.relative_to(skills_dir)
                # Read provenance from SKILL.md comment
                try:
                    first_line = sf.read_text(encoding="utf-8").splitlines()[0]
                    if first_line.startswith("<!-- installed:"):
                        import re
                        m = re.match(r"<!-- installed:([^|]+)\|source:([^>]+) -->", first_line)
                        if m:
                            ts, src = m.group(1), m.group(2)
                            print("  {} [installed:{} | source:{}]".format(rel, ts[:10], src))
                            continue
                except Exception:
                    pass
                print("  {}".format(rel))
        else:
            print("  (no installed skills)")
    else:
        print("  (skills directory not found)")
    print()
    print("--- RECENT SESSIONS (last 5) ---")
    try:
        from hermes_state import SessionStore
        sessions_dir = Path(hermes_home) / "sessions"
        ss = SessionStore(sessions_dir)
        sessions = ss.list_sessions_rich(limit=5)
        if sessions:
            for s in sessions:
                sid = s.get("session_id", "")[:12]
                title = s.get("title") or "(untitled)"
                started = s.get("started_at", "")[:10]
                print("  [{}] {}... {}".format(started, sid, title))
        else:
            print("  (no sessions)")
    except Exception as e:
        print("  (could not load sessions: {})".format(e))
    print()
