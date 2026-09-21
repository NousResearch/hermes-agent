#!/usr/bin/env python3
"""
Migrate session history from any profile's state.db to LanceDB vector memory.

Usage:
    python migrate_sessions_to_lancedb.py --profile <name> [--dry-run]

What this does:
    1. Read all sessions from the profile's state.db
    2. For each session with valuable content: extract user + assistant messages
    3. Filter: skip short/trivial sessions
    4. Skip sessions already migrated (dedup by session_id)
    5. ★ Strip Pipeline — remove skill template prefixes at write time
    6. ★ Timestamp fix — use message-level timestamps, not write time
    7. Embed via Ollama HTTP API, write to profile's LanceDB

Replaces: migrate_chip/zunhunfan/financial_expert_sessions_to_lancedb.py
"""

import argparse
import json
import sqlite3
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import requests

# ── Import Strip Pipeline from optimize script (no code duplication) ──────────
_optimize_path = Path(__file__).resolve().parent / "optimize_lance_memory.py"
if not _optimize_path.exists():
    print(f"ERROR: optimize_lance_memory.py not found at {_optimize_path}")
    sys.exit(1)

import importlib.util as _iu
_spec = _iu.spec_from_file_location("optimize_lance_memory", str(_optimize_path))
_opt = _iu.module_from_spec(_spec)
_spec.loader.exec_module(_opt)

StripPipeline = _opt.StripPipeline
DEFAULT_STRIP_STAGES = _opt.DEFAULT_STRIP_STAGES

# ─── Config ───────────────────────────────────────────────────────────────────
HERMES_HOME = Path.home() / ".hermes"
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "bge-m3:latest"
VECTOR_DIM = 1024
MIN_ASST_CHARS = 200
BATCH_SIZE = 4

# ─── Ollama helpers ───────────────────────────────────────────────────────────

def ollama_embed(texts: list[str]) -> list[np.ndarray]:
    """Call Ollama /api/embed endpoint directly."""
    resp = requests.post(
        f"{OLLAMA_HOST}/api/embed",
        json={"model": OLLAMA_MODEL, "input": texts},
        timeout=300,
    )
    resp.raise_for_status()
    return [np.array(e, dtype=np.float32) for e in resp.json()["embeddings"]]


def check_ollama() -> bool:
    """Verify Ollama is running and model is available."""
    try:
        resp = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        models = [m["name"] for m in resp.json().get("models", [])]
        if OLLAMA_MODEL not in models:
            print(f"WARNING: Model '{OLLAMA_MODEL}' not loaded in Ollama.")
            print(f"  Loaded models: {models}")
            return False
        return True
    except Exception as e:
        print(f"ERROR: Cannot reach Ollama at {OLLAMA_HOST}: {e}")
        return False


# ─── LanceDB helpers ──────────────────────────────────────────────────────────

def build_schema():
    import pyarrow as pa
    return pa.schema([
        pa.field("id",         pa.string()),
        pa.field("content",    pa.string()),
        pa.field("role",       pa.string()),
        pa.field("session_id", pa.string()),
        pa.field("vector",     pa.list_(pa.float32(), VECTOR_DIM)),
        pa.field("created_at", pa.float64()),
        pa.field("metadata",   pa.string()),
    ])


def get_or_create_lance_db(profile_home: Path):
    import lancedb
    lance_dir = profile_home / "lance_memory"
    lance_dir.mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(str(lance_dir))
    table_name = "memories"
    try:
        table = db.open_table(table_name)
    except Exception:
        db.create_table(table_name, schema=build_schema())
        table = db.open_table(table_name)
        print(f"  Created new LanceDB table: {lance_dir}")
    return db, table_name, table


def get_existing_session_ids(table) -> set[str]:
    try:
        all_rows = table.search([0.0] * VECTOR_DIM).limit(10000).to_list()
        return {r["session_id"] for r in all_rows if r.get("session_id")}
    except Exception:
        return set()


# ─── Session processor ────────────────────────────────────────────────────────

SHORT_PATTERNS = [
    "你好", "您好", "hello",
    "你是什么模型", "你现在用的什么模型", "现在你是什么模型",
    "有什么我可以帮", "有什么能帮",
]


def is_empty_after_strip(cleaned: str) -> bool:
    """Check if stripped content is effectively empty."""
    empty_markers = ["[EMPTY:", "[EMPTY "]
    cleaned_s = cleaned.strip()
    return not cleaned_s or any(cleaned_s.startswith(m) for m in empty_markers)


def process_session(raw_rows: list, session_id: str, started_at: float,
                    start_dt: str, msg_count: int) -> dict | None:
    """
    Single-pass: check if valuable AND extract content simultaneously.
    Returns None if not worth migrating, otherwise returns migration dict.

    Content format:
    [user]
    xxx
    [assistant]
    yyy
    ...
    """
    user_msgs, asst_msgs = [], []
    # Collect timestamps for metadata
    msg_timestamps = []

    for row in raw_rows:
        if len(row) >= 3:
            role, content, timestamp = row[0], row[1], row[2]
        else:
            role, content = row[0], row[1]
            timestamp = None

        if role not in ('user', 'assistant'):
            continue
        if not content or not content.strip():
            continue
        if role == 'user' and len(content.strip()) < 5:
            continue

        if role == 'user':
            user_msgs.append(content.strip())
        else:
            asst_msgs.append(content.strip())

        if timestamp:
            msg_timestamps.append(timestamp)

    # ── Filter checks ──────────────────────────────────────────────────────
    if not user_msgs or not asst_msgs:
        return None
    if all(len(m) < 10 for m in user_msgs):
        return None
    pure_confirmation = all(
        any(p in m for p in SHORT_PATTERNS) and len(m) < 30
        for m in user_msgs
    )
    if pure_confirmation:
        return None
    total_asst = sum(len(m) for m in asst_msgs)
    if total_asst < MIN_ASST_CHARS:
        return None

    # ── Build content ──────────────────────────────────────────────────────
    lines = []
    for user_c, asst_c in zip(user_msgs, asst_msgs):
        lines.append(f"[user]\n{user_c}")
        lines.append(f"[assistant]\n{asst_c}")
    if len(user_msgs) > len(asst_msgs):
        for content in user_msgs[len(asst_msgs):]:
            lines.append(f"[user]\n{content}")
            lines.append("[assistant]\n")

    content = "\n\n".join(lines)
    if len(content) < 50:
        return None

    # Use the first message timestamp for created_at
    first_ts = msg_timestamps[0] if msg_timestamps else started_at

    return {
        "session_id": session_id,
        "started_at": first_ts,  # ★ fixed: use message timestamp
        "start_dt": start_dt,
        "msg_count": msg_count,
        "asst_chars": total_asst,
        "content": content,
        "preview": content[:200].replace("\n", " "),
        "msg_timestamps": msg_timestamps,
    }


# ─── Migration ─────────────────────────────────────────────────────────────────

def migrate(profile: str, dry_run: bool = False):
    t0 = time.time()

    # Resolve profile paths
    if profile == "default":
        profile_home = HERMES_HOME
    else:
        profile_home = HERMES_HOME / "profiles" / profile

    state_db = profile_home / "state.db"
    if not state_db.exists():
        print(f"ERROR: state.db not found at {state_db}")
        return

    # Load Strip Pipeline
    stages = DEFAULT_STRIP_STAGES
    strip_pipeline = StripPipeline(stages)

    print("=" * 60)
    print("Session History Migration: state.db → LanceDB")
    print(f"Profile: {profile}")
    print("=" * 60)
    print(f"  Source:    {state_db}")
    print(f"  Ollama:    {OLLAMA_HOST}")
    print(f"  Model:     {OLLAMA_MODEL}")
    print(f"  Min asst:  {MIN_ASST_CHARS} chars")
    print(f"  Strip:     {len(stages)} stages ({sum(1 for s in stages if s.get('enabled', True))} enabled)")
    print(f"  Dry run:   {dry_run}")
    print()

    if not check_ollama():
        print("ERROR: Ollama check failed. Aborting.")
        return

    # ── Read sessions ──────────────────────────────────────────────────────
    conn = sqlite3.connect(str(state_db))
    cur = conn.cursor()
    cur.execute("""
        SELECT id, message_count, started_at,
               datetime(started_at, 'unixepoch') as start_dt
        FROM sessions
        WHERE message_count > 0
        ORDER BY started_at
    """)
    sessions = cur.fetchall()
    print(f"Found {len(sessions)} sessions with messages.\n")

    # ── Get already migrated session IDs ──────────────────────────────────
    db, table_name, table = get_or_create_lance_db(profile_home)
    existing_ids = get_existing_session_ids(table)
    print(f"LanceDB already has {len(existing_ids)} sessions.\n")

    # ── Process sessions ───────────────────────────────────────────────────
    sessions_to_migrate = []
    skipped_already = 0
    skipped_short = 0

    for sid, msg_count, started_at, start_dt in sessions:
        if sid in existing_ids:
            skipped_already += 1
            continue

        cur.execute("""
            SELECT role, content, timestamp
            FROM messages
            WHERE session_id = ?
            ORDER BY timestamp
        """, (sid,))
        raw_messages = cur.fetchall()

        result = process_session(raw_messages, sid, started_at, start_dt, msg_count)
        if result:
            sessions_to_migrate.append(result)
        else:
            skipped_short += 1

    conn.close()

    print(f"=== Session Summary ===")
    print(f"  Total sessions:    {len(sessions)}")
    print(f"  Already migrated:  {skipped_already}")
    print(f"  Skipped (short):   {skipped_short}")
    print(f"  To migrate:        {len(sessions_to_migrate)}")
    print()

    if not sessions_to_migrate:
        print("No new sessions to migrate. Exiting.")
        return

    if dry_run:
        print("=== Sessions to migrate ===\n")
        for s in sessions_to_migrate:
            print(f"  {s['session_id']}")
            print(f"    started:  {s['start_dt']}")
            print(f"    msgs:     {s['msg_count']}")
            print(f"    asst:     {s['asst_chars']} chars")
            preview = s['preview'][:100]
            print(f"    preview:  {preview}...")
            print()
        print(f"[DRY RUN] Would migrate {len(sessions_to_migrate)} sessions.")
        return

    # ── Migrate with Strip Pipeline ────────────────────────────────────────
    print(f"=== Migrating {len(sessions_to_migrate)} sessions ===\n")

    migrated = 0
    errors = 0
    stripped_total = 0
    stripped_len_before = 0
    stripped_len_after = 0

    for batch_start in range(0, len(sessions_to_migrate), BATCH_SIZE):
        batch = sessions_to_migrate[batch_start:batch_start + BATCH_SIZE]

        # ★ Apply Strip Pipeline to each session's content
        stripped_texts = []
        for s in batch:
            raw = s["content"]
            cleaned, _found = strip_pipeline.run(raw, s["session_id"])
            if is_empty_after_strip(cleaned):
                stripped_texts.append(None)
                stripped_total += 1
            else:
                stripped_len_before += len(raw)
                stripped_len_after += len(cleaned)
                stripped_texts.append(cleaned)

        # Filter out empty-after-strip sessions
        valid_batch = [(s, t) for s, t in zip(batch, stripped_texts) if t is not None]
        if not valid_batch:
            print(f"  [SKIP] Batch {batch_start}: all stripped empty")
            continue

        texts = [t for _, t in valid_batch]

        # Embed
        try:
            embeddings = ollama_embed(texts)
        except Exception as e:
            print(f"  ERROR embedding batch: {e}")
            errors += len(texts)
            continue

        # Write
        now = time.time()
        rows = []
        for (s, cleaned), emb in zip(valid_batch, embeddings):
            msg_ts = s.get("msg_timestamps", [])
            rows.append({
                "id": str(uuid.uuid4()),
                "content": cleaned,  # ★ stripped content
                "role": "session_migrated",
                "session_id": s["session_id"],
                "vector": emb.tolist(),
                "created_at": s["started_at"],  # ★ message timestamp
                "metadata": json.dumps({
                    "started_at": s["start_dt"],
                    "msg_count": s["msg_count"],
                    "asst_chars": s["asst_chars"],
                    "message_timestamps": [ts if isinstance(ts, str) else
                        time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(ts))
                        for ts in msg_ts],
                    "strip_applied": True,
                }),
            })

        try:
            table.add(rows)
            migrated += len(rows)
            print(f"  [OK] Batch {batch_start}: {len(rows)} rows written")
        except Exception as e:
            print(f"  ERROR writing batch: {e}")
            errors += len(rows)

    # ── Summary ────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    strip_pct = (1 - stripped_len_after / stripped_len_before) * 100 if stripped_len_before else 0

    print()
    print("=" * 60)
    print("Migration Complete")
    print("=" * 60)
    print(f"  Sessions migrated:  {migrated}")
    print(f"  Stripped:           {stripped_total} (empty after strip)")
    print(f"  Strip ratio:        {strip_pct:.1f}% removed")
    print(f"  Errors:             {errors}")
    print(f"  Time:               {elapsed:.1f}s")
    print(f"  LanceDB total:      {table.count_rows()} rows")
    print()
    print("✓ Migrated content is stripped — no separate optimization needed.")
    print("  To verify: vec_memory_search any query → cos(vectors) should be < 0.95")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Migrate profile session history to LanceDB (with built-in Strip Pipeline)",
    )
    parser.add_argument("--profile", "-p", required=True,
                        help="Profile name (e.g. 'default', 'chip_expert')")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be migrated without writing")
    args = parser.parse_args()

    migrate(profile=args.profile, dry_run=args.dry_run)
