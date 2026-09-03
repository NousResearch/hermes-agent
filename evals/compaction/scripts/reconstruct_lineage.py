#!/usr/bin/env python3
"""Reconstruct the full uncompacted transcript of a session LINEAGE.

Rotation children start with a copy of the compressed parent (head + summary +
tail). To rebuild the real history: walk the chain root->leaf, append messages
not seen before (hash of role+content+tool_calls), and skip synthetic
compaction summaries / todo snapshots so we get the organic transcript.

Usage: reconstruct_lineage.py <state_db_copy> <root_session_id> <out_json>

ALWAYS run against a COPY of state.db, never the live file.
"""
import hashlib
import json
import sqlite3
import sys

DB = sys.argv[1]
ROOT = sys.argv[2]
OUT = sys.argv[3]

db = sqlite3.connect(DB)
db.row_factory = sqlite3.Row

# collect the whole descendant tree, chronological by started_at
import collections
children = collections.defaultdict(list)
for r in db.execute(
    "SELECT id, parent_session_id FROM sessions WHERE parent_session_id IS NOT NULL"
):
    children[r["parent_session_id"]].append(r["id"])
chain = []
frontier = [ROOT]
while frontier:
    sid = frontier.pop(0)
    chain.append(sid)
    frontier.extend(children.get(sid, []))
starts = {r["id"]: r["started_at"] or "" for r in db.execute(
    f"SELECT id, started_at FROM sessions WHERE id IN ({','.join('?'*len(chain))})", chain)}
chain.sort(key=lambda s: starts.get(s, ""))
print(f"chain: {len(chain)} sessions")

SYNTH_MARKERS = (
    "[CONTEXT COMPACTION", "[CONTEXT SUMMARY", "[PRIOR CONTEXT",
    "preserved across context compression",
)

seen = set()
out = []
sysprompt = None
for sid in chain:
    if sysprompt is None:
        row = db.execute(
            "SELECT s.system_prompt, sp.prompt AS dedup_prompt FROM sessions s "
            "LEFT JOIN system_prompts sp ON sp.hash = s.system_prompt_hash "
            "WHERE s.id=?", (sid,)).fetchone()
        if row:
            sysprompt = row["system_prompt"] or row["dedup_prompt"] or None
    for r in db.execute(
        "SELECT * FROM messages WHERE session_id=? ORDER BY id", (sid,)
    ):
        c = r["content"] or ""
        if any(m in c for m in SYNTH_MARKERS):
            continue  # synthetic compaction artifact, not organic history
        h = hashlib.md5(
            (r["role"] + "\x00" + c + "\x00" + (r["tool_calls"] or "")).encode(
                "utf-8", "replace")
        ).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        m = {"role": r["role"], "content": c}
        if r["tool_calls"]:
            try:
                m["tool_calls"] = json.loads(r["tool_calls"])
            except Exception:
                pass
        if r["tool_call_id"]:
            m["tool_call_id"] = r["tool_call_id"]
        if r["tool_name"]:
            m["tool_name"] = r["tool_name"]
        out.append(m)

msgs = [{"role": "system", "content": sysprompt or ""}] + out
chars = sum(len(m.get("content") or "") + len(json.dumps(m.get("tool_calls", ""), default=str)) for m in msgs)
print(f"reconstructed: {len(msgs)} msgs, {chars:,} chars (~{chars//4:,} tok)")
json.dump({"root": ROOT, "chain": chain, "messages": msgs}, open(OUT, "w", encoding="utf-8"), default=str)
print(f"wrote {OUT}")
