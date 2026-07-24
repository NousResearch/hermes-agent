#!/usr/bin/env python3
"""
Session Hook — Auto-capture facts from conversation context.

Called at session boundaries or on demand to extract and store
notable facts, decisions, and context from the current session.

Usage:
  python3 session_hook.py "summary of what happened this session"
  python3 session_hook.py --file path/to/session_log.md

Extracts facts using pattern matching (no LLM needed):
  - Decisions: "decided to...", "we chose...", "going with..."
  - Learnings: "learned that...", "turns out...", "discovered..."
  - Configs: "configured...", "set up...", "installed..."
  - Results: "fixed...", "resolved...", "deployed...", "built..."
  - Plans: "next step...", "todo...", "will need to..."
"""

import sys
import os
import re
import json
import sqlite3
import hashlib
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

DB_PATH = Path.home() / ".hermes/memory-engine/db/memory.db"

# ── Fact Extraction Patterns ──────────────────────────────────

PATTERNS = {
    "decision": [
        r"(?:decided|choosing|going with|we chose|picked|selected)\s+(.{20,150})",
        r"(?:decision|approach):\s*(.{20,150})",
    ],
    "learning": [
        r"(?:learned|discovered|turns out|realized|found out)\s+(?:that\s+)?(.{20,150})",
        r"(?:key insight|lesson|takeaway):\s*(.{20,150})",
    ],
    "config": [
        r"(?:configured|set up|installed|enabled|activated)\s+(.{15,150})",
        r"(?:config|setup):\s*(.{15,150})",
    ],
    "result": [
        r"(?:fixed|resolved|deployed|built|created|launched|completed)\s+(.{15,150})",
        r"(?:done|shipped|finished):\s*(.{15,150})",
    ],
    "plan": [
        r"(?:next step|todo|will need to|should|plan to)\s+(.{15,150})",
        r"(?:next|upcoming):\s*(.{15,150})",
    ],
    "metric": [
        r"(\d+(?:\.\d+)?(?:ms|s|KB|MB|GB|ops\/s|%)\s+.{10,100})",
        r"(?:p50|p99|throughput|latency)[\s:]+(.{10,100})",
    ],
}


def extract_facts(text: str) -> list:
    """Extract notable facts from text using patterns."""
    facts = []
    seen = set()
    
    for fact_type, patterns in PATTERNS.items():
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                content = match.group(1).strip()
                # Clean up
                content = re.sub(r'\s+', ' ', content)
                content = content.rstrip('.,;:')
                
                if len(content) < 15 or len(content) > 200:
                    continue
                
                # Dedup by hash
                h = hashlib.md5(content.lower().encode()).hexdigest()[:8]
                if h in seen:
                    continue
                seen.add(h)
                
                facts.append({
                    "content": content,
                    "type": fact_type,
                    "confidence": 0.8 if fact_type in ("plan", "metric") else 0.85
                })
    
    return facts


def store_facts(facts: list, source: str = "session_capture"):
    """Store extracted facts using the mem CLI internals."""
    if not facts:
        print("No facts extracted.")
        return 0
    
    # Lazy import embedder
    try:
        from mem import embed_text, detect_tags, get_conn
    except ImportError:
        # Direct implementation
        from fastembed import TextEmbedding
        model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        
        def embed_text(t):
            return list(model.embed([t]))[0].tolist()
        
        def detect_tags(t):
            return ["domain:general"]
        
        def get_conn():
            return sqlite3.connect(str(DB_PATH))
    
    conn = get_conn()
    cur = conn.cursor()
    stored = 0
    
    for fact in facts:
        content = fact["content"]
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        
        # Skip if exists
        cur.execute("SELECT id FROM memory_facts WHERE content_hash = ?", (content_hash,))
        if cur.fetchone():
            continue
        
        embedding = embed_text(content)
        tags = detect_tags(content)
        tags.append(f"type:{fact['type']}")
        now = datetime.now().isoformat()
        
        cur.execute("""INSERT INTO memory_facts 
                       (content, embedding, source, confidence, tags,
                        content_hash, created_at, updated_at, decay_weight,
                        freshness_tier, referenced_count, is_active, fact_type)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1.0, 'fresh', 0, 1, ?)""",
                    (content, json.dumps(embedding), source, fact["confidence"],
                     json.dumps(sorted(set(tags))), content_hash, now, now, fact["type"]))
        stored += 1
    
    conn.commit()
    conn.close()
    return stored


def main():
    if len(sys.argv) < 2:
        print("Usage: session_hook.py \"session summary text\"")
        print("       session_hook.py --file path/to/log.md")
        sys.exit(1)
    
    if sys.argv[1] == "--file":
        path = Path(sys.argv[2])
        if not path.exists():
            print(f"File not found: {path}")
            sys.exit(1)
        text = path.read_text()
    else:
        text = " ".join(sys.argv[1:])
    
    facts = extract_facts(text)
    print(f"Extracted {len(facts)} facts:")
    for f in facts:
        print(f"  [{f['type']}] {f['content'][:80]}")
    
    if facts:
        stored = store_facts(facts)
        print(f"\n✅ Stored {stored} new facts ({len(facts) - stored} duplicates skipped)")


if __name__ == "__main__":
    main()
