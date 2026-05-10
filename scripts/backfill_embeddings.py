#!/usr/bin/env python3
"""Backfill message_embedding for messages that lack them.

Handles two cases:
1. Normal messages: use content field
2. Assistant messages with empty content but reasoning: use reasoning field

Usage:
    python3 scripts/backfill_embeddings.py              # dry run (preview)
    python3 scripts/backfill_embeddings.py --apply      # execute
    python3 scripts/backfill_embeddings.py --apply --batch-size 100  # larger batches
"""

import argparse
import json
import os
import sqlite3
import struct
import sys
import time
import urllib.request

# Default DB path
DB_PATH = os.path.join(os.path.expanduser("~/.hermes"), "state.db")

# Config from config.yaml
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "http://2080ti:8081")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "Qwen3-Embedding-0.6B")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))
MAX_CHARS = 2048
BATCH_SIZE_DEFAULT = 50


def compute_embedding(text: str) -> bytes | None:
    """Compute embedding via local embedding endpoint."""
    if not text or not text.strip():
        return None
    try:
        payload = json.dumps({
            "model": EMBEDDING_MODEL,
            "input": text[:MAX_CHARS],
            "encoding_type": "float",
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{EMBEDDING_BASE_URL}/v1/embeddings",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {EMBEDDING_API_KEY}" if EMBEDDING_API_KEY else "",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read())

        data = result.get("data", [])
        if data and len(data) > 0:
            embedding = data[0].get("embedding")
            if embedding and isinstance(embedding, list):
                return struct.pack(f"{len(embedding)}f", *embedding)
    except Exception as e:
        print(f"  Embedding failed: {e}", file=sys.stderr)
    return None


def compute_embedding_batch(texts: list[str]) -> list[bytes | None]:
    """Compute embeddings for multiple texts in one API call."""
    valid_texts = [t for t in texts if t and t.strip()]
    if not valid_texts:
        return [None] * len(texts)

    truncated = [t[:MAX_CHARS] for t in valid_texts]

    try:
        payload = json.dumps({
            "model": EMBEDDING_MODEL,
            "input": truncated,
            "encoding_type": "float",
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{EMBEDDING_BASE_URL}/v1/embeddings",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {EMBEDDING_API_KEY}" if EMBEDDING_API_KEY else "",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())

        data = result.get("data", [])
        # Build result list preserving order
        results = [None] * len(texts)
        valid_idx = 0
        for i, text in enumerate(texts):
            if valid_idx < len(data) and data[valid_idx].get("embedding"):
                emb = data[valid_idx]["embedding"]
                results[i] = struct.pack(f"{len(emb)}f", *emb)
            valid_idx += 1
        return results
    except Exception as e:
        print(f"  Batch embedding failed: {e}", file=sys.stderr)
        return [None] * len(texts)


def main():
    parser = argparse.ArgumentParser(description="Backfill message embeddings")
    parser.add_argument("--apply", action="store_true", help="Actually update the database")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE_DEFAULT, help="Batch size for API calls")
    parser.add_argument("--db", default=DB_PATH, help="Path to state.db")
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Count messages needing backfill
    cur.execute("SELECT COUNT(*) FROM messages WHERE message_embedding IS NULL")
    total_needing = cur.fetchone()[0]
    print(f"Messages needing backfill: {total_needing}")

    if total_needing == 0:
        print("Nothing to do!")
        conn.close()
        return

    # Categorize messages
    cur.execute("""
        SELECT id, role, content, reasoning,
               CASE WHEN content IS NULL OR TRIM(content) = '' THEN 'empty' ELSE 'normal' END as content_status,
               CASE WHEN reasoning IS NOT NULL AND TRIM(reasoning) != '' THEN 'has_reasoning' ELSE 'no_reasoning' END as reasoning_status
        FROM messages
        WHERE message_embedding IS NULL
    """)

    empty_content = []
    normal_content = []
    no_reasoning = []

    for row in cur.fetchall():
        if row["content_status"] == "empty" and row["reasoning_status"] == "has_reasoning":
            empty_content.append(row)
        elif row["content_status"] == "normal":
            normal_content.append(row)
        else:
            no_reasoning.append(row)

    print(f"  Normal content: {len(normal_content)}")
    print(f"  Empty content + has reasoning: {len(empty_content)}")
    print(f"  Empty content + no reasoning (skip): {len(no_reasoning)}")

    if not args.apply:
        print("\nDry run mode. Add --apply to execute.")
        conn.close()
        return

    # Process normal content messages
    print(f"\nProcessing {len(normal_content)} normal content messages...")
    processed = 0
    failed = 0
    start_time = time.time()

    for i in range(0, len(normal_content), args.batch_size):
        batch = normal_content[i:i + args.batch_size]
        texts = [r["content"] for r in batch]
        embeddings = compute_embedding_batch(texts)

        for row, emb in zip(batch, embeddings):
            if emb:
                cur.execute(
                    "UPDATE messages SET message_embedding = ? WHERE id = ?",
                    (emb, row["id"])
                )
                processed += 1
            else:
                failed += 1

        if (i // args.batch_size + 1) % 5 == 0 or i + args.batch_size >= len(normal_content):
            elapsed = time.time() - start_time
            print(f"  Batch {i // args.batch_size + 1}: processed={processed}, failed={failed}, elapsed={elapsed:.1f}s")

    # Process empty content + reasoning messages
    print(f"\nProcessing {len(empty_content)} empty content + reasoning messages...")
    for i in range(0, len(empty_content), args.batch_size):
        batch = empty_content[i:i + args.batch_size]
        texts = [r["reasoning"] for r in batch]
        embeddings = compute_embedding_batch(texts)

        for row, emb in zip(batch, embeddings):
            if emb:
                cur.execute(
                    "UPDATE messages SET message_embedding = ? WHERE id = ?",
                    (emb, row["id"])
                )
                processed += 1
            else:
                failed += 1

        if (i // args.batch_size + 1) % 5 == 0 or i + args.batch_size >= len(empty_content):
            elapsed = time.time() - start_time
            print(f"  Batch {i // args.batch_size + 1}: processed={processed}, failed={failed}, elapsed={elapsed:.1f}s")

    conn.commit()
    elapsed = time.time() - start_time
    print(f"\nDone! Total: processed={processed}, failed={failed}, time={elapsed:.1f}s")

    # Verify
    cur.execute("SELECT COUNT(*) FROM messages WHERE message_embedding IS NULL")
    remaining = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM messages")
    total = cur.fetchone()[0]
    print(f"Remaining without embedding: {remaining}/{total} ({remaining/total*100:.1f}%)")

    conn.close()


if __name__ == "__main__":
    main()
