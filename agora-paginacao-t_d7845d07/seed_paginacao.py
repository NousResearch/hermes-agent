#!/usr/bin/env python3
"""Seed >PAGE_SIZE messages in an isolated Ágora test channel.

Uses direct SQLite writes to agora.db so we don't spam real channels.
The channel is named ``qa-paginacao`` and is cleaned up/recycled on each run.
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import sys
sys.path.insert(0, '/home/felipi/.hermes/hermes-agent')
from hermes_constants import get_default_hermes_root

DB_PATH = get_default_hermes_root() / "agora.db"
CHANNEL_SLUG = "qa-paginacao"
CHANNEL_NAME = "QA Paginação"
MESSAGE_COUNT = 150
PAGE_SIZE = 50


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def seed() -> dict:
    conn = _connect()
    now = int(time.time())

    # Recycle test channel if it exists
    existing = conn.execute(
        "SELECT id FROM agora_channels WHERE slug = ?", (CHANNEL_SLUG,)
    ).fetchone()
    if existing:
        channel_id = existing["id"]
        conn.execute("DELETE FROM agora_messages WHERE channel_id = ?", (channel_id,))
        conn.execute("DELETE FROM agora_threads WHERE channel_id = ?", (channel_id,))
        conn.execute("DELETE FROM agora_events WHERE entity_id = ?", (str(channel_id),))
    else:
        cur = conn.execute(
            "INSERT INTO agora_channels (slug, name, description, created_at) VALUES (?, ?, ?, ?)",
            (CHANNEL_SLUG, CHANNEL_NAME, "Canal isolado para QA de paginação.", now),
        )
        channel_id = cur.lastrowid

    # Insert messages with strictly increasing created_at/id so ordering is predictable.
    base_time = now - (MESSAGE_COUNT * 60)
    records = []
    for i in range(MESSAGE_COUNT):
        created_at = base_time + (i * 60)
        body = f"msg-{i:03d} — mensagem de teste {i + 1}/{MESSAGE_COUNT}"
        author_type = "agent" if i % 2 == 0 else "human"
        author_profile = "agora-qa" if author_type == "agent" else "human"
        records.append((channel_id, author_type, author_profile, body, created_at))

    conn.executemany(
        "INSERT INTO agora_messages (channel_id, author_type, author_profile, body, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        records,
    )
    conn.commit()

    total = conn.execute(
        "SELECT COUNT(*) FROM agora_messages WHERE channel_id = ?", (channel_id,)
    ).fetchone()[0]
    conn.close()

    return {
        "channel_id": channel_id,
        "channel_slug": CHANNEL_SLUG,
        "channel_name": CHANNEL_NAME,
        "message_count": MESSAGE_COUNT,
        "page_size": PAGE_SIZE,
        "actual_total": total,
    }


if __name__ == "__main__":
    import json
    print(json.dumps(seed(), indent=2, ensure_ascii=False))
