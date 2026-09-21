"""End-to-end test of the canonical Postiz publish pipeline.

Runs the real ``publish_to_postiz.publish_approved_drafts()`` flow against a
disposable SQLite DB with a mocked ``queue_post`` adapter (sandbox Postiz). The
mock stands in for the Postiz DB insert; everything else — draft discovery,
G03 gate, delivery tracker (idempotency/retry/DLQ/events) — is exercised for real.

This is the disposable-target end-to-end test: it proves the entire pipeline
up to mock publication without touching Sahil's personal X/LinkedIn accounts.
"""

import json
import sqlite3
import uuid
from pathlib import Path

import pytest


def _make_db(path: Path, rows: list[dict]) -> None:
    conn = sqlite3.connect(str(path))
    conn.execute(
        """CREATE TABLE drafts (
            id TEXT PRIMARY KEY, brand TEXT, platform TEXT,
            content_type TEXT, pillar TEXT, topic TEXT, title TEXT,
            body_text TEXT, visual_path TEXT, ai_image_path TEXT,
            status TEXT, approved_at TEXT, published_at TEXT,
            postiz_id TEXT, enqueue_state TEXT
        )"""
    )
    for r in rows:
        conn.execute(
            """INSERT INTO drafts
               (id, brand, platform, content_type, pillar, topic, title,
                body_text, visual_path, ai_image_path, status, approved_at,
                published_at, postiz_id, enqueue_state)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                r["id"], r.get("brand", "sahil_twitter"),
                r.get("platform", "twitter"), "text", "pillar", "topic",
                r.get("title", "T"), r["body"], None, None,
                r.get("status", "approved"), r.get("approved_at", "2026-01-01T00:00:00+00:00"),
                r.get("published_at"), r.get("postiz_id"), r.get("enqueue_state"),
            ),
        )
    conn.commit()
    conn.close()


def _load_module(monkeypatch, db_path, events_path):
    monkeypatch.setenv("CONTENT_ENGINE_DB_PATH", str(db_path))
    monkeypatch.setenv("PUBLISH_EVENTS_PATH", str(events_path))
    monkeypatch.delenv("DISCORD_OPS_CHANNEL_ID", raising=False)
    monkeypatch.delenv("DISCORD_CONTENT_CHANNEL_ID", raising=False)
    ce_root = Path(__file__).resolve().parents[1]
    import sys

    if str(ce_root) not in sys.path:
        sys.path.insert(0, str(ce_root))
    # The `database` module resolves DB_PATH at import time and is cached in
    # sys.modules across the suite; pin it to this test's DB so mark_published()
    # writes to the disposable DB regardless of which test ran first.
    import database as _db_mod

    monkeypatch.setattr(_db_mod, "DB_PATH", db_path)
    # Load a FRESH module instance (unique name) so the module-level DB_PATH,
    # which is read from the env at import time, matches this test's DB. Reusing
    # the cached module across tests would point publish_approved_drafts at a
    # stale DB from a previous test.
    import importlib.util

    src = ce_root / "publish_to_postiz.py"
    mod_name = f"publish_to_postiz_e2e_{uuid.uuid4().hex[:8]}"
    spec = importlib.util.spec_from_file_location(mod_name, src)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not build spec for publish_to_postiz.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _delivery_status(db_path, draft_id):
    from publish_tracker import PublishTracker

    tr = PublishTracker(db_path=str(db_path))
    return tr.status(draft_id)


def test_e2e_approved_draft_publishes_once_through_mock(monkeypatch, tmp_path):
    db = tmp_path / "content_engine.db"
    events = tmp_path / "events.jsonl"
    _make_db(db, [{"id": "d-e2e", "brand": "sahil_twitter", "platform": "twitter",
                   "body": "Hello world #buildinpublic"}])

    mod = _load_module(monkeypatch, db, events)
    monkeypatch.setattr(mod, "gate_publish", lambda draft_id: True)
    calls = {"n": 0}

    def fake_queue_post(**kw):
        calls["n"] += 1
        return f"postiz-e2e-{calls['n']}"

    monkeypatch.setattr(mod, "queue_post", fake_queue_post)

    assert mod.publish_approved_drafts() == 1
    assert calls["n"] == 1  # exactly one enqueue attempt

    # Delivery tracker recorded terminal 'published' with the idempotency key.
    st = _delivery_status(db, "d-e2e")
    assert st is not None
    assert st["status"] == "published"
    assert st["idempotency_key"] == "d-e2e::twitter"

    # Draft is marked published + enqueued in the canonical drafts table.
    conn = sqlite3.connect(str(db))
    row = conn.execute(
        "SELECT status, enqueue_state, postiz_id FROM drafts WHERE id = 'd-e2e'"
    ).fetchone()
    conn.close()
    assert row[0] == "published"
    assert row[1] == "enqueued"
    assert row[2] == "postiz-e2e-1"

    # Event log persisted with a published event.
    lines = events.read_text().strip().splitlines()
    statuses = [json.loads(l)["status"] for l in lines]
    assert "published" in statuses


def test_e2e_gate_blocks_publication(monkeypatch, tmp_path):
    db = tmp_path / "content_engine.db"
    events = tmp_path / "events.jsonl"
    _make_db(db, [{"id": "d-blocked", "brand": "sahil_twitter", "platform": "twitter",
                   "body": "should not publish"}])

    mod = _load_module(monkeypatch, db, events)
    # gate_publish denies -> the draft must NOT be published.
    monkeypatch.setattr(mod, "gate_publish", lambda draft_id: False)
    monkeypatch.setattr(mod, "queue_post", lambda **kw: pytest.fail("queue_post must not be called"))

    assert mod.publish_approved_drafts() == 0

    conn = sqlite3.connect(str(db))
    status = conn.execute("SELECT status FROM drafts WHERE id = 'd-blocked'").fetchone()[0]
    conn.close()
    assert status == "approved"  # unchanged — not published


def test_e2e_transient_failure_retries_then_publishes(monkeypatch, tmp_path):
    db = tmp_path / "content_engine.db"
    events = tmp_path / "events.jsonl"
    _make_db(db, [{"id": "d-retry", "brand": "sahil_twitter", "platform": "twitter",
                   "body": "retry me"}])

    mod = _load_module(monkeypatch, db, events)
    monkeypatch.setattr(mod, "gate_publish", lambda draft_id: True)

    # First two publish_approved_drafts() cycles fail (Postiz down), third succeeds.
    cycles = {"n": 0}

    def flaky_queue_post(**kw):
        cycles["n"] += 1
        if cycles["n"] <= 2:
            raise RuntimeError("postiz unreachable")
        return "postiz-retried"

    monkeypatch.setattr(mod, "queue_post", flaky_queue_post)

    # Cycle 1: fails, attempt 1 recorded, draft released to pending.
    assert mod.publish_approved_drafts() == 0
    # Cycle 2: fails, attempt 2 recorded.
    assert mod.publish_approved_drafts() == 0
    # Cycle 3: succeeds -> attempt 3 publishes.
    assert mod.publish_approved_drafts() == 1

    st = _delivery_status(db, "d-retry")
    assert st["status"] == "published"
    assert st["attempt_count"] == 3

    # No dead-letter (it eventually succeeded) and exactly one published.
    conn = sqlite3.connect(str(db))
    n_dl = conn.execute("SELECT COUNT(*) FROM dead_letter_queue").fetchone()[0]
    conn.close()
    assert n_dl == 0
