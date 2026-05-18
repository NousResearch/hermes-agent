"""Tests for the local-only Blue canonical approval store facade."""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path


STORE_PATH = Path("/home/atlas/.hermes/blue/blue_approval_store.py")
PREFLIGHT_PATH = Path("/home/atlas/.hermes/scripts/blue_sweep_congruence_preflight.py")


def _load_store(tmp_path):
    spec = importlib.util.spec_from_file_location(f"blue_approval_store_test_{id(tmp_path)}", STORE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.BLUE_DIR = tmp_path
    module.APPROVAL_DB_PATH = tmp_path / "approval.sqlite"
    module.APPROVAL_INDEX_PATH = tmp_path / "approval-index.json"
    module.HANDLED_ACTIONS_PATH = tmp_path / "handled-actions.json"
    return module

def _load_preflight_module(tmp_path):
    spec = importlib.util.spec_from_file_location(f"blue_sweep_congruence_preflight_test_{id(tmp_path)}", PREFLIGHT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _capture_preflight_output(module) -> str:
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        module.main()
    return output.getvalue()


def _packet(approval_id="approval-1", draft_text="Hi Jane, thanks for reaching out."):
    draft_hash = "sha256:" + __import__("hashlib").sha256(draft_text.encode("utf-8")).hexdigest()
    return {
        "approval_id": approval_id,
        "schema_version": "blue.approval.v1",
        "source": {"source_type": "unit_test", "source_path": "/tmp/packet.json", "imported_at": "2026-05-14T00:00:00+00:00"},
        "brand": {"name": "Solar Renew"},
        "contact": {"contact_id": "contact-1", "name": "Jane"},
        "conversation": {"conversation_id": "conversation-1", "channel": "SMS", "latest_customer_at": "2026-05-14T00:00:00+00:00"},
        "send_target": {"contact_id": "contact-1", "conversation_id": "conversation-1", "channel": "SMS"},
        "draft": {"customer_facing": True, "draft_text": draft_text, "draft_hash": draft_hash},
    }


def test_store_upserts_pending_and_looks_up_by_key_and_alias(tmp_path):
    store = _load_store(tmp_path)
    packet = _packet()

    pending = store.upsert_pending({"packet": packet, "source_type": "unit_test", "source_links": {"artifact": "packet.json"}})
    repeat = store.upsert_pending({"packet": packet, "source_type": "unit_test", "source_links": {"artifact": "packet.json"}})

    assert pending["canonical_idempotency_key"].startswith("blue:v1:customer_message:solar-renew:contact-1:conversation-1:send_sms:")
    assert repeat["approval_id"] == pending["approval_id"]
    assert repeat["current_status"] == "pending_approval"
    assert store.lookup_by_key_or_alias(pending["canonical_idempotency_key"])["id"] == "approval-1"
    assert store.lookup_by_key_or_alias("conversation-1")["id"] == "approval-1"

    mirror = json.loads((tmp_path / "approval-index.json").read_text())
    assert mirror["idempotency_index"][pending["canonical_idempotency_key"]] == "approval-1"
    assert mirror["alias_index"]["conversation-1"] == "approval-1"


def test_terminal_and_superseded_states_mirror_handled_actions_without_sends(tmp_path):
    store = _load_store(tmp_path)
    pending = store.upsert_pending({"packet": _packet()})

    terminal = store.mark_superseded(pending["canonical_idempotency_key"], {"notes": "newer customer inbound changed the draft"})

    assert terminal["current_status"] == "superseded"
    handled = json.loads((tmp_path / "handled-actions.json").read_text())
    assert list(handled["handled_actions"].values())[0]["state"] == "superseded"
    assert list(handled["handled_actions"].values())[0]["evidence"]["notes"] == "newer customer inbound changed the draft"
    with sqlite3.connect(tmp_path / "approval.sqlite") as conn:
        assert conn.execute("SELECT final_state FROM handled_actions").fetchall() == [("superseded",)]


def test_action_requests_are_idempotent_and_stale_queued_requests_are_observable(tmp_path):
    store = _load_store(tmp_path)
    packet = _packet()
    pending = store.upsert_pending({"packet": packet})
    payload = {"packet": packet, "draft_hash": packet["draft"]["draft_hash"]}

    first = store.record_action_request("approval-1", "reverify", payload)
    second = store.record_action_request("approval-1", "reverify", payload)

    assert second["duplicate_of_existing"] is True
    assert second["request_id"] == first["request_id"]
    old = (datetime.now(timezone.utc) - timedelta(minutes=30)).isoformat()
    with sqlite3.connect(tmp_path / "approval.sqlite") as conn:
        conn.execute("UPDATE approval_action_requests SET updated_at = ? WHERE request_id = ?", (old, first["request_id"]))
    stale = store.stale_queued_requests(threshold_minutes=15)

    assert stale == [
        {
            "request_id": first["request_id"],
            "approval_id": "approval-1",
            "action_type": "reverify",
            "status": "queued_for_reverify",
            "updated_at": old,
            "canonical_idempotency_key": pending["canonical_idempotency_key"],
            "stale_minutes": 15,
        }
    ]


def test_upsert_pending_suppresses_duplicate_canonical_key_after_terminal_state(tmp_path):
    store = _load_store(tmp_path)
    original = store.upsert_pending({"packet": _packet("approval-original")})
    store.mark_terminal(original["canonical_idempotency_key"], "handled_sent", {"notes": "message already sent"})

    duplicate = store.upsert_pending({"packet": _packet("approval-duplicate")})

    assert duplicate["approval_id"] == "approval-duplicate"
    assert duplicate["current_status"] == "superseded"
    assert duplicate["freshness"]["duplicate"] is True
    assert duplicate["freshness"]["stale"] is True
    assert duplicate["freshness"]["existing_status"] == "handled_sent"
    assert "handled_sent" in duplicate["freshness"]["reason"]
    assert duplicate["source_links"]["duplicate_of_approval_id"] == "approval-original"
    assert duplicate["source_links"]["duplicate_canonical_key"] == original["canonical_idempotency_key"]
    with sqlite3.connect(tmp_path / "approval.sqlite") as conn:
        rows = conn.execute("SELECT approval_id, current_status FROM approval_index ORDER BY approval_id").fetchall()
    assert rows == [("approval-duplicate", "superseded"), ("approval-original", "handled_sent")]


def test_preflight_classifies_sqlite_pending_approval_rows_as_active(tmp_path, monkeypatch):
    preflight = _load_preflight_module(tmp_path)
    monkeypatch.setattr(preflight, "load_json", lambda path, default: default)
    monkeypatch.setattr(
        preflight,
        "sqlite_approval_rows",
        lambda: [
            {
                "kind": "sqlite-approval-index",
                "id": "approval-pending",
                "key": "blue:v1:customer_message:solar-renew:contact-1:conversation-1:send_sms:hash:2026-05-14",
                "state": "pending_approval",
                "brand": "solar-renew",
                "contact": "contact-1",
                "conversation": "conversation-1",
                "action": "send_sms",
                "updated_at": "2026-05-14T00:00:00+00:00",
            }
        ],
    )

    output = _capture_preflight_output(preflight)

    active_section = output.split("## Active/pending/held items to avoid duplicating", 1)[1].split("## Already handled/superseded/rejected items", 1)[0]
    assert "state=pending_approval id=approval-pending" in active_section
    assert "## Other state rows" not in output
