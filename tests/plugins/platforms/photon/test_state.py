"""Persistent Photon state tests."""
from __future__ import annotations

import json
import stat
from datetime import datetime, timedelta, timezone
from pathlib import Path

from plugins.platforms.photon.state import PhotonStateStore


def _iso(seconds_ago: int = 0) -> str:
    dt = datetime.now(tz=timezone.utc) - timedelta(seconds=seconds_ago)
    return dt.isoformat().replace("+00:00", "Z")


def test_load_missing_state_returns_empty(tmp_path: Path) -> None:
    store = PhotonStateStore(tmp_path / "missing.json")

    state = store.load()

    assert state["sent_messages"] == {}
    assert state["last_inbound_by_chat"] == {}
    assert state["reactions"] == {}
    assert state["audit"] == []
    assert state["load_error"] is None


def test_load_corrupt_state_fails_open(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    path.write_text("{not-json", encoding="utf-8")
    store = PhotonStateStore(path)

    state = store.load()

    assert state["sent_messages"] == {}
    assert state["load_error"]


def test_load_supported_schema_keeps_existing_state(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    path.write_text(
        json.dumps({
            "schema_version": 1,
            "updated_at": _iso(),
            "sent_messages": {
                "msg-1": {"sent_at": _iso(), "kind": "text", "space_id": "space"}
            },
            "last_inbound_by_chat": {},
            "reactions": {},
            "audit": [],
        }),
        encoding="utf-8",
    )
    store = PhotonStateStore(path)

    state = store.load()

    assert state["load_error"] is None
    assert state["sent_messages"]["msg-1"]["space_id"] == "space"


def test_load_future_schema_warns_but_preserves_known_state(
    tmp_path: Path, caplog
) -> None:
    path = tmp_path / "state.json"
    path.write_text(
        json.dumps({
            "schema_version": 2,
            "updated_at": _iso(),
            "sent_messages": {
                "msg-1": {"sent_at": _iso(), "kind": "text", "space_id": "space"}
            },
            "last_inbound_by_chat": {},
            "reactions": {},
            "audit": [],
        }),
        encoding="utf-8",
    )
    store = PhotonStateStore(path)

    state = store.load()

    assert state["load_error"] is None
    assert state["sent_messages"]["msg-1"]["space_id"] == "space"
    assert "newer Photon state schema" in caplog.text


def test_load_invalid_schema_still_fails_open(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    path.write_text(
        json.dumps({
            "schema_version": 0,
            "sent_messages": {
                "msg-1": {"sent_at": _iso(), "kind": "text", "space_id": "space"}
            },
        }),
        encoding="utf-8",
    )
    store = PhotonStateStore(path)

    state = store.load()

    assert state["sent_messages"] == {}
    assert state["load_error"] == "unsupported schema version"


def test_record_methods_create_private_atomic_snapshot(tmp_path: Path) -> None:
    path = tmp_path / "plugins" / "photon" / "state.json"
    store = PhotonStateStore(path)
    store.load()

    store.record_sent_message(
        "msg-1", chat_key="+15551234567", space_id="any;-;+15551234567"
    )
    store.record_last_inbound("+15551234567", "inbound-1", space_id="any;-;+1555")
    store.record_reaction_added("any;-;+1555", "inbound-1", "like", "reaction-1")
    store.record_audit(
        action="send",
        status="failed",
        chat_key="+15551234567",
        message_id="msg-1",
        error_class="RuntimeError",
        error="x" * 1000,
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["sent_messages"]["msg-1"]["chat_key"] == "+15551234567"
    assert payload["last_inbound_by_chat"]["+15551234567"]["message_id"] == "inbound-1"
    assert store.reaction_for("any;-;+1555", "inbound-1")["reaction_id"] == "reaction-1"
    assert len(payload["audit"][0]["error"]) == 300
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_state_prunes_by_count_and_age(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    payload = {
        "schema_version": 1,
        "updated_at": _iso(),
        "sent_messages": {
            "old": {"sent_at": _iso(7200), "kind": "text"},
            "keep-1": {"sent_at": _iso(20), "kind": "text"},
            "keep-2": {"sent_at": _iso(10), "kind": "text"},
            "keep-3": {"sent_at": _iso(5), "kind": "text"},
        },
        "last_inbound_by_chat": {
            "old-chat": {"message_id": "old", "seen_at": _iso(7200)},
            "chat-1": {"message_id": "m1", "seen_at": _iso(10)},
        },
        "reactions": {},
        "audit": [
            {"at": _iso(3), "action": "send", "status": "started"},
            {"at": _iso(2), "action": "send", "status": "succeeded"},
            {"at": _iso(1), "action": "react", "status": "succeeded"},
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    store = PhotonStateStore(
        path,
        sent_max=2,
        last_inbound_max=10,
        audit_max=2,
        retention_seconds=3600,
    )

    state = store.load()

    assert set(state["sent_messages"]) == {"keep-2", "keep-3"}
    assert set(state["last_inbound_by_chat"]) == {"chat-1"}
    assert [item["action"] for item in state["audit"]] == ["send", "react"]


def test_reaction_removed_is_not_returned(tmp_path: Path) -> None:
    store = PhotonStateStore(tmp_path / "state.json")
    store.load()
    store.record_reaction_added("space", "message", "like", "reaction")

    store.record_reaction_removed("space", "message", succeeded=True)

    assert store.reaction_for("space", "message") is None
    assert store.health()["active_reactions"] == 0


def test_failed_reaction_removal_keeps_active_slot(tmp_path: Path) -> None:
    store = PhotonStateStore(tmp_path / "state.json")
    store.load()
    store.record_reaction_added("space", "message", "like", "reaction")

    store.record_reaction_removed("space", "message", succeeded=False)

    assert store.reaction_for("space", "message")["reaction_id"] == "reaction"
    assert store.health()["active_reactions"] == 1


def test_state_does_not_persist_message_content_or_secrets(tmp_path: Path) -> None:
    store = PhotonStateStore(tmp_path / "state.json")
    store.load()

    store.record_sent_message("msg-secret", chat_key="+1", space_id="space")
    store.record_audit(
        action="send",
        status="failed",
        chat_key="+1",
        message_id="msg-secret",
        error="sidecar rejected send",
    )

    raw = (tmp_path / "state.json").read_text(encoding="utf-8")
    assert "hello world" not in raw
    assert "PHOTON_PROJECT_SECRET" not in raw
    assert "test-project-secret" not in raw
    assert "attachment-bytes" not in raw
    assert "msg-secret" in raw


def test_write_failure_is_fail_open(tmp_path: Path, monkeypatch) -> None:
    store = PhotonStateStore(tmp_path / "state.json")
    store.load()

    def boom(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("plugins.platforms.photon.state.atomic_json_write", boom)

    store.record_sent_message("msg-1", chat_key="+1", space_id="space")

    assert store.write_error == "disk full"
    assert store.snapshot()["sent_messages"]["msg-1"]["space_id"] == "space"


def test_write_failure_is_visible_to_fresh_store(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "state.json"
    store = PhotonStateStore(path)
    store.load()

    def boom(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("plugins.platforms.photon.state.atomic_json_write", boom)

    store.record_sent_message("msg-1", chat_key="+1", space_id="space")

    fresh = PhotonStateStore(path)
    fresh.load()

    assert fresh.health()["write_error"] == "disk full"
    assert stat.S_IMODE(path.with_name("state.json.write_error").stat().st_mode) == 0o600


def test_successful_write_clears_persisted_write_failure(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "state.json"
    store = PhotonStateStore(path)
    store.load()

    def boom(*_args, **_kwargs):
        raise OSError("disk full")

    with monkeypatch.context() as mp:
        mp.setattr("plugins.platforms.photon.state.atomic_json_write", boom)
        store.record_sent_message("msg-1", chat_key="+1", space_id="space")

    recovered = PhotonStateStore(path)
    recovered.load()
    recovered.record_sent_message("msg-2", chat_key="+1", space_id="space")

    fresh = PhotonStateStore(path)
    fresh.load()

    assert fresh.health()["write_error"] is None
    assert store.load()["write_error"] is None
