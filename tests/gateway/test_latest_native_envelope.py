"""Admission lookups that used to decode a chat's whole ledger on the gateway loop."""
import json

from hermes_state import SessionDB
from hermes_state_runtime import (
    find_session_admission, latest_native_envelope, list_principal_admissions,
)


def _insert(conn, *, session_id, admission_id, request_id, principal_id, payload, status="terminal"):
    conn.execute(
        "INSERT INTO session_admissions (admission_id, request_id, principal_id, target_session_id, "
        "lineage_json, payload_json, payload_digest, intent, status) VALUES (?,?,?,?,?,?,?,?,?)",
        (admission_id, request_id, principal_id, session_id, "{}", json.dumps(payload),
         admission_id, "queue", status))


def test_latest_native_envelope_is_the_newest_row_that_carries_provenance(tmp_path):
    db = SessionDB(tmp_path / "state.db")

    def load(conn):
        conn.execute("INSERT INTO sessions (id, source, started_at) VALUES ('chat', 'slack', 1)")
        for i in range(3):
            _insert(conn, session_id="chat", admission_id=f"old{i}", request_id=f"r{i}",
                    principal_id="messaging:human",
                    payload={"text": "earlier", "native_text_v1": {"provenance": {"n": i}}})
        _insert(conn, session_id="chat", admission_id="keep", request_id="keep",
                principal_id="messaging:human",
                payload={"text": "human", "native_text_v1": {"provenance": {"ok": True}, "marker": "keep"}})
        _insert(conn, session_id="chat", admission_id="noise", request_id="noise",
                principal_id="automation:chat", payload={"text": "not native"})

    db._execute_write(load)
    assert latest_native_envelope(db, session_id="chat")["marker"] == "keep"
    assert latest_native_envelope(db, session_id="missing") is None


def test_latest_native_envelope_stops_when_the_newest_native_row_has_no_provenance(tmp_path):
    db = SessionDB(tmp_path / "state.db")

    def load(conn):
        conn.execute("INSERT INTO sessions (id, source, started_at) VALUES ('chat', 'slack', 1)")
        _insert(conn, session_id="chat", admission_id="old", request_id="old",
                principal_id="messaging:human",
                payload={"text": "old", "native_text_v1": {"provenance": {"ok": True}}})
        _insert(conn, session_id="chat", admission_id="new", request_id="new",
                principal_id="messaging:human",
                payload={"text": "new", "native_text_v1": {"event": {}}})

    db._execute_write(load)
    assert latest_native_envelope(db, session_id="chat") is None


def test_principal_listing_skips_other_principals_and_keeps_sibling_payloads(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    sibling = json.dumps(["completion", "sibling"])

    def load(conn):
        conn.execute("INSERT INTO sessions (id, source, started_at) VALUES ('chat', 'slack', 1)")
        _insert(conn, session_id="chat", admission_id="human", request_id="m1",
                principal_id="messaging:human",
                payload={"text": "hi", "native_text_v1": {"provenance": {"ok": True}}})
        _insert(conn, session_id="chat", admission_id="batch", request_id="primary",
                principal_id="automation:chat",
                payload={"text": "done", "native_text_v1": {"automation": {
                    "identity": "primary", "identities": ["primary", sibling]}}})

    db._execute_write(load)
    rows = list_principal_admissions(db, principal_id="automation:chat", session_id="chat")
    assert [row["admission_id"] for row in rows] == ["batch"]
    assert find_session_admission(
        db, principal_id="automation:chat", session_id="chat", request_id=sibling) is None
    named = rows[0]["payload"]["native_text_v1"]["automation"]["identities"]
    assert sibling in named
