"""An owner-less dashboard delete records the deleted session's hosted documents as retired media.

The offline store's delete gets the profile's native-inputs root, so a hosted document under it that an admission of
the deleted session references becomes a retired-media candidate, as a delete through the gateway owner records.
"""

import hashlib
import json

from gateway.owner_identity import host_owner_subject
from gateway.session_contract import SessionActor
from gateway.session_ingress_media import native_inputs_root
from hermes_cli import web_server_sessions
from persistence_assembly import open_persistence, profile_scope
from tests.fixtures.storage_holder import raw_holder
from tests.fixtures.storage_seed import seed_session

ADMISSION_SQL = (
    "INSERT INTO session_admissions (admission_id, request_id, principal_id, target_session_id, lineage_json,"
    " payload_json, payload_digest, intent, status, outcome)"
    " VALUES (?, ?, 'principal', ?, '[]', ?, 'digest', 'queue', 'terminal', 'completed')"
)


def test_offline_delete_records_hosted_documents_under_the_native_inputs_root(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    document = str(native_inputs_root() / hashlib.sha256(b"doc").hexdigest() / "doc.txt")
    text = f"shared\n[Shared attachment] file: {document}\n"
    with open_persistence(profile_scope(home=tmp_path)) as store:
        seed_session(store, "hosted", "desktop")
        raw_holder(store)._execute_write(lambda conn: conn.execute(
            ADMISSION_SQL, ("adm-hosted", "hosted:room:hosted", "hosted", json.dumps({"text": text}))))

    actor = SessionActor(host_owner_subject(), str(tmp_path), frozenset({"session:read", "session:control"}), "http")
    result = web_server_sessions._mutate_offline_store(None, actor, {
        "session_id": "hosted", "request_id": "delete", "expected_revision": 0, "expected_generation": 0,
        "operation": "delete", "payload": {}})

    assert result["deleted_ids"] == ["hosted"]
    with open_persistence(profile_scope(home=tmp_path)) as store:
        assert store.sessions.get("hosted") is None
        assert [value["path"] for _key, value in store.admissions.retired_candidates("media")] == [document]
