"""An owner-less dashboard import bounds the payload by the receiving store's ``sessions.import_max_*``.

With no gateway owner the dashboard writes through the offline store. Its pre-check and the write's own re-check both
read the store's profile config.yaml, so a narrowed key refuses a payload the defaults accept and 0 lets a payload over
a default land.
"""

import pytest
from fastapi import HTTPException

from gateway.owner_identity import host_owner_subject
from gateway.session_contract import SessionActor
from hermes_cli import web_server_sessions
from persistence.housekeeping.maintenance import IMPORT_MAX_SESSIONS
from persistence_assembly import open_persistence, profile_scope


def _offline_import(actor, request_id, sessions):
    return web_server_sessions._mutate_offline_store(None, actor, {
        "session_id": sessions[0]["id"], "request_id": request_id, "expected_revision": 0,
        "operation": "import", "payload": {"sessions": sessions}})


def test_offline_import_honours_the_profiles_import_limits(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config = tmp_path / "config.yaml"
    actor = SessionActor(host_owner_subject(), str(tmp_path),
                         frozenset({"session:read", "session:control", "session:create"}), "http")
    two_turns = [{"id": "limited", "source": "cli",
                  "messages": [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]}]
    config.write_text("sessions:\n  import_max_messages_per_session: 1\n")
    with pytest.raises(HTTPException) as refused:
        _offline_import(actor, "limited", two_turns)
    assert refused.value.status_code == 400
    assert refused.value.detail["errors"][0]["error"] == "messages exceeds the per-session import limit"

    many = [{"id": f"many-{i}", "source": "cli", "messages": []} for i in range(IMPORT_MAX_SESSIONS + 1)]
    # Different sizes: the config cache may key on mtime + size, and both writes can share a tick.
    config.write_text("sessions:\n  import_max_sessions: 0  # off\n")
    assert _offline_import(actor, "many-off", many)["imported"] == IMPORT_MAX_SESSIONS + 1
    with open_persistence(profile_scope(home=tmp_path)) as store:
        assert store.sessions.get("limited") is None
        assert store.sessions.get(f"many-{IMPORT_MAX_SESSIONS}") is not None
