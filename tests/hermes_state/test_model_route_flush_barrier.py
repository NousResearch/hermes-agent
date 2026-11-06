"""Route writes must not overtake accounting a failed flush could not drain."""
import json
from types import SimpleNamespace

import pytest

from hermes_state import SessionDB


@pytest.mark.parametrize("writer", ["model", "meta", "billing"])
def test_route_writer_refuses_failed_flush_and_preserves_queued_usage(tmp_path, monkeypatch, writer):
    with SessionDB(tmp_path / "state.db") as db:
        db.create_session("s", "cli", model="old")
        db._token_writer_thread = SimpleNamespace(is_alive=lambda: True)
        db.queue_token_counts("s", model="old", billing_provider="old-provider", api_call_count=1, input_tokens=7)
        db._token_writer_thread = None
        before = tuple(db._read_one("SELECT model,model_config,billing_provider FROM sessions WHERE id='s'"))
        real_flush = db.flush_token_counts
        monkeypatch.setattr(db, "flush_token_counts", lambda *a, **k: False)
        with pytest.raises(TimeoutError):
            if writer == "model":
                db.update_session_model("s", "new", provider="new-provider")
            elif writer == "meta":
                db.update_session_meta("s", json.dumps({"model": "new", "provider": "new-provider"}), "new")
            else:
                db.update_session_billing_route("s", provider="new-provider", base_url="http://127.0.0.1:9/v1")
        assert tuple(db._read_one("SELECT model,model_config,billing_provider FROM sessions WHERE id='s'")) == before
        monkeypatch.setattr(db, "flush_token_counts", real_flush)
        assert db.flush_token_counts()
        assert db.get_session("s")["input_tokens"] == 7


def test_model_only_retry_uses_fresh_patch(tmp_path, monkeypatch):
    import hermes_state_lineage

    with SessionDB(tmp_path / "state.db") as db:
        db.create_session("s", "cli", model="old")
        db.update_session_model("s", "old", provider="first")
        def retry_with_changed_row(db, session_id, write, **kwargs):
            conn = db._conn
            conn.execute("SAVEPOINT first_attempt")
            write(conn)
            conn.execute("ROLLBACK TO first_attempt")
            conn.execute("RELEASE first_attempt")
            conn.execute("UPDATE sessions SET model_config=? WHERE id=?", ('{}', session_id))
            write(conn)
            conn.commit()
        monkeypatch.setattr(hermes_state_lineage, "guarded_session_write", retry_with_changed_row)
        db.update_session_model("s", "new")
        config = json.loads(db.get_session("s")["model_config"])
        assert config == {"model": "new"}


def test_tip_model_update_refuses_ambiguous_child_without_partial_write(tmp_path, monkeypatch):
    from hermes_state_lineage import SessionLineageAmbiguousError
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    profile = tmp_path / "profiles" / "science"
    profile.mkdir(parents=True)
    with SessionDB(tmp_path / "state.db") as local, SessionDB(profile / "state.db") as sibling:
        local.create_session("parent", "cli", model="old")
        local.publish_compression_child(parent_session_id="parent", child_session_id="child", source="cli",
                                        messages=[{"role": "user", "content": "summary"}],
                                        require_compression_lease=False)
        sibling._conn.execute("INSERT INTO sessions(id,source,started_at,profile_name) VALUES ('child','cli',1,'science')")
        sibling._conn.commit()
        before = (dict(local.get_session("parent")), dict(local.get_session("child")), dict(sibling.get_session("child")))
        with pytest.raises(SessionLineageAmbiguousError):
            local.update_session_model("parent", "picked", provider="openrouter", resolve_tip=True)
        assert (dict(local.get_session("parent")), dict(local.get_session("child")), dict(sibling.get_session("child"))) == before
