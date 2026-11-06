"""Adoption must never retire a donor whose transcript the target does not preserve."""
import pytest
from hermes_state import SessionDB

@pytest.fixture
def stores(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    donor_path = home / "state.db"
    target_path = home / "profiles" / "science" / "state.db"
    target_path.parent.mkdir(parents=True)
    with SessionDB(donor_path) as donor, SessionDB(target_path) as target:
        yield donor, target

def snapshot(db):
    tables = ("sessions", "messages", "state_meta", "gateway_routing", "adoption_transfers")
    names = {r[0] for r in db._conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    return {table: [tuple(r) for r in db._conn.execute(f"SELECT * FROM {table} ORDER BY 1")]
            for table in tables if table in names}

def copy_existing(donor, target):
    target._conn.execute("INSERT INTO sessions(id,source,started_at) VALUES ('same','cli',1)")
    target._conn.execute("INSERT INTO sessions(id,source,started_at) VALUES ('padding','cli',1)")
    target._conn.execute("INSERT INTO messages(session_id,role,content,timestamp) VALUES ('padding','user','padding',1)")
    columns = [r[1] for r in donor._conn.execute("PRAGMA table_info(messages)") if r[1] != "id"]
    sql = f"INSERT INTO messages({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})"
    rows = donor._conn.execute(f"SELECT {','.join(columns)} FROM messages WHERE session_id='same' ORDER BY id")
    target._conn.executemany(sql, [tuple(r) for r in rows])
    target._conn.commit()

@pytest.mark.parametrize("change", ["content", "shorter", "tool_calls", "reasoning", "display_metadata"])
def test_preexisting_divergence_refuses_without_any_mutation(stores, change):
    donor, target = stores
    donor.create_session("same", "cli")
    donor.append_message("same", "assistant", "answer",
        tool_calls=[{"id": "call", "type": "function", "function": {"name": "search", "arguments": "{}"}}])
    donor.append_message("same", "user", "next")
    copy_existing(donor, target)
    if change == "shorter":
        target._conn.execute("DELETE FROM messages WHERE session_id='same' AND role='user'")
    else:
        field, value = {
            "content": ("content", "different"),
            "tool_calls": ("tool_calls", '[{"id":"changed"}]'),
            "reasoning": ("reasoning_content", "changed reasoning"),
            "display_metadata": ("display_metadata", '{"changed":true}'),
        }[change]
        target._conn.execute(f"UPDATE messages SET {field}=? WHERE session_id='same' AND role='assistant'", (value,))
    target._conn.commit()
    before = snapshot(donor), snapshot(target)
    result = target.adopt_session_lineage_from(donor, "same")
    assert not result.get("donor_retired"), result
    assert (snapshot(donor), snapshot(target)) == before

def test_remapped_matching_prefix_and_target_extension_are_valid(stores):
    donor, target = stores
    donor.create_session("same", "cli")
    donor.append_message("same", "user", "kept")
    copy_existing(donor, target)
    assert donor.get_messages("same")[0]["id"] != target.get_messages("same")[0]["id"]
    target._conn.execute("INSERT INTO messages(session_id,role,content,timestamp) VALUES ('same','assistant','extension',2)")
    target._conn.commit()
    result = target.adopt_session_lineage_from(donor, "same")
    assert result["donor_retired"], result
    assert [r["content"] for r in target.get_messages("same")] == ["kept", "extension"]

def test_fresh_target_copy_and_retirement_preserves_history(stores):
    donor, target = stores
    donor.create_session("same", "cli")
    donor.append_message("same", "user", "kept")
    result = target.adopt_session_lineage_from(donor, "same")
    assert result["donor_retired"], result
    assert target.get_messages("same")[0]["content"] == "kept"


def test_unrepresentable_donor_sidecar_rolls_back_fresh_copy(stores):
    donor, target = stores
    donor.create_session("same", "cli")
    donor.append_message("same", "user", "kept")
    # The existing import writer clears user-row reasoning, so this copy would lose a sidecar.
    donor._conn.execute("UPDATE messages SET reasoning_content='retained sidecar' WHERE session_id='same'")
    donor._conn.commit()
    before = snapshot(donor), snapshot(target)
    result = target.adopt_session_lineage_from(donor, "same")
    assert not result.get("donor_retired"), result
    assert (snapshot(donor), snapshot(target)) == before

def test_donor_change_before_retirement_keeps_donor_active(stores, monkeypatch):
    donor, target = stores
    donor.create_session("same", "cli")
    donor.append_message("same", "user", "before")
    original = SessionDB._retire_donor_segment
    def changed(self, source, sid):
        source._conn.execute("UPDATE messages SET content='changed' WHERE session_id=?", (sid,))
        source._conn.commit()
        return original(self, source, sid)
    monkeypatch.setattr(SessionDB, "_retire_donor_segment", changed)
    result = target.adopt_session_lineage_from(donor, "same")
    assert not result["donor_retired"]
    assert donor.get_session("same")["ended_at"] is None
    assert donor.get_messages("same")[0]["content"] == "changed"
    assert target.get_messages("same")[0]["content"] == "before"


def test_json_boolean_and_number_are_distinct_sidecars(stores):
    donor, target = stores
    donor.create_session("same", "cli")
    donor.append_message("same", "user", "kept")
    donor._conn.execute("UPDATE messages SET display_metadata=? WHERE session_id='same'", ('{"x":true}',))
    donor._conn.commit()
    copy_existing(donor, target)
    target._conn.execute("UPDATE messages SET display_metadata=? WHERE session_id='same'", ('{"x":1}',))
    target._conn.commit()
    before = snapshot(donor), snapshot(target)
    result = target.adopt_session_lineage_from(donor, "same")
    assert not result.get("donor_retired"), result
    assert (snapshot(donor), snapshot(target)) == before

def test_equivalent_structured_content_encoding_is_compatible(stores):
    import json
    donor, target = stores
    donor.create_session("same", "cli")
    donor.append_message("same", "user", [{"type": "text", "text": "hello"}])
    copy_existing(donor, target)
    encoded = "\x00json:" + json.dumps([{"type": "text", "text": "hello"}], separators=(",", ":"))
    target._conn.execute("UPDATE messages SET content=? WHERE session_id='same'", (encoded,))
    target._conn.commit()
    result = target.adopt_session_lineage_from(donor, "same")
    assert result["donor_retired"], result
