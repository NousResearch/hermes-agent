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
    tables = ("sessions", "messages", "session_model_usage", "system_prompts", "state_meta",
              "gateway_routing", "adoption_transfers")
    names = {r[0] for r in db._conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    return {table: [tuple(r) for r in db._conn.execute(f"SELECT * FROM {table} ORDER BY 1")]
            for table in tables if table in names}

def copy_existing(donor, target):
    target._conn.execute("INSERT INTO sessions(id,source,started_at,created_source) VALUES ('same','cli',1,?)",
                         (donor.get_session("same")["created_source"],))
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


@pytest.mark.parametrize("busy_owner", ["donor", "target"])
def test_adoption_refuses_queued_writer_that_is_busy_before_guard(stores, monkeypatch, busy_owner):
    import threading
    import time
    from contextlib import contextmanager
    import hermes_state_portability as portability
    donor, target = stores
    donor.create_session("same", "cli")
    donor.append_message("same", "user", "kept")
    if busy_owner == "target":
        target.create_session("other", "cli")
    owner = donor if busy_owner == "donor" else target
    owner_sid = "same" if busy_owner == "donor" else "other"
    before = snapshot(donor), snapshot(target)
    started, release = threading.Event(), threading.Event()
    original_apply = owner._apply_token_batch
    original_guard = portability.session_lineage_guard
    def delayed_batch(batch):
        started.set()
        assert release.wait(15), "adoption did not release the waiting writer"
        return original_apply(batch)
    @contextmanager
    def queue_before_guard(path, ids, **kwargs):
        owner.queue_token_counts(owner_sid, input_tokens=7, model="m")
        assert started.wait(5)
        with original_guard(path, ids, **kwargs) as observed:
            yield observed
    monkeypatch.setattr(owner, "_apply_token_batch", delayed_batch)
    monkeypatch.setattr(portability, "session_lineage_guard", queue_before_guard)
    try:
        start = time.monotonic()
        result = target.adopt_session_lineage_from(donor, "same")
        elapsed = time.monotonic() - start
        assert elapsed < 3, elapsed
        assert not result.get("donor_retired"), result
        assert not result.get("adopted"), result
        assert (snapshot(donor), snapshot(target)) == before
    finally:
        release.set()
        assert owner.flush_token_counts(timeout=10)
    assert owner.get_session(owner_sid)["input_tokens"] == 7


@pytest.mark.parametrize("busy_owner", ["donor", "target"])
def test_adoption_refuses_unflushed_accounting_before_guard(stores, monkeypatch, busy_owner):
    donor, target = stores
    donor.create_session("same", "cli")
    donor.append_message("same", "user", "kept")
    before = snapshot(donor), snapshot(target)
    owner = donor if busy_owner == "donor" else target
    monkeypatch.setattr(owner, "flush_token_counts", lambda *a, **k: False)
    result = target.adopt_session_lineage_from(donor, "same")
    assert not result.get("adopted") and not result.get("donor_retired"), result
    assert (snapshot(donor), snapshot(target)) == before


def _seed_portable_accounting(db):
    db.create_session("same", "cli", system_prompt="original prompt")
    db.append_message("same", "user", "kept")
    db.update_session_tool_names("same", [{"name": "search"}])
    db.update_token_counts("same", input_tokens=7, model="m", billing_provider="p",
                           billing_base_url="https://provider.invalid/v1", api_call_count=1)
    db.record_auxiliary_usage("same", "vision", model="vision", billing_provider="p", input_tokens=3)

def _copy_portable_existing(donor, target):
    payload = donor.export_session("same", include_inactive=True)
    result = target.import_sessions([payload], transfer_from=donor.db_path)
    assert result["imported"] == 1
    source = donor.get_session("same")
    pin = source["tool_names"]
    target._execute_write(lambda conn: conn.execute(
        "UPDATE sessions SET tool_names=?,created_source=?,title_source=? WHERE id='same'",
        (target._store_system_prompt(conn, pin), source["created_source"], source["title_source"])))
    columns = [r[1] for r in donor._conn.execute("PRAGMA table_info(session_model_usage)")]
    values = [tuple(row) for row in donor._conn.execute("SELECT * FROM session_model_usage WHERE session_id='same'")]
    target._conn.executemany(
        f"INSERT INTO session_model_usage({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})", values)
    target._conn.commit()

@pytest.mark.parametrize("reason", ["missing_usage", "lower_summary", "lower_usage", "prompt", "tool_pin", "first_seen", "billing_route"])
def test_existing_copy_requires_portable_accounting_and_context(stores, reason):
    donor, target = stores
    _seed_portable_accounting(donor)
    _copy_portable_existing(donor, target)
    updates = {
        "missing_usage": lambda conn: conn.execute("DELETE FROM session_model_usage WHERE session_id='same' AND task='vision'"),
        "lower_summary": lambda conn: conn.execute("UPDATE sessions SET input_tokens=6 WHERE id='same'"),
        "lower_usage": lambda conn: conn.execute(
            "UPDATE session_model_usage SET input_tokens=6 WHERE session_id='same' AND task=''"),
        "prompt": lambda conn: conn.execute("UPDATE sessions SET system_prompt_hash=? WHERE id='same'",
                                           (target._store_system_prompt(conn, "chosen other prompt"),)),
        "tool_pin": lambda conn: conn.execute("UPDATE sessions SET tool_names=? WHERE id='same'",
                                             (target._store_system_prompt(conn, '[{"name":"other"}]'),)),
        "first_seen": lambda conn: conn.execute(
            "UPDATE session_model_usage SET first_seen=first_seen+1 WHERE session_id='same' AND task=''"),
        "billing_route": lambda conn: conn.execute(
            "UPDATE sessions SET billing_base_url='https://other.invalid/v1' WHERE id='same'"),
    }
    target._execute_write(updates[reason])
    before = snapshot(donor), snapshot(target)
    result = target.adopt_session_lineage_from(donor, "same")
    assert not result.get("donor_retired"), result
    assert (snapshot(donor), snapshot(target)) == before

def test_fresh_copy_keeps_main_aux_usage_and_context_on_retry(stores, monkeypatch):
    donor, target = stores
    _seed_portable_accounting(donor)
    source_usage = [tuple(r) for r in donor._conn.execute("SELECT * FROM session_model_usage ORDER BY task")]
    with monkeypatch.context() as patch:
        patch.setattr(SessionDB, "_retire_donor_segment", lambda *_: False)
        first = target.adopt_session_lineage_from(donor, "same")
    assert first["adopted"] and not first["donor_retired"], first
    assert [tuple(r) for r in target._conn.execute("SELECT * FROM session_model_usage ORDER BY task")] == source_usage
    assert target.get_session("same")["tool_names"] == donor.get_session("same")["tool_names"]
    assert target.get_session("same")["system_prompt"] == "original prompt"
    retry = target.adopt_session_lineage_from(donor, "same")
    assert retry["donor_retired"], retry
    assert [tuple(r) for r in target._conn.execute("SELECT * FROM session_model_usage ORDER BY task")] == source_usage

def test_existing_matching_accounting_extension_is_not_overwritten(stores):
    donor, target = stores
    _seed_portable_accounting(donor)
    _copy_portable_existing(donor, target)
    target._conn.execute("UPDATE sessions SET input_tokens=input_tokens+3,api_call_count=api_call_count+1 WHERE id='same'")
    target._conn.execute("UPDATE session_model_usage SET input_tokens=input_tokens+3,api_call_count=api_call_count+1,"
                         "last_seen=last_seen+1 WHERE session_id='same' AND task=''")
    target._conn.execute("INSERT INTO messages(session_id,role,content,timestamp) VALUES ('same','assistant','new',2)")
    target._conn.commit()
    expected_usage = [tuple(r) for r in target._conn.execute("SELECT * FROM session_model_usage ORDER BY task")]
    result = target.adopt_session_lineage_from(donor, "same")
    assert result["donor_retired"], result
    assert target.get_session("same")["input_tokens"] == 10
    assert [tuple(r) for r in target._conn.execute("SELECT * FROM session_model_usage ORDER BY task")] == expected_usage
    assert [m["content"] for m in target.get_messages("same")] == ["kept", "new"]


def test_fresh_copy_keeps_bare_auxiliary_usage(stores):
    donor, target = stores
    donor.create_session("same", "cli")
    donor.append_message("same", "user", "kept")
    donor.record_auxiliary_usage("same", "vision", model="vision", billing_provider="p", input_tokens=3)
    before = [tuple(row) for row in donor._conn.execute("SELECT * FROM session_model_usage")]
    assert donor.get_session("same")["input_tokens"] == 0
    result = target.adopt_session_lineage_from(donor, "same")
    assert result["donor_retired"], result
    assert [tuple(row) for row in target._conn.execute("SELECT * FROM session_model_usage")] == before

def test_missing_attribution_origin_does_not_prove_a_growing_copy(stores):
    donor, target = stores
    _seed_portable_accounting(donor)
    _copy_portable_existing(donor, target)
    donor._conn.execute("UPDATE session_model_usage SET first_seen=NULL WHERE session_id='same' AND task=''")
    target._conn.execute("UPDATE session_model_usage SET first_seen=NULL, input_tokens=input_tokens+3 "
                         "WHERE session_id='same' AND task=''")
    target._conn.execute("UPDATE sessions SET input_tokens=input_tokens+3 WHERE id='same'")
    donor._conn.commit()
    target._conn.commit()
    before = snapshot(donor), snapshot(target)
    result = target.adopt_session_lineage_from(donor, "same")
    assert not result.get("donor_retired"), result
    assert (snapshot(donor), snapshot(target)) == before


def test_accounting_change_after_copy_before_journal_rolls_back(stores, monkeypatch):
    import hermes_state_portability as portability
    donor, target = stores
    _seed_portable_accounting(donor)
    target_before = snapshot(target)
    original = portability._copy_adoption_portable_state
    def changes_after_copy(db, donor_conn, target_conn, sid):
        original(db, donor_conn, target_conn, sid)
        donor._conn.execute("UPDATE sessions SET input_tokens=input_tokens+2 WHERE id=?", (sid,))
        donor._conn.execute("UPDATE session_model_usage SET input_tokens=input_tokens+2 "
                            "WHERE session_id=? AND task=''", (sid,))
        donor._conn.commit()
    monkeypatch.setattr(portability, "_copy_adoption_portable_state", changes_after_copy)
    result = target.adopt_session_lineage_from(donor, "same")
    assert not result.get("adopted") and not result.get("donor_retired"), result
    assert donor.get_session("same")["ended_at"] is None
    assert donor.get_session("same")["input_tokens"] == 9
    assert snapshot(target) == target_before

def test_accounting_change_before_retirement_leaves_donor_unretired(stores, monkeypatch):
    donor, target = stores
    _seed_portable_accounting(donor)
    original = SessionDB._retire_donor_segment
    def changes_before_retirement(self, source, sid):
        source._conn.execute("UPDATE sessions SET input_tokens=input_tokens+2 WHERE id=?", (sid,))
        source._conn.execute("UPDATE session_model_usage SET input_tokens=input_tokens+2 "
                             "WHERE session_id=? AND task=''", (sid,))
        source._conn.commit()
        return original(self, source, sid)
    monkeypatch.setattr(SessionDB, "_retire_donor_segment", changes_before_retirement)
    result = target.adopt_session_lineage_from(donor, "same")
    assert result["adopted"] and not result["donor_retired"], result
    assert donor.get_session("same")["ended_at"] is None
    assert donor.get_session("same")["input_tokens"] == 9
    assert target.get_session("same")["input_tokens"] == 7

def test_fresh_adoption_does_not_copy_runtime_authority_keys(stores):
    donor, target = stores
    _seed_portable_accounting(donor)
    donor._conn.execute("INSERT INTO state_meta(key,value) VALUES ('fixture-secret-authority','secret-value')")
    donor._conn.commit()
    result = target.adopt_session_lineage_from(donor, "same")
    assert result["donor_retired"], result
    assert target._conn.execute("SELECT value FROM state_meta WHERE key='fixture-secret-authority'").fetchone() is None


def _seed_local_cli_provenance(donor, title_source):
    donor.create_session("same", "cli")
    donor.append_message("same", "user", "kept")
    donor.record_gateway_session_peer("same", source="local", session_key="local:same",
                                     chat_id="same", chat_type="dm")
    if title_source == "user":
        assert donor.set_session_title("same", "User title")
    else:
        assert donor.set_auto_title("same", "Derived title", source="derived")
    assert donor.get_session("same")["created_source"] == "cli"

def test_fresh_adoption_preserves_user_title_and_cli_picker(stores):
    from hermes_cli.gateway_session_store import SurfaceLabelReader
    donor, target = stores
    _seed_local_cli_provenance(donor, "user")
    result = target.adopt_session_lineage_from(donor, "same")
    assert result["donor_retired"], result
    assert not target.set_auto_title("same", "Generated", source="llm")
    assert target.get_session("same")["title"] == "User title"
    assert target.get_session_title_source("same") == "user"
    assert "same" in [row["id"] for row in SurfaceLabelReader(target).search_sessions(source="cli", limit=10)]

def test_fresh_adoption_preserves_auto_title_upgrade_precedence(stores):
    donor, target = stores
    _seed_local_cli_provenance(donor, "derived")
    result = target.adopt_session_lineage_from(donor, "same")
    assert result["donor_retired"], result
    assert target.set_auto_title("same", "Improved", source="llm")
    assert target.get_session_title_source("same") == "llm"

@pytest.mark.parametrize("field,value", [("created_source", "tui"), ("title_source", "derived"),
                                         ("title", "Chosen target title")])
def test_existing_target_provenance_conflict_refuses_without_overwrite(stores, field, value):
    donor, target = stores
    _seed_local_cli_provenance(donor, "user")
    payload = donor.export_session("same", include_inactive=True)
    assert target.import_sessions([payload], transfer_from=donor.db_path)["imported"] == 1
    target._conn.execute("UPDATE sessions SET created_source='cli',title_source='user' WHERE id='same'")
    target._conn.execute(f"UPDATE sessions SET {field}=? WHERE id='same'", (value,))
    target._conn.commit()
    before = snapshot(donor), snapshot(target)
    result = target.adopt_session_lineage_from(donor, "same")
    assert not result.get("donor_retired"), result
    assert (snapshot(donor), snapshot(target)) == before

def test_generic_import_does_not_trust_source_provenance(stores):
    _donor, target = stores
    result = target.import_sessions([{"id": "untrusted", "source": "local", "created_source": "cli",
        "title": "Externally supplied", "title_source": "derived", "messages": []}])
    assert result["imported"] == 1
    row = target.get_session("untrusted")
    assert row["created_source"] is None and row["title_source"] is None


@pytest.mark.parametrize("target_title_source", ["derived", ""])
def test_legacy_unknown_title_provenance_cannot_downgrade_user_protection(stores, target_title_source):
    donor, target = stores
    _seed_local_cli_provenance(donor, "user")
    donor._conn.execute("UPDATE sessions SET title_source=NULL WHERE id='same'")
    donor._conn.commit()
    assert target.import_sessions([donor.export_session("same", include_inactive=True)],
                                  transfer_from=donor.db_path)["imported"] == 1
    target._conn.execute("UPDATE sessions SET created_source='cli',title_source=? WHERE id='same'",
                         (target_title_source,))
    target._conn.commit()
    before = snapshot(donor), snapshot(target)
    result = target.adopt_session_lineage_from(donor, "same")
    assert not result.get("donor_retired"), result
    assert (snapshot(donor), snapshot(target)) == before

def test_fresh_provenance_survives_pending_retirement_retry(stores, monkeypatch):
    from hermes_cli.gateway_session_store import SurfaceLabelReader
    donor, target = stores
    _seed_local_cli_provenance(donor, "derived")
    with monkeypatch.context() as patch:
        patch.setattr(SessionDB, "_retire_donor_segment", lambda *_: False)
        first = target.adopt_session_lineage_from(donor, "same")
    assert first["adopted"] and not first["donor_retired"], first
    assert target.get_session("same")["created_source"] == "cli"
    assert target.get_session_title_source("same") == "derived"
    second = target.adopt_session_lineage_from(donor, "same")
    assert second["donor_retired"], second
    assert target.get_session("same")["created_source"] == "cli"
    assert target.get_session_title_source("same") == "derived"
    assert "same" in [row["id"] for row in SurfaceLabelReader(target).search_sessions(source="cli", limit=10)]
