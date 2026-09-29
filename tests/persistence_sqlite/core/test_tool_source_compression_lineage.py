"""A compressed ``--source tool`` conversation stays one conversation (#112550 / #113826).

The parent-bound marker distinguishes published continuations from integration runs and old unmarked
children. Exercise the native interface as well as the backend predicates shared by its sibling paths.
"""

import json
import os
import time

import pytest

from hermes_cli.web_server_sessions import _session_latest_descendant
from persistence import CompressionBusy, PermanentStoreError
from persistence.authority import runtime as rt
from persistence.authority.bindings import local_session_id
from persistence.conversation.lineage import CompressionChild
from persistence.conversation.messages import NewMessage
from persistence.sqlite.conversation._store.compression import find_live_compression_child, _session_turn_lease_key_on_conn
from persistence.sqlite.conversation._store.messages import _is_explicit_fork_child_row
from persistence.sqlite.conversation._store.local import POLICY_PREFIX
from persistence.sqlite.conversation._store.sql_common import _non_continuation_child_sql
from tests.fixtures.storage_seed import seed_messages, seed_session
from tests.persistence_sqlite.workers._runtime_worker import worker as worker  # noqa: PLC0414 -- pytest fixture


def _rotate(db, parent, child, source="tool", config=None):
    return db.lineage.publish_compression_child(
        parent, CompressionChild(child, source, model_config=config),
        [NewMessage(role="user", content="[summary]")], require_compression_lease=False)


def _session_turn_lease_key(holder, session_id):
    with holder._read_ctx() as conn:
        return _session_turn_lease_key_on_conn(holder, conn, session_id)


@pytest.fixture
def db(backend, scope_a):
    database = backend.open(scope_a)
    seed_session(database, "root", "tool")
    seed_messages(database, "root", {"role": "user", "content": "automated repair prompt"})
    return database


def test_turn_lease_key_survives_rotation_of_a_tool_conversation(db):
    holder = f"pid={os.getpid()}:turn=live"
    assert db.coordination.try_acquire_turn_lease(session_id="root", holder=holder)
    _rotate(db, "root", "tip")
    assert _session_turn_lease_key(db.holder(), "tip") == "root"
    assert db.coordination.refresh_turn_lease(session_id="tip", holder=holder)
    assert not db.coordination.try_acquire_turn_lease(session_id="tip", holder="other")
    db.coordination.release_turn_lease(session_id="tip", holder=holder)


def test_tool_continuation_is_the_lineage_tip_and_resume_target(db):
    _rotate(db, "root", "mid")
    _rotate(db, "mid", "tip")
    assert db.lineage.tip("root") == "tip"
    assert db.lineage.lineage("mid") == ("root", "mid", "tip")
    assert db.lineage.resolve_resume_target("root") == "tip"
    assert db.lineage.resume_lineage_ids("tip") == ("root", "mid", "tip")
    assert not db.lineage.is_fork_child("tip")
    assert db.lineage.is_compression_child(db.sessions.get("tip"))


def test_compressed_tool_parent_is_not_reopened_as_an_orphan(db):
    _rotate(db, "root", "tip")
    assert find_live_compression_child(db.holder(), "root")["id"] == "tip"
    assert not db.lineage.reopen_orphaned_compression("root")
    assert db.sessions.get("root").end_reason == "compression"


def test_prune_keeps_the_compressed_start_of_a_live_tool_conversation(db):
    old = time.time() - 120 * 86400
    h = db.holder()
    h._write_sql("UPDATE sessions SET started_at = ?, last_activity_at = ? WHERE id = 'root'", (old, old))
    h._write_sql("UPDATE messages SET timestamp = ? WHERE session_id = 'root'", (old,))
    _rotate(db, "root", "tip")
    h._write_sql("UPDATE sessions SET ended_at = ?, last_activity_at = ? WHERE id = 'root'", (old + 60, old + 60))
    seed_messages(db, "tip", {"role": "user", "content": "still running today"})
    assert "root" not in {c.session_id for c in db.maintenance.list_prune_candidates(older_than_days=90, whole_lineages=True)}
    db.maintenance.prune_sessions(older_than_days=90, media_root="")
    assert db.sessions.get("root") is not None


def test_unmarked_tool_children_stay_separate_conversations(db):
    seed_session(db, "side-run", "tool", parent_session_id="root")
    _rotate(db, "root", "tip")
    seed_session(db, "inherited", "tool", parent_session_id="root", model_config={"_compressed_from": "elsewhere"})
    assert db.lineage.lineage("side-run") == ("side-run",)
    assert db.lineage.lineage("inherited") == ("inherited",)
    assert db.lineage.is_fork_child("side-run")
    assert db.lineage.tip("root") == "tip"
    assert find_live_compression_child(db.holder(), "root")["id"] == "tip"


def test_tool_child_of_a_non_tool_parent_is_unchanged(db):
    seed_session(db, "chat", "cli")
    seed_session(db, "integration", "tool", parent_session_id="chat")
    _rotate(db, "chat", "chat-2", source="cli")
    assert db.lineage.tip("chat") == "chat-2"
    assert db.lineage.lineage("integration") == ("integration",)
    assert db.sessions.get("chat-2").model_config_json is None


def test_inherited_marker_is_restamped_with_the_new_parent(db):
    config = {"_compressed_from": "grandparent", "temperature": 0.2}
    for parent, child in [("root", "mid"), ("mid", "tip")]:
        stamps = _rotate(db, parent, child, config=config)
        assert len(stamps) == 1 and stamps[0].id
        assert db.sessions.get(child).model_config == {"_compressed_from": parent, "temperature": 0.2}
    assert config == {"_compressed_from": "grandparent", "temperature": 0.2}
    assert db.lineage.lineage("tip") == ("root", "mid", "tip")
    assert _session_turn_lease_key(db.holder(), "tip") == "root"


@pytest.mark.parametrize("marker", ["_branched_from", "_delegate_from", "_reset_from"])
def test_bound_fork_marker_still_fences_a_marked_tool_child(db, marker):
    _rotate(db, "root", "tip")
    seed_session(db, "fork", "tool", parent_session_id="root", model_config={"_compressed_from": "root", marker: "root"})
    seed_messages(db, "fork", {"role": "user", "content": "separate conversation"})
    assert db.lineage.lineage("fork") == ("fork",)
    assert db.lineage.tip("root") == "tip"
    assert db.lineage.resolve_resume_target("root") == "tip"
    assert find_live_compression_child(db.holder(), "root")["id"] == "tip"
    assert _session_turn_lease_key(db.holder(), "fork") == "fork"
    assert not db.lineage.is_compression_child(db.sessions.get("fork"))


@pytest.mark.parametrize("source", ["tool", "cli"])
@pytest.mark.parametrize("config,continues", [
    (None, False), ({}, False), ({"_compressed_from": "1"}, True),
    ({"_compressed_from": "other"}, False), ({"_compressed_from": 1}, False),
    ({"_compressed_from": True}, False), ({"_compressed_from": ["1"]}, False),
    ({"_compressed_from": {"id": "1"}}, False),
    ({"_compressed_from": "1", "_branched_from": "1"}, False),
    ({"_compressed_from": "1", "_delegate_from": "1"}, False),
    ({"_compressed_from": "1", "_reset_from": "1"}, False),
    ({"_branched_from": "other"}, False), ([], False), ("not JSON", False),
])
def test_python_fork_predicate_matches_the_sql_continuation_filter(db, source, config, continues):
    seed_session(db, "1", "cli", end_reason="compression")
    seed_session(db, "C", source, parent_session_id="1")
    seed_session(db, "orphan", source)
    raw = config if isinstance(config, str) else json.dumps(config)
    h = db.holder()
    h._write_sql("UPDATE sessions SET model_config = ? WHERE id IN ('C', 'orphan')", (raw,))
    fork = _is_explicit_fork_child_row(h, dict(h._read_one("SELECT * FROM sessions WHERE id='C'")), include_reset=True)
    sql_continues = h._read_one("SELECT 1 FROM sessions WHERE id='C'" + _non_continuation_child_sql(parent="'1'")) is not None
    expected = continues if source == "tool" else not (isinstance(config, dict) and any(
        config.get(marker) == "1" for marker in ("_branched_from", "_delegate_from", "_reset_from")))
    assert sql_continues is expected
    assert fork is not expected
    assert db.lineage.is_compression_child(db.sessions.get("C")) is expected
    assert db.lineage.is_fork_child("orphan") is (source == "tool" or bool(
        isinstance(config, dict) and any(k in config for k in ("_branched_from", "_delegate_from"))))


def test_failed_publication_does_not_stamp_a_child_or_close_the_parent(db):
    config = {"_compressed_from": "old-parent", "temperature": 0.2}
    assert db.coordination.try_acquire_compression_lock(session_id="root", holder="owner")
    with pytest.raises(CompressionBusy):
        db.lineage.publish_compression_child("root", CompressionChild("blocked", "tool", model_config=config),
            [NewMessage(role="user", content="[summary]")], compression_lock_holder="intruder")
    assert config == {"_compressed_from": "old-parent", "temperature": 0.2}
    assert db.sessions.get("blocked") is None
    assert db.sessions.get("root").ended_at is None
    assert db.coordination.get_compression_lock_holder(session_id="root") == "owner"


def test_publication_rolls_back_after_the_marker_was_inserted(db):
    h = db.holder()
    h._write_sql("CREATE TRIGGER reject_handoff BEFORE INSERT ON messages WHEN NEW.session_id='blocked' "
                 "BEGIN SELECT RAISE(ABORT, 'reject handoff'); END")
    config = {"_compressed_from": "older"}
    with pytest.raises(PermanentStoreError, match="reject handoff"):
        _rotate(db, "root", "blocked", config=config)
    assert db.sessions.get("blocked") is None
    assert db.sessions.get("root").ended_at is None
    assert config == {"_compressed_from": "older"}


def test_worker_publication_stamps_both_rotations_in_the_shared_transaction(worker):
    db, store = worker
    db.holder()._write_sql("UPDATE sessions SET source='tool' WHERE id='owned'")
    config = {"_compressed_from": "grandparent", "temperature": 0.2}
    for parent, child in [("owned", "mid"), ("mid", "tip")]:
        assert store.try_acquire_compression_lock(parent, "holder")
        store.publish_compression_child(parent_session_id=parent, child_session_id=child, source="tool", model_config=config,
            messages=[{"role": "user", "content": "[summary]"}], compression_lock_holder="holder")
        assert store.scope["session_id"] == child
        assert db.sessions.get(child).model_config == {"_compressed_from": parent, "temperature": 0.2}
    assert config == {"_compressed_from": "grandparent", "temperature": 0.2}
    assert db.lineage.lineage("tip") == ("owned", "mid", "tip")
    assert _session_turn_lease_key(db.holder(), "tip") == "owned"


def test_gateway_peer_stamp_and_dashboard_descendants_follow_tool_continuations(db):
    _rotate(db, "root", "mid")
    _rotate(db, "mid", "tip")
    seed_session(db, "side", "tool", parent_session_id="mid")
    assert {e.session_id for e in db.sessions.descendants("root")} == {"root", "mid", "tip"}
    assert _session_latest_descendant("root", db) == ("tip", ["root", "mid", "tip"])
    db.routing.record_session_peer("tip", session_key="agent:cli:dm:1", source="tool", chat_id="1",
                                   include_compression_ancestors=True)
    assert all(db.sessions.get(s).session_key == "agent:cli:dm:1" for s in ("root", "mid", "tip"))
    assert db.sessions.get("side").session_key is None


def test_lineage_flags_expand_both_directions_without_touching_tool_forks(db):
    _rotate(db, "root", "mid")
    _rotate(db, "mid", "tip")
    seed_session(db, "side", "tool", parent_session_id="mid")
    db.sessions.set_pinned("mid", True)
    assert all(db.sessions.get(s).pinned for s in ("root", "mid", "tip"))
    assert not db.sessions.get("side").pinned


def test_canonical_compress_mutation_keeps_tool_lineage_across_two_rotations(db, scope_a):
    epoch = db.runtime.begin_epoch(instance_id="owner")
    sid = str(local_session_id(scope_a.name, "p", "create-1"))
    route = "agent:main:local:" + sid
    seed_session(db, sid, "tool", user_id="p", session_key=route, chat_id=sid)
    entry = {"session_key": route, "session_id": sid, "created_at": "2026-10-27T09:00:00", "updated_at": "2026-10-27T09:00:00"}
    db.routing.upsert_routing_entry(route, json.dumps(entry))
    receipt = {"session_id": sid, "profile_id": scope_a.name, "principal_id": "p",
        "request_id": "create-1", "route": route, "entry": entry, "lineage": [sid],
        "policy": {"source": "tool", "model": "m", "config_json": "{}", "cwd": None}}
    db.holder()._write_sql("INSERT INTO state_meta(key,value) VALUES (?,?)", (POLICY_PREFIX + sid, json.dumps(receipt)))
    lineage = [sid]
    for request in ("compress-1", "compress-2"):
        row = db.sessions.get(sid)
        fence = dict(principal_id="p", session_id=sid, request_id=request, expected_revision=row.runtime_revision,
                     expected_generation=row.runtime_generation, command=rt.Compress())
        snapshot = db.runtime.prepare_mutation(epoch, **fence)
        values = {"in_place": False, "messages": [{"role": "user", "content": "summary"}], "summary": "s"}
        mutation = dict(fence, prepared=rt.PreparedMutation(snapshot, values), owner=None)
        outcome = db.runtime.mutate(epoch, **mutation)
        child = outcome.result["target_session_id"]
        assert db.runtime.mutate(epoch, **mutation).result == outcome.result
        assert db.sessions.get(child).model_config["_compressed_from"] == lineage[-1]
        lineage.append(child)
        assert db.lineage.tip(sid) == child
        assert db.lineage.lineage(child) == tuple(lineage)
        assert _session_turn_lease_key(db.holder(), child) == sid
