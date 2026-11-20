"""Contract cases for value isolation: a nested list or mapping a verb returns, or one a caller passed in, is the caller's own copy, so changing it in place never changes what the store reads back.

Every case runs each registered backend through public operations only.
"""

from __future__ import annotations

import json

from persistence.conversation.messages import NewMessage
from persistence.conversation.sessions import NewSession, ReplaceSessionConfig, RuntimeLock
from persistence.core.identity import SessionId

S = SessionId("isolation-s1")
CALL = {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}


def _session(store, **fields):
    store.sessions.create(NewSession(session_id=S, source="cli", **fields))


def test_returned_tool_call_uids_are_the_callers_own(backend, scope_a):
    store = backend.open(scope_a)
    _session(store)
    batch = store.messages.append_batch(S, [NewMessage(role="assistant", content="", tool_calls=(CALL,), tool_call_uids={"c1": ["u1", "u2"]})])
    (outcome,) = batch.outcomes
    outcome.tool_call_uids["c1"].append("changed")
    (row,) = store.messages.read_transcript(S)
    assert json.loads(row.tool_call_uids) == {"c1": ["u1", "u2"]}


def test_a_repair_with_its_snapshot_copies_values_both_ways(backend, scope_a):
    store = backend.open(scope_a)
    _session(store)
    (first,) = store.messages.append_batch(S, [NewMessage(role="assistant", content="", tool_calls=(CALL,), display_metadata={"tags": ["a"]})]).outcomes
    uids = {"c1": ["r1"]}
    (repaired,) = store.messages.append_batch(S, [NewMessage(role="assistant", content="", tool_calls=(CALL,), tool_call_uids=uids,
                                                             repair_target=first.id, db_row_snapshot=first.db_row_snapshot)]).outcomes
    uids["c1"].append("changed")
    repaired.canonical_row["display_metadata"]["tags"].append("changed")
    (row,) = store.messages.read_transcript(S)
    assert (json.loads(row.tool_call_uids), row.display_metadata) == ({"c1": ["r1"]}, {"tags": ["a"]})


def test_a_repair_without_a_snapshot_copies_content_both_ways(backend, scope_a):
    store = backend.open(scope_a)
    _session(store)
    blank, held = store.messages.append_batch(S, [NewMessage(role="assistant", content=""),
                                                  NewMessage(role="assistant", content=[{"type": "text", "text": "kept"}])]).outcomes
    filled = [{"type": "text", "text": "filled"}]
    (_, adopted) = store.messages.append_batch(S, [NewMessage(role="assistant", content=filled, repair_target=blank.id),
                                                   NewMessage(role="assistant", content="late", repair_target=held.id)]).outcomes
    filled[0]["text"] = "changed"
    adopted.canonical_row["content"][0]["text"] = "changed"
    assert [row.content for row in store.messages.read_transcript(S)] == [[{"type": "text", "text": "filled"}], [{"type": "text", "text": "kept"}]]


def test_returned_model_config_and_origin_are_the_callers_own(backend, scope_a):
    store = backend.open(scope_a)
    _session(store, model_config={"options": {"depth": 1}, "tags": ["a"]}, origin={"platform": "cli", "chat": {"id": "c"}})
    record = store.sessions.get(S)
    record.model_config["options"]["depth"] = 9
    record.model_config["tags"].append("b")
    record.origin["chat"]["id"] = "changed"
    store.sessions.get_model_config_value(S, "options")["depth"] = 9
    again = store.sessions.get(S)
    assert (again.model_config["options"], again.model_config["tags"], again.origin["chat"]) == ({"depth": 1}, ["a"], {"id": "c"})
    assert store.sessions.get_model_config_value(S, "options") == {"depth": 1}


def test_written_model_config_and_origin_are_copied_in(backend, scope_a):
    store = backend.open(scope_a)
    config, origin = {"options": {"depth": 1}}, {"platform": "cli", "chat": {"id": "c"}}
    _session(store, model_config=config, origin=origin)
    config["options"]["depth"] = 9
    origin["chat"]["id"] = "changed"
    record = store.sessions.get(S)
    assert (record.model_config["options"], record.origin["chat"]) == ({"depth": 1}, {"id": "c"})


def test_config_writes_copy_their_nested_values_in(backend, scope_a):
    store = backend.open(scope_a)
    _session(store)
    whole, patch, options = {"options": {"depth": 1}}, {"extra": ["a"]}, {"effort": ["high"]}
    store.sessions.replace_session_config(S, ReplaceSessionConfig(model_config=whole))
    store.sessions.patch_model_config(S, patch)
    store.sessions.set_runtime_lock(S, RuntimeLock(model="m", model_options=options))
    whole["options"]["depth"] = 9
    patch["extra"].append("b")
    options["effort"].append("low")
    assert store.sessions.get_model_config_value(S, "options") == {"depth": 1}
    assert store.sessions.get_model_config_value(S, "extra") == ["a"]
    assert store.sessions.get_model_config_value(S, "browser_model_lock")["model_options"] == {"effort": ["high"]}
