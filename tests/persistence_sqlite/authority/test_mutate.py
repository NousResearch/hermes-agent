"""``SessionRuntimeRepository.mutate`` and ``prepare_mutation`` on SQLite against main's ``mutate_runtime_session``.

Every command runs main's function and the verb on copies of one seed: the same result dict, the same stored receipt row
(key and value) and the same rows in every table the command writes; an exact retry returns the stored result with
``replayed`` set, and every refusal matches. ``gone`` is the event main hands ``emit_session_deleted`` after its commit.
The owner's checks arrive as data and fire where main calls its hooks: ``live_guard`` after the receipt replay,
existence, revision and generation checks and the delete target walk, ``history`` right after it, and an exact
replay bypasses both. Each call is one write transaction, and a write that fails right after its first mutating
statement leaves every table unchanged. The import command reaches the housekeeping family's import bodies and is
covered in ``test_mutate_import.py``.
"""

from __future__ import annotations

import itertools
import json
import uuid
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Callable

import pytest

from persistence.authority import runtime as rt
from persistence.authority.admissions import GateRefusal
from persistence.authority.bindings import local_session_id
from tests.persistence_sqlite.authority._admission_support import BackendSeed, admit, claim
from tests.persistence_sqlite.authority._support import AUTHORITY_TABLES, assert_same, counted_transactions, run_both
from tests.persistence_sqlite.authority.test_binding_local import make_binding, reserved
from tests.persistence_sqlite.authority.test_binding_local import seed as seed_store
from tests.persistence_sqlite.authority.test_runtime_picks import assert_one_write

PRINCIPAL = "user-1"
LOCAL = str(local_session_id("default", PRINCIPAL, "req-1"))
TABLES = (*AUTHORITY_TABLES, "gateway_routing", "system_prompts")
SUBJECT = 'auth:v1:["github","","octo"]'


def legacy_mutate(db: Any, command: Any, *, session_id: str, request_id: str = "req-m", expected_revision: int = 0,
                  expected_generation: int | None = None, epoch: int = 1, **hooks: Any) -> dict:
    from hermes_state_runtime import mutate_runtime_session

    return mutate_runtime_session(db, epoch=epoch, principal_id=PRINCIPAL, session_id=session_id, request_id=request_id,
                                  expected_revision=expected_revision, operation=command.operation, payload=command.payload(),
                                  expected_generation=expected_generation, **hooks)


def mutate(store: Any, command: Any, *, session_id: str, request_id: str = "req-m", expected_revision: int = 0,
           expected_generation: int | None = None, epoch: int = 1, owner: Any = None, **kwargs: Any) -> rt.MutationOutcome:
    return store.runtime.mutate(epoch, principal_id=PRINCIPAL, session_id=session_id, request_id=request_id,
                                expected_revision=expected_revision, command=command, expected_generation=expected_generation,
                                owner=owner, **kwargs)


def compare(store: Any, frozen: Callable[[], None], emitted: list, legacy: Callable[[Any], Any], ours: Callable[[], Any],
            *, replayed: bool = False) -> tuple:
    """Run both on copies of the seed; the same outcome, rows and receipt, and ``gone`` is main's emitted event.
    Returns the verb's outcome."""
    def run_legacy(db: Any) -> Any:
        frozen()
        emitted.clear()
        return legacy(db)

    def run_ours() -> Any:
        frozen()
        return ours()

    old, new = run_both(store.store_path, run_legacy, run_ours, TABLES)
    assert_same(old, new, lambda outcome: outcome.result)
    if new[0] == "ok":
        assert new[1].replayed is replayed
        assert new[1].gone == (emitted[0] if emitted else None)
    return new


def expect(outcome: tuple, expected: str | None) -> None:
    """``expected`` is the refusal reason, or None for a committed outcome."""
    if expected is None:
        assert outcome[0] == "ok", outcome
    else:
        assert getattr(outcome[1], "reason", None) == expected, outcome


def commit_receipt(command: Any, *, session_id: str, request_id: str = "req-m", **kwargs: Any):
    """A seed step committing ``command`` through main's function."""
    return lambda db: legacy_mutate(db, command, session_id=session_id, request_id=request_id, **kwargs)


def raw(sql: str, *params: Any):
    """A seed step running one statement on the legacy writer."""
    return lambda db: db._execute_write(lambda conn: conn.execute(sql, params))


def plain(session_id: str = "s-plain", **kwargs: Any):
    return lambda db: db.create_session(session_id, source="cli", **kwargs)


def lineage(db: Any) -> None:
    db.create_session("s-root", source="cli")
    (db.store.sessions.end if isinstance(db, BackendSeed) else db.end_session)("s-root", "compression")
    db.create_session("s-tip", source="cli", parent_session_id="s-root")


def transcript(session_id: str = "s-plain"):
    def run(db: Any) -> None:
        db.create_session(session_id, source="cli")
        for role, content in (("user", "one"), ("assistant", "reply"), ("user", "two"), ("assistant", "reply two")):
            db.append_message(session_id, role=role, content=content)
    return run


def admitted(session_id: str):
    return lambda db: admit(db, session_id, "q")


def claimed(session_id: str):
    return lambda db: claim(db, session_id)


def deferred(value: dict):
    from persistence.sqlite.authority._runtime_picks import DEFERRED_PICK_PREFIX

    return lambda db: db.set_meta(DEFERRED_PICK_PREFIX + LOCAL, json.dumps(value))


def guard(reason: str = "runtime_coordination_required"):
    """Main's live guard callable and the owner's verdict the verb takes for it."""
    def legacy(targets: list) -> None:
        from persistence.core.errors import AuthorityRefusal

        raise AuthorityRefusal(reason, field="live")
    return legacy, GateRefusal(reason, field="live")


# ── metadata commands ─────────────────────────────────────────────────────────

METADATA = {
    "rename": ([plain()], rt.Rename("Planning"), "s-plain"),
    "rename_taken": ([plain(), plain("s-other"), raw("UPDATE sessions SET title='Planning' WHERE id='s-other'")], rt.Rename("Planning"), "s-plain"),
    "archive_first": ([plain()], rt.Archive(True), "s-plain"),
    "archive_again": ([plain(), raw("UPDATE sessions SET archived=1 WHERE id='s-plain'")], rt.Archive(True), "s-plain"),
    "unarchive": ([plain(), raw("UPDATE sessions SET archived=1 WHERE id='s-plain'")], rt.Archive(False), "s-plain"),
    "sidebar_lineage": ([lineage], rt.Sidebar(archived=True, pinned=True, hidden=True, unread=True), "s-tip"),
    "sidebar_unread": ([lineage], rt.Sidebar(unread=False), "s-tip"),
    "sidebar_title": ([plain()], rt.Sidebar(title="Named", pinned=False), "s-plain"),
}


@pytest.mark.parametrize("case", sorted(METADATA))
def test_metadata_commands_match_m1(store, frozen, emitted, case):
    steps, command, sid = METADATA[case]
    seed_store(store, *steps)
    new = compare(store, frozen, emitted, lambda db: legacy_mutate(db, command, session_id=sid),
                  lambda: mutate(store, command, session_id=sid))
    if case != "rename_taken":
        expect(new, None)


# ── the fences, in main's order ───────────────────────────────────────────────

FENCES = {
    "not_found": ([], {"session_id": "s-absent"}, "not_found"),
    "revision_conflict": ([plain()], {"expected_revision": 1}, "revision_conflict"),
    "stale_generation": ([plain()], {"expected_generation": 4}, "stale_generation"),
    "stale_epoch": ([plain()], {"epoch": 2}, "stale_epoch"),
    "admission_conflict": ([plain(), commit_receipt(rt.Sidebar(hidden=True), session_id="s-plain")], {}, "admission_conflict"),
    "exact_replay": ([plain(), commit_receipt(rt.Sidebar(pinned=True), session_id="s-plain")], {}, None),
    # The receipt answers before the revision check: the row's revision moved past the request's.
    "replay_past_revision": ([plain(), commit_receipt(rt.Sidebar(pinned=True), session_id="s-plain"),
                              commit_receipt(rt.Sidebar(hidden=True), session_id="s-plain", request_id="req-2", expected_revision=1)], {}, None),
    "bad_text": ([plain()], {"request_id": ""}, "invalid_params"),
    "bad_revision": ([plain()], {"expected_revision": -1}, "invalid_params"),
    "bad_generation": ([plain()], {"expected_generation": True}, "invalid_params"),
}


@pytest.mark.parametrize("case", sorted(FENCES))
def test_fences_match_m1(store, frozen, emitted, case):
    steps, overrides, expected = FENCES[case]
    seed_store(store, *steps)
    call = {"session_id": "s-plain", **overrides}
    new = compare(store, frozen, emitted, lambda db: legacy_mutate(db, rt.Sidebar(pinned=True), **call),
                  lambda: mutate(store, rt.Sidebar(pinned=True), **call), replayed=expected is None)
    expect(new, expected)


def test_generation_is_required_where_m1_requires_it(store, frozen, emitted):
    seed_store(store, plain())
    for command in (rt.Delete(), rt.Rewind(1), rt.Reset(), rt.Branch(), rt.ModelPick("m"), rt.Compress()):
        compare(store, frozen, emitted, lambda db: legacy_mutate(db, command, session_id="s-plain"),
                lambda: mutate(store, command, session_id="s-plain"))


def test_a_value_that_is_no_command_refuses_invalid_params(store):
    from persistence import AuthorityRefusal

    seed_store(store, plain(), legacy=False)
    for command in (SimpleNamespace(operation="sidebar", payload=lambda: {"color": "red"}), "sidebar", None):
        with pytest.raises(AuthorityRefusal, match="^invalid_params$"):
            mutate(store, command, session_id="s-plain")


# ── delete ────────────────────────────────────────────────────────────────────

DELETES = {
    "plain": [transcript()],
    "busy": [transcript(), admitted("s-plain")],
    "lineage_tip": [lineage, raw("INSERT INTO messages(session_id, role, content, timestamp) VALUES('s-root','user','x',1.0)")],
    "local_receipt": [reserved(make_binding())],
}


@pytest.mark.parametrize("case", sorted(DELETES))
def test_delete_matches_m1_and_names_the_gone_ids(store, frozen, emitted, case):
    sid = {"lineage_tip": "s-tip", "local_receipt": LOCAL}.get(case, "s-plain")
    seed_store(store, *DELETES[case])
    new = compare(store, frozen, emitted, lambda db: legacy_mutate(db, rt.Delete(), session_id=sid, expected_generation=0),
                  lambda: mutate(store, rt.Delete(), session_id=sid, expected_generation=0))
    expect(new, "session_busy" if case == "busy" else None)
    if case == "lineage_tip":
        assert new[1].gone == rt.SessionGone(("s-tip", "s-root"), "delete")


HOSTED_ADMISSION = ("INSERT INTO session_admissions (admission_id, request_id, principal_id, target_session_id,"
                    " lineage_json, payload_json, payload_digest, intent, status, outcome)"
                    " VALUES ('adm-h', 'hosted:room:1', 'p', 's-plain', '[]', ?, 'digest', 'queue', 'terminal', 'completed')")


@pytest.mark.parametrize("root", ["a", "b"])
def test_delete_retires_the_hosted_documents_under_its_media_root(store, frozen, emitted, monkeypatch, tmp_path, root):
    import hashlib

    import gateway.session_ingress_media
    from session_rules.media import HOSTED_DOCUMENT_LINE

    roots = {name: (tmp_path / name / "native-inputs").resolve() for name in ("a", "b")}
    documents = [str(r / hashlib.sha256(name.encode()).hexdigest() / f"{name}.txt") for name, r in roots.items()]
    text = "shared" + "".join(HOSTED_DOCUMENT_LINE + path + "\n" for path in documents)
    seed_store(store, transcript(), raw(HOSTED_ADMISSION, json.dumps({"text": text})))
    monkeypatch.setattr(gateway.session_ingress_media, "_media_root", lambda: roots[root])
    new = compare(store, frozen, emitted, lambda db: legacy_mutate(db, rt.Delete(), session_id="s-plain", expected_generation=0),
                  lambda: mutate(store, rt.Delete(), session_id="s-plain", expected_generation=0, media_root=str(roots[root])))
    expect(new, None)
    with store.holder()._read_ctx() as conn:
        retired = [json.loads(value)["path"] for (value,) in conn.execute(
            "SELECT value FROM state_meta WHERE key LIKE 'gateway.media.retired.v1.%'")]
    assert retired == [documents[0 if root == "a" else 1]]


def test_delete_without_a_media_root_retires_no_hosted_document(store, frozen, emitted, monkeypatch, tmp_path):
    """Without a media root a Delete names no hosted document, as main does when its root holds none of them, and
    still retires the admission's native and API media."""
    import hashlib

    import gateway.session_ingress_media
    from session_rules.media import HOSTED_DOCUMENT_LINE

    document = str((tmp_path / "native-inputs").resolve() / hashlib.sha256(b"doc").hexdigest() / "doc.txt")
    native, api = str(tmp_path / "native.png"), str(tmp_path / "api.png")
    payload = {"text": "shared" + HOSTED_DOCUMENT_LINE + document + "\n",
               "attachments_v1": {"media": [{"path": native}]}, "api_turn_v1": {"media": [{"path": api}]}}
    seed_store(store, transcript(), raw(HOSTED_ADMISSION, json.dumps(payload)))
    monkeypatch.setattr(gateway.session_ingress_media, "_media_root", lambda: (tmp_path / "elsewhere").resolve())
    new = compare(store, frozen, emitted, lambda db: legacy_mutate(db, rt.Delete(), session_id="s-plain", expected_generation=0),
                  lambda: mutate(store, rt.Delete(), session_id="s-plain", expected_generation=0, media_root=None))
    expect(new, None)
    with store.holder()._read_ctx() as conn:
        retired = sorted(json.loads(value)["path"] for (value,) in conn.execute(
            "SELECT value FROM state_meta WHERE key LIKE 'gateway.media.retired.v1.%'"))
    assert retired == sorted([native, api])


@pytest.mark.parametrize("key", [None, "", "agent", "agent:main:x", "agent::x", "agent:main~:y", "agent:work:z",
                                 "foo:bar", 12, "agent:default:q"])
def test_the_branch_route_profile_matches_m1(key):
    from hermes_state_keys import profile_from_session_key
    from persistence.sqlite.authority._mutation_publish import _profile_from_session_key

    assert _profile_from_session_key(key) == profile_from_session_key(key)


def test_delete_receipt_replays_after_the_row_is_gone(store, frozen, emitted):
    seed_store(store, transcript(), commit_receipt(rt.Delete(), session_id="s-plain", expected_generation=0))
    compare(store, frozen, emitted, lambda db: legacy_mutate(db, rt.Delete(), session_id="s-plain", expected_generation=0),
            lambda: mutate(store, rt.Delete(), session_id="s-plain", expected_generation=0), replayed=True)


# ── rewind ────────────────────────────────────────────────────────────────────


def _user_ids(store: Any) -> list[int]:
    import sqlite3

    with sqlite3.connect(store.store_path) as conn:
        return [row[0] for row in conn.execute("SELECT id FROM messages WHERE role='user' ORDER BY id")]


@pytest.mark.parametrize("case", ["second_turn", "first_turn", "assistant_row", "compacted_row", "unknown_row", "preserve_without_carrier"])
def test_rewind_matches_m1(store, frozen, emitted, case):
    steps = [transcript()]
    if case == "compacted_row":
        steps.append(raw("UPDATE messages SET active=0, compacted=1 WHERE role='user' AND content='two'"))
    seed_store(store, *steps)
    first, second = _user_ids(store)
    target = {"first_turn": first, "assistant_row": second + 1, "unknown_row": 999}.get(case, second)
    command = rt.Rewind(target, True if case == "preserve_without_carrier" else None)
    new = compare(store, frozen, emitted, lambda db: legacy_mutate(db, command, session_id="s-plain", expected_generation=0),
                  lambda: mutate(store, command, session_id="s-plain", expected_generation=0))
    if case in ("second_turn", "first_turn"):
        expect(new, None)
    elif case != "preserve_without_carrier":
        expect(new, "invalid_params")
        detail = rt.COMPACTED_TARGET_DETAIL if case == "compacted_row" else "target user message is no longer in session history"
        assert (new[1].field, new[1].detail) == ("target_message_id", detail)


# ── prepared policy commits ───────────────────────────────────────────────────


def prepare(store: Any, command: Any, **kwargs: Any) -> Any:
    call = {"session_id": LOCAL, "request_id": "req-m", "expected_revision": 0, **kwargs}
    return store.runtime.prepare_mutation(1, principal_id=PRINCIPAL, command=command, **call)


def _snapshot(snapshot: rt.LocalSnapshot) -> dict:
    return {"snapshot": {"receipt": snapshot.receipt, "target": snapshot.target, "target_revision": snapshot.target_revision}}


@pytest.mark.parametrize("case", ["model", "branch", "model_queued", "branch_queued", "no_receipt", "guarded"])
def test_prepare_mutation_is_m1s_prepare_only(store, legacy_store, frozen, case):
    command = rt.Branch() if case.startswith("branch") else rt.ModelPick("model-b")
    steps = [plain(LOCAL)] if case == "no_receipt" else [reserved(make_binding())]
    if case.endswith("queued"):
        steps.append(admitted(LOCAL))
    seed_store(store, *steps)
    legacy_guard, verdict = guard()
    guarded = case == "guarded"
    old, new = run_both(
        store.store_path,
        lambda db: legacy_mutate(db, command, session_id=LOCAL, expected_generation=0, _prepare_only=True,
                                 _live_guard=legacy_guard if guarded else None),
        lambda: prepare(store, command, expected_generation=0, live_guard=verdict if guarded else None), TABLES)
    assert_same(old, new, _snapshot)
    # CONTRACT: the branch prepare refuses queued input too (require_idle); the others only executing work.
    if case == "branch_queued":
        assert new[1].reason == "session_busy"
    if case == "model_queued":
        assert isinstance(new[1], rt.LocalSnapshot)


def test_prepare_mutation_replays_a_committed_receipt(store, frozen, emitted):
    policy = {"source": "cli", "model": "model-b", "cwd": "/workspace", "request_json": '{"kind":"launch","n":1}',
              "config_json": json.dumps({"model": {"provider": "p2"}})}
    seed_store(store, reserved(make_binding()))

    def legacy(db: Any) -> dict:
        prepared = legacy_mutate(db, rt.EditPolicy("ask"), session_id=LOCAL, _prepare_only=True)
        legacy_mutate(db, rt.EditPolicy("ask"), session_id=LOCAL, _prepared={**prepared, "policy": policy})
        return legacy_mutate(db, rt.EditPolicy("ask"), session_id=LOCAL, _prepare_only=True)

    def ours() -> Any:
        snapshot = store.runtime.prepare_mutation(1, principal_id=PRINCIPAL, session_id=LOCAL, request_id="req-m", expected_revision=0, command=rt.EditPolicy("ask"))
        mutate(store, rt.EditPolicy("ask"), session_id=LOCAL, prepared=rt.PreparedMutation(snapshot, {"policy": policy}))
        return store.runtime.prepare_mutation(1, principal_id=PRINCIPAL, session_id=LOCAL, request_id="req-m", expected_revision=0, command=rt.EditPolicy("ask"))

    old, new = run_both(store.store_path, lambda db: (frozen(), legacy(db))[1], lambda: (frozen(), ours())[1], TABLES)
    assert_same(old, new, lambda outcome: outcome.result)
    assert new[1].replayed and new[1].gone is None


def _policy(**changes: Any) -> dict:
    base = {"source": "cli", "model": "model-a", "cwd": "/workspace", "request_json": '{"kind":"launch","n":1}'}
    return {**base, **changes}


PREPARED = {
    "model": (rt.ModelPick("model-b", provider="p2"), {"policy": _policy(model="model-b", config_json=json.dumps({"model": {"provider": "p2"}}))}, []),
    "model_keeps_deferred_level": (rt.ModelPick("model-b"), {"policy": _policy(model="model-b", config_json=json.dumps({"model": {"provider": "p2"}}))},
                                   [deferred({"principal_id": "p", "payload": {"model": "old"}, "reasoning": {"payload": {"effort": "low"}}})]),
    "model_with_level": (rt.ModelPick("model-b", reasoning="high"), {"policy": _policy(model="model-b", config_json=json.dumps({"model": {"provider": "p2"}}))},
                         [deferred({"principal_id": "p", "payload": {"model": "old"}, "reasoning": {"payload": {"effort": "low"}}})]),
    "reasoning": (rt.ReasoningPick("high"), {"policy": _policy(reasoning_effort="high")},
                  [deferred({"principal_id": "p", "payload": {"model": "old", "reasoning": "low"}, "application_id": "deferred-pick:d"})]),
    "reasoning_reset": (rt.ReasoningPick(None), {"policy": _policy()}, [deferred({"reasoning": {"payload": {"effort": "low"}}})]),
    "workspace": (rt.Workspace("/w2"), {"policy": _policy(cwd="/w2"), "git": {"branch": "main", "root": ""}}, []),
    "editor_mcp": (rt.EditorMcp("f"), {"policy": _policy(editor_mcp_json='{"servers":1}'), "tools_changed": True, "mcp_servers": ["fs"]}, []),
    "edit_policy": (rt.EditPolicy("session"), {"policy": _policy(edit_approval_policy="session")}, []),
}


@pytest.mark.parametrize("case", sorted(PREPARED))
def test_prepared_policy_commits_match_m1(store, frozen, emitted, case):
    command, values, steps = PREPARED[case]
    seed_store(store, reserved(make_binding()), *steps)
    generation = 0 if command.operation == "model" else None

    def legacy(db: Any) -> dict:
        prepared = legacy_mutate(db, command, session_id=LOCAL, expected_generation=generation, _prepare_only=True)
        return legacy_mutate(db, command, session_id=LOCAL, expected_generation=generation, _prepared={**prepared, **values})

    def ours() -> Any:
        snapshot = store.runtime.prepare_mutation(1, principal_id=PRINCIPAL, session_id=LOCAL, request_id="req-m", expected_revision=0,
                                                  command=command, expected_generation=generation)
        return mutate(store, command, session_id=LOCAL, expected_generation=generation, prepared=rt.PreparedMutation(snapshot, values))

    new = compare(store, frozen, emitted, legacy, ours)
    expect(new, None)


@pytest.mark.parametrize("case", ["no_prepared", "moved_snapshot"])
def test_prepared_commit_refusals_match_m1(store, frozen, emitted, case):
    seed_store(store, reserved(make_binding()))
    values = {"policy": _policy(edit_approval_policy="ask")}

    def legacy(db: Any) -> dict:
        if case == "no_prepared":
            return legacy_mutate(db, rt.EditPolicy("ask"), session_id=LOCAL)
        prepared = legacy_mutate(db, rt.EditPolicy("ask"), session_id=LOCAL, _prepare_only=True)
        prepared["snapshot"]["target_revision"] += 1
        return legacy_mutate(db, rt.EditPolicy("ask"), session_id=LOCAL, _prepared={**prepared, **values})

    def ours() -> Any:
        if case == "no_prepared":
            return mutate(store, rt.EditPolicy("ask"), session_id=LOCAL)
        snapshot = store.runtime.prepare_mutation(1, principal_id=PRINCIPAL, session_id=LOCAL, request_id="req-m", expected_revision=0, command=rt.EditPolicy("ask"))
        moved = rt.LocalSnapshot(snapshot.receipt, snapshot.target, snapshot.target_revision + 1)
        return mutate(store, rt.EditPolicy("ask"), session_id=LOCAL, prepared=rt.PreparedMutation(moved, values))

    new = compare(store, frozen, emitted, legacy, ours)
    assert new[1].reason == ("runtime_coordination_required" if case == "no_prepared" else "revision_conflict")


# ── compress, reset and branch publish ────────────────────────────────────────


@pytest.mark.parametrize("in_place", [True, False])
def test_compress_commits_the_prepared_summary_like_m1(store, frozen, emitted, in_place):
    seed_store(store, reserved(make_binding()), lambda db: [db.append_message(LOCAL, role, content) for role, content in
                                                            (("user", "one"), ("assistant", "reply"), ("user", "two"), ("assistant", "reply two"))])
    values = {"in_place": in_place, "summary": "s", "messages": [{"role": "user", "content": "[summary] earlier turns"}, {"role": "assistant", "content": "ok"}]}

    def legacy(db: Any) -> dict:
        prepared = legacy_mutate(db, rt.Compress(), session_id=LOCAL, expected_generation=0, _prepare_only=True)
        return legacy_mutate(db, rt.Compress(), session_id=LOCAL, expected_generation=0, _prepared={**prepared, **values})

    def ours() -> Any:
        snapshot = store.runtime.prepare_mutation(1, principal_id=PRINCIPAL, session_id=LOCAL, request_id="req-m", expected_revision=0,
                                                  command=rt.Compress(), expected_generation=0)
        return mutate(store, rt.Compress(), session_id=LOCAL, expected_generation=0, prepared=rt.PreparedMutation(snapshot, values))

    new = compare(store, frozen, emitted, legacy, ours)
    assert new[0] == "ok", new


@pytest.mark.parametrize("case", ["fresh", "busy_running", "queued_follower", "no_receipt"])
def test_reset_publishes_a_fresh_target_like_m1(store, frozen, emitted, case):
    steps = {"fresh": [reserved(make_binding())], "no_receipt": [plain(LOCAL)],
             "queued_follower": [reserved(make_binding()), admitted(LOCAL)],
             "busy_running": [reserved(make_binding()), admitted(LOCAL), claimed(LOCAL)]}[case]
    seed_store(store, *steps)
    new = compare(store, frozen, emitted, lambda db: legacy_mutate(db, rt.Reset(), session_id=LOCAL, expected_generation=0),
                  lambda: mutate(store, rt.Reset(), session_id=LOCAL, expected_generation=0))
    if case in ("fresh", "queued_follower"):
        assert new[0] == "ok", new


BRANCHES = {
    "untitled": (rt.Branch(), [], None),
    "titled_through": (rt.Branch(title="Fork", through_message_id=1), [], None),
    "editor_unprepared": (rt.Branch(), [], {"editor_mcp_json": '{"servers":1}'}),
    "queued": (rt.Branch(), [admitted(LOCAL)], None),
    "through_unknown": (rt.Branch(through_message_id=999), [], None),
}


@pytest.mark.parametrize("case", sorted(BRANCHES))
def test_branch_publishes_an_independent_route_like_m1(store, frozen, emitted, case):
    command, extra, policy = BRANCHES[case]
    seed_store(store, reserved(make_binding(policy=policy)),
               lambda db: [db.append_message(LOCAL, role, content) for role, content in (("user", "one"), ("assistant", "reply"), ("tool", "t"), ("user", "two"))],
               *extra)
    new = compare(store, frozen, emitted, lambda db: legacy_mutate(db, command, session_id=LOCAL, expected_generation=0),
                  lambda: mutate(store, command, session_id=LOCAL, expected_generation=0))
    if case in ("untitled", "titled_through"):
        assert new[0] == "ok", new


def test_branch_commits_the_prepared_editor_selection_like_m1(store, frozen, emitted):
    from hermes_state_keys import local_identity

    seed_store(store, reserved(make_binding(policy={"editor_mcp_json": '{"servers":1}'})))
    values = {"request_id": "branch:fixed", "child": local_identity("default", PRINCIPAL, "branch:fixed"),
              "parent_editor_mcp_json": '{"servers":1}', "editor_mcp_json": '{"servers":2}', "toolsets": ["fs"], "config_secret_ref": None}

    def legacy(db: Any) -> dict:
        prepared = legacy_mutate(db, rt.Branch(), session_id=LOCAL, expected_generation=0, _prepare_only=True)
        return legacy_mutate(db, rt.Branch(), session_id=LOCAL, expected_generation=0, _prepared={**prepared, **values})

    def ours() -> Any:
        snapshot = store.runtime.prepare_mutation(1, principal_id=PRINCIPAL, session_id=LOCAL, request_id="req-m", expected_revision=0,
                                                  command=rt.Branch(), expected_generation=0)
        return mutate(store, rt.Branch(), session_id=LOCAL, expected_generation=0, prepared=rt.PreparedMutation(snapshot, values))

    new = compare(store, frozen, emitted, legacy, ours)
    assert new[0] == "ok" and new[1].result["branched_session_id"] == values["child"]


# ── the owner's checks: live_guard and history ────────────────────────────────


def _actor(*, may_create: bool = True) -> tuple[Any, rt.HistoryActor]:
    """Main's actor (the gateway's) and the contract's ``HistoryActor`` for the same subject."""
    from hermes_state_mutation_binding import _login_label

    legacy = SimpleNamespace(profile_id="default", subject=SUBJECT, capabilities=("session:create",) if may_create else ())
    return legacy, rt.HistoryActor(profile_id="default", subject=SUBJECT, login_label=_login_label(SUBJECT), may_create=may_create)


def _legacy_claim(actor: Any) -> Callable[[Any], None]:
    """The gateway's ``authorize_write`` hook for a control edit of cold history (main's gateway/session_mutations.py)."""
    def authorize_write(conn: Any) -> None:
        from hermes_state_mutation_binding import authorize_history
        from persistence.core.errors import AuthorityRefusal

        if not authorize_history(conn, actor, "s-plain", claim=True):
            raise AuthorityRefusal("permission_denied")
    return authorize_write


HISTORY = {
    "claim_unbound": ([plain()], {}),
    "claim_own_login": ([plain(user_id="github:octo")], {}),
    "claim_foreign_user": ([plain(user_id="someone-else")], {}),
    "claim_without_create": ([plain()], {"may_create": False}),
    "routed_row": ([plain(session_key="agent:main:telegram:dm:1")], {}),
    "other_binding": ([plain(), lambda db: db.set_meta("gateway.history_control.v1.s-plain", '["default","auth:v1:x"]')], {}),
    "guard_first": ([plain(user_id="someone-else")], {"guard": True}),
    "stale_revision_first": ([plain(user_id="someone-else")], {"expected_revision": 3}),
    # The stored receipt answers before the guard and the claim, which would both refuse now.
    "replay_bypasses": ([plain(), commit_receipt(rt.Rename("Mine"), session_id="s-plain"),
                         raw("UPDATE sessions SET user_id='someone-else' WHERE id='s-plain'")], {"guard": True}),
}


HISTORY_OUTCOMES = {"claim_unbound": None, "claim_own_login": None, "replay_bypasses": None,
                    "guard_first": "runtime_coordination_required", "stale_revision_first": "revision_conflict"}


@pytest.mark.parametrize("case", sorted(HISTORY))
def test_history_claim_and_live_guard_fire_at_m1s_points(store, frozen, emitted, case):
    steps, options = HISTORY[case]
    seed_store(store, *steps)
    legacy_actor, actor = _actor(may_create=options.get("may_create", True))
    legacy_guard, verdict = guard()
    revision = options.get("expected_revision", 0)
    hooks = {"_authorize_write": _legacy_claim(legacy_actor), **({"_live_guard": legacy_guard} if options.get("guard") else {})}
    kwargs = {"history": rt.ClaimHistory(actor), **({"live_guard": verdict} if options.get("guard") else {})}
    new = compare(store, frozen, emitted,
                  lambda db: legacy_mutate(db, rt.Rename("Mine"), session_id="s-plain", expected_revision=revision, **hooks),
                  lambda: mutate(store, rt.Rename("Mine"), session_id="s-plain", expected_revision=revision, **kwargs),
                  replayed=case == "replay_bypasses")
    expect(new, HISTORY_OUTCOMES.get(case, "permission_denied"))


@pytest.mark.parametrize("case", ["guard", "walk_refuses_first"])
def test_delete_live_guard_follows_the_target_walk(store, frozen, emitted, case):
    from hermes_state_local import POLICY_PREFIX

    steps = [transcript()]
    if case == "walk_refuses_first":
        steps.append(lambda db: db.set_meta(POLICY_PREFIX + "s-plain", json.dumps(
            {"session_id": "s-plain", "profile_id": "default", "principal_id": "p", "request_id": "r", "entry": {"session_id": "s-plain"}})))
    seed_store(store, *steps)
    legacy_guard, verdict = guard()
    new = compare(store, frozen, emitted,
                  lambda db: legacy_mutate(db, rt.Delete(), session_id="s-plain", expected_generation=0, _live_guard=legacy_guard),
                  lambda: mutate(store, rt.Delete(), session_id="s-plain", expected_generation=0, live_guard=verdict))
    assert new[1].reason == ("storage_unavailable" if case == "walk_refuses_first" else "runtime_coordination_required")


# ── one transaction ───────────────────────────────────────────────────────────


def test_each_command_is_one_write_that_rolls_back_whole(store, frozen):
    seed_store(store, lineage, transcript(), reserved(make_binding()), legacy=False)
    second = _user_ids(store)[1]
    assert_one_write(store, lambda: mutate(store, rt.Sidebar(archived=True, pinned=True), session_id="s-tip"))
    assert_one_write(store, lambda: mutate(store, rt.Rewind(second), session_id="s-plain", expected_generation=0))
    assert_one_write(store, lambda: mutate(store, rt.Reset(), session_id=LOCAL, expected_generation=0))
    assert_one_write(store, lambda: mutate(store, rt.Delete(), session_id="s-plain", request_id="del", expected_revision=1, expected_generation=1))


def test_prepare_and_prepared_commit_are_one_write_each(store, frozen):
    seed_store(store, reserved(make_binding()), legacy=False)
    with counted_transactions(store.holder()) as counts:
        snapshot = store.runtime.prepare_mutation(1, principal_id=PRINCIPAL, session_id=LOCAL, request_id="req-m", expected_revision=0,
                                                  command=rt.EditPolicy("session"))
    assert counts == {"write": 1, "read": 0}
    values = {"policy": _policy(edit_approval_policy="session")}
    assert_one_write(store, lambda: mutate(store, rt.EditPolicy("session"), session_id=LOCAL, prepared=rt.PreparedMutation(snapshot, values)))
