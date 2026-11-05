"""Contract cases for the closed ``MutationCommand`` union: ``parse_mutation`` builds the typed command whose payload is the one it parsed, and the commands ``mutate`` runs."""

from __future__ import annotations

import pytest

from persistence import AuthorityRefusal, CapabilityUnsupported, FenceStale, ProofRejected, StaleFenceKind
from persistence.authority import runtime as rt
from persistence.authority.bindings import local_session_id
from persistence.conversation.messages import NewMessage
from persistence.conversation.sessions import NewSession

VALID = [
    ("model", {"model": "gpt", "provider": "openai", "reasoning": "high"}, rt.ModelPick("gpt", provider="openai", reasoning="high")),
    ("branch", {"title": "b", "through_message_id": 3, "mcp_servers": {"fingerprint": "f"}}, rt.Branch(title="b", through_message_id=3, mcp_servers_fingerprint="f")),
    ("reasoning", {"effort": None}, rt.ReasoningPick(None)),
    ("reasoning", {"effort": "XHigh"}, rt.ReasoningPick("XHigh")),
    ("workspace", {"cwd": "/w"}, rt.Workspace("/w")),
    ("editor_mcp", {"mcp_servers": {"fingerprint": "f"}}, rt.EditorMcp("f")),
    ("edit_policy", {"policy": "session"}, rt.EditPolicy("session")),
    ("compress", {"focus": "x", "partial": True, "keep_last": 2}, rt.Compress(focus="x", partial=True, keep_last=2)),
    ("delete", {}, rt.Delete()),
    ("reset", {}, rt.Reset()),
    ("import", {"sessions": [{"id": "s"}]}, rt.Import(({"id": "s"},))),
    ("rewind", {"target_message_id": 4, "preserve_compaction_handoff": False}, rt.Rewind(4, False)),
    ("rename", {"title": "t"}, rt.Rename("t")),
    ("archive", {"archived": True}, rt.Archive(True)),
    ("sidebar", {"pinned": True, "unread": False}, rt.Sidebar(pinned=True, unread=False)),
]

INVALID = [
    ("model", {"model": " "}), ("model", {"model": "m", "reasoning": "hgih"}), ("model", {"model": "m", "extra": "x"}),
    ("branch", {"through_message_id": 0}), ("branch", {"mcp_servers": {"fingerprint": 1}}), ("reasoning", {"effort": "loud"}),
    ("workspace", {"cwd": "  "}), ("edit_policy", {"policy": "never"}), ("compress", {"keep_last": 2}), ("compress", {"preview": 1}),
    ("delete", {"x": 1}), ("import", {"sessions": {}}), ("rewind", {"target_message_id": True}), ("rename", {"title": 1}),
    ("archive", {"archived": 1}), ("sidebar", {}), ("sidebar", {"color": "red"}), ("unknown", {}), (None, {}), ("rename", ["title"]),
]


@pytest.mark.parametrize(("operation", "payload", "command"), VALID)
def test_parse_mutation_builds_the_typed_command_whose_payload_round_trips(operation, payload, command):
    parsed = rt.parse_mutation(operation, payload)
    assert parsed == command and parsed.operation == operation and parsed.payload() == payload


@pytest.mark.parametrize(("operation", "payload"), INVALID)
def test_parse_mutation_refuses_what_validate_action_refuses(operation, payload):
    with pytest.raises(AuthorityRefusal, match="^invalid_params$"):
        rt.parse_mutation(operation, payload)


def test_the_union_is_closed_over_fourteen_operations():
    assert len({command.operation for _op, _payload, command in VALID}) == 14
    assert {cls.operation for cls in rt.MutationCommand.__args__} == {command.operation for _op, _payload, command in VALID}


def _local(store, request_id: str = "create-1", **policy) -> str:
    sid = str(local_session_id(store.owner_profile, "p", request_id))
    store.sessions.create(NewSession(session_id=sid, source="tui"))
    route = "agent:main:local:" + sid
    entry = {"session_key": route, "session_id": sid, "created_at": "2026-10-27T09:00:00", "updated_at": "2026-10-27T09:00:00"}
    store.runtime.seed_local_receipt(sid, {"session_id": sid, "profile_id": store.owner_profile, "principal_id": "p", "request_id": request_id,
                                           "route": route, "entry": entry, "lineage": [sid],
                                           "policy": {"source": "tui", "model": "m1", "config_json": '{"model": {"provider": "p1"}}', "cwd": "/w", **policy}})
    return sid


def _commit(store, epoch, sid, command, values, *, request_id="req", expected_generation=None):
    snapshot = store.runtime.prepare_mutation(epoch, principal_id="p", session_id=sid, request_id=request_id, expected_revision=store.sessions.get(sid).runtime_revision,
                                              command=command, expected_generation=expected_generation)
    row = store.sessions.get(sid)
    return store.runtime.mutate(epoch, principal_id="p", session_id=sid, request_id=request_id, expected_revision=row.runtime_revision, command=command,
                                expected_generation=expected_generation, prepared=rt.PreparedMutation(snapshot, values), owner=None)


def test_model_commits_the_prepared_policy_and_supersedes_a_deferred_pick(backend, scope_a):
    store = backend.open(scope_a)
    epoch = store.runtime.begin_epoch(instance_id="instance-cmd")
    sid = _local(store)
    key, digest = rt.mutation_receipt_key("p", sid, "d", operation="model", payload={"model": "old"}, expected_revision=0)
    store.runtime.defer_model_pick(epoch, key=key, digest=digest, session_id=sid, pick=rt.DeferredPick("model", "p", "d", {"model": "old"}))
    policy = {"model": "m2", "config_json": '{"model": {"provider": "p2"}}', "cwd": "/w"}
    outcome = _commit(store, epoch, sid, rt.ModelPick("m2"), {"policy": policy}, expected_generation=0)
    assert {k: outcome.result[k] for k in ("model", "provider", "execution_generation")} == {"model": "m2", "provider": "p2", "execution_generation": 1}
    assert store.sessions.get(sid).model == "m2" and store.runtime.deferred_pick(sid) is None


def test_prepared_commits_refuse_without_a_snapshot_or_after_it_moved(backend, scope_a):
    store = backend.open(scope_a)
    epoch = store.runtime.begin_epoch(instance_id="instance-cmd")
    sid = _local(store)
    with pytest.raises(AuthorityRefusal, match="^runtime_coordination_required$"):
        store.runtime.mutate(epoch, principal_id="p", session_id=sid, request_id="r", expected_revision=0, command=rt.EditPolicy("ask"), owner=None)
    snapshot = store.runtime.prepare_mutation(epoch, principal_id="p", session_id=sid, request_id="r", expected_revision=0, command=rt.EditPolicy("ask"))
    store.runtime.commit_refreshed_policy(epoch, session_id=sid, old=snapshot.receipt["policy"], new={**snapshot.receipt["policy"], "cwd": "/v"}, generation=0)
    with pytest.raises(FenceStale) as moved:
        store.runtime.mutate(epoch, principal_id="p", session_id=sid, request_id="r", expected_revision=0, command=rt.EditPolicy("ask"),
                             prepared=rt.PreparedMutation(snapshot, {"policy": snapshot.receipt["policy"]}), owner=None)
    assert moved.value.kind is StaleFenceKind.REVISION_CONFLICT
    assert not store.runtime.has_mutation_receipt("p", sid, "r")


def test_workspace_editor_mcp_and_edit_policy_projections(backend, scope_a):
    store = backend.open(scope_a)
    epoch = store.runtime.begin_epoch(instance_id="instance-cmd")
    sid = _local(store)
    base = store.runtime.prepare_mutation(epoch, principal_id="p", session_id=sid, request_id="w", expected_revision=0, command=rt.Workspace("/v")).receipt["policy"]
    moved = _commit(store, epoch, sid, rt.Workspace("/v"), {"policy": {**base, "cwd": "/v"}, "git": {"branch": "main", "root": ""}}, request_id="w")
    assert {k: moved.result[k] for k in ("cwd", "branch", "git_repo_root")} == {"cwd": "/v", "branch": "main", "git_repo_root": None}
    assert store.sessions.get(sid).cwd == "/v"
    policy = store.runtime.prepare_mutation(epoch, principal_id="p", session_id=sid, request_id="e", expected_revision=1, command=rt.EditorMcp("f")).receipt["policy"]
    editor = _commit(store, epoch, sid, rt.EditorMcp("f"), {"policy": policy, "tools_changed": True, "mcp_servers": ["fs"]}, request_id="e")
    assert (editor.result["tools_changed"], editor.result["mcp_servers"]) == (True, ["fs"])
    policy = store.runtime.prepare_mutation(epoch, principal_id="p", session_id=sid, request_id="a", expected_revision=2, command=rt.EditPolicy("session")).receipt["policy"]
    approval = _commit(store, epoch, sid, rt.EditPolicy("session"), {"policy": {**policy, "edit_approval_policy": "session"}}, request_id="a")
    assert approval.result["edit_approval_policy"] == "session" and approval.result["execution_generation"] == 3


def test_a_history_claim_on_a_row_without_claimable_history_refuses(backend, scope_a):
    store = backend.open(scope_a)
    epoch = store.runtime.begin_epoch(instance_id="instance-cmd")
    claim = rt.ClaimHistory(rt.HistoryActor("default", "user-1", None, True))
    store.sessions.create(NewSession(session_id="s-routed", source="cli", session_key="agent:main:cli:dm:c", chat_id="c"))
    store.sessions.create(NewSession(session_id="s-gateway", source="telegram"))
    store.sessions.create(NewSession(session_id="s-cold", source="cli"))
    # CONTRACT: a routed row, or an unrouted one outside the history sources, holds no history to claim, so the claim refuses.
    for sid in ("s-routed", "s-gateway"):
        with pytest.raises(ProofRejected):
            store.runtime.mutate(epoch, principal_id="p", session_id=sid, request_id="claim", expected_revision=0, command=rt.Rename("t"), history=claim, owner=None)
        assert store.sessions.get(sid).title is None and not store.runtime.has_mutation_receipt("p", sid, "claim")
    store.runtime.mutate(epoch, principal_id="p", session_id="s-cold", request_id="claim", expected_revision=0, command=rt.Rename("t"), history=claim, owner=None)
    assert store.sessions.get("s-cold").title == "t"


def test_a_history_claim_ignores_the_desktop_nous_account_label(backend, scope_a):
    store = backend.open(scope_a)
    epoch = store.runtime.begin_epoch(instance_id="instance-cmd")
    claim = rt.ClaimHistory(rt.HistoryActor("default", "user-1", None, True))
    store.sessions.create(NewSession(session_id="s-nous", source="desktop", user_id="nous:alice"))
    store.sessions.create(NewSession(session_id="s-bob", source="desktop", user_id="bob"))
    # CONTRACT: the Nous account Desktop stamps on its chats names no owner, so another subject may claim the row; any other user id is an owner the subject must match.
    store.runtime.mutate(epoch, principal_id="p", session_id="s-nous", request_id="claim", expected_revision=0, command=rt.Rename("t"), history=claim, owner=None)
    assert store.sessions.get("s-nous").title == "t"
    with pytest.raises(ProofRejected):
        store.runtime.mutate(epoch, principal_id="p", session_id="s-bob", request_id="claim", expected_revision=0, command=rt.Rename("t"), history=claim, owner=None)
    assert store.sessions.get("s-bob").title is None


def test_rewind_takes_back_the_transcript_from_the_user_row(backend, scope_a):
    store = backend.open(scope_a)
    epoch = store.runtime.begin_epoch(instance_id="instance-cmd")
    store.sessions.create(NewSession(session_id="s-rw", source="cli"))
    first = store.messages.append("s-rw", NewMessage(role="user", content="one"))
    store.messages.append("s-rw", NewMessage(role="assistant", content="reply"))
    target = store.messages.append("s-rw", NewMessage(role="user", content="two"))
    store.messages.append("s-rw", NewMessage(role="assistant", content="reply two"))

    def rewind(message_id, request_id):
        row = store.sessions.get("s-rw")
        return store.runtime.mutate(epoch, principal_id="p", session_id="s-rw", request_id=request_id, expected_revision=row.runtime_revision,
                                    command=rt.Rewind(int(message_id)), expected_generation=row.runtime_generation, owner=None)

    outcome = rewind(target, "rw-1")
    assert (outcome.result["rewound_count"], outcome.result["new_head_id"], outcome.result["replacement_message_id"]) == (2, int(first) + 1, None)
    assert outcome.result["target_message"]["content"] == "two" and outcome.result["execution_generation"] == 1
    assert [r.content for r in store.messages.read_transcript("s-rw")] == ["one", "reply"]
    with pytest.raises(AuthorityRefusal) as gone:
        rewind(target, "rw-2")
    assert (gone.value.reason, gone.value.field, gone.value.detail) == ("invalid_params", "target_message_id", "target user message is no longer in session history")


def test_a_rewind_receipt_carries_the_typed_text_of_a_withheld_prompt(backend, scope_a):
    # CONTRACT: the receipt's target_message content swaps the withheld placeholder for display_metadata's withheld_prompt, so /retry resends and /undo restores what the user typed; the stored row keeps the placeholder.
    store = backend.open(scope_a)
    epoch = store.runtime.begin_epoch(instance_id="instance-cmd")
    store.sessions.create(NewSession(session_id="s-held", source="cli"))
    placeholder = "[Message withheld: a pre_llm_call policy hook blocked it.]"
    target = store.messages.append("s-held", NewMessage(role="user", content=placeholder, display_metadata={"withheld_prompt": "the typed ask"}))
    store.messages.append("s-held", NewMessage(role="assistant", content="blocked"))
    row = store.sessions.get("s-held")
    outcome = store.runtime.mutate(epoch, principal_id="p", session_id="s-held", request_id="rw-held", expected_revision=row.runtime_revision,
                                   command=rt.Rewind(int(target)), expected_generation=row.runtime_generation, owner=None)
    assert outcome.result["target_message"]["content"] == "the typed ask"
    assert outcome.result["target_message"]["display_metadata"] == {"withheld_prompt": "the typed ask"}


def test_import_creates_absent_rows_and_refuses_a_deleted_id(backend, scope_a):
    store = backend.open(scope_a)
    epoch = store.runtime.begin_epoch(instance_id="instance-cmd")
    store.sessions.create(NewSession(session_id="s-exists", source="cli"))
    payload = ({"id": "s-new", "source": "import", "messages": [{"role": "user", "content": "hi"}]}, {"id": "s-exists", "source": "import", "messages": []})
    outcome = store.runtime.mutate(epoch, principal_id="p", session_id="s-new", request_id="imp", expected_revision=0, command=rt.Import(payload), owner=None)
    assert {k: outcome.result[k] for k in ("ok", "imported", "skipped", "imported_ids", "skipped_ids", "detached", "errors")} == {
        "ok": True, "imported": 1, "skipped": 1, "imported_ids": ["s-new"], "skipped_ids": ["s-exists"], "detached": 0, "errors": []}
    assert store.sessions.get("s-new") is not None and [r.content for r in store.messages.read_transcript("s-new")] == ["hi"]
    with pytest.raises(AuthorityRefusal, match="^invalid_params$"):
        store.runtime.mutate(epoch, principal_id="p", session_id="s-other", request_id="imp-2", expected_revision=0, command=rt.Import(payload), owner=None)
    row = store.sessions.get("s-new")
    store.runtime.mutate(epoch, principal_id="p", session_id="s-new", request_id="del", expected_revision=row.runtime_revision,
                         expected_generation=row.runtime_generation, command=rt.Delete(), owner=None)
    with pytest.raises(AuthorityRefusal) as deleted:
        store.runtime.mutate(epoch, principal_id="p", session_id="s-new", request_id="imp-3", expected_revision=0, command=rt.Import(payload), owner=None)
    assert (deleted.value.reason, deleted.value.field) == ("admission_conflict", "sessions")


def test_reset_branch_and_compress_run_their_fences_first(backend, backend_name, scope_a):
    store = backend.open(scope_a)
    epoch = store.runtime.begin_epoch(instance_id="instance-cmd")
    store.sessions.create(NewSession(session_id="s-plain", source="cli"))
    for command in (rt.Reset(), rt.Branch()):
        with pytest.raises(AuthorityRefusal, match="^runtime_coordination_required$"):
            store.runtime.mutate(epoch, principal_id="p", session_id="s-plain", request_id="x", expected_revision=0, expected_generation=0, command=command, owner=None)
    sid = _local(store, editor_mcp_json='{"servers": 1}')
    # CONTRACT: a branch of a policy with editor servers needs the owner's prepared selection.
    with pytest.raises(AuthorityRefusal, match="^runtime_coordination_required$"):
        store.runtime.mutate(epoch, principal_id="p", session_id=sid, request_id="b", expected_revision=0, expected_generation=0, command=rt.Branch(), owner=None)
    with pytest.raises(AuthorityRefusal, match="^runtime_coordination_required$"):
        store.runtime.mutate(epoch, principal_id="p", session_id=sid, request_id="c", expected_revision=0, expected_generation=0, command=rt.Compress(), owner=None)
    store.admissions.admit(epoch=epoch, principal_id="p", session_id=sid, request_id="q", payload={"text": "q"})
    with pytest.raises(AuthorityRefusal, match="^session_busy$"):
        store.runtime.mutate(epoch, principal_id="p", session_id=sid, request_id="b", expected_revision=0, expected_generation=0, command=rt.Branch(), owner=None)
    reset = dict(principal_id="p", session_id=sid, request_id="r", expected_revision=0, expected_generation=0, command=rt.Reset(), owner=None)
    if backend_name == "in_memory":
        # Reference limit: past their fences the reference refuses the publication itself (module docstring of mutation_commands).
        with pytest.raises(CapabilityUnsupported):
            store.runtime.mutate(epoch, **reset)
        assert not store.runtime.has_mutation_receipt("p", sid, "r")
        return
    # CONTRACT: queued input does not hold a reset; it publishes a fresh child of the target.
    outcome = store.runtime.mutate(epoch, **reset)
    child = outcome.result["target_session_id"]
    assert (outcome.result["previous_target_session_id"], outcome.result["execution_generation"]) == (sid, 1) and child != sid
    assert store.sessions.get(sid).end_reason == "session_reset"
    assert store.sessions.get(child).parent_session_id == sid
    assert store.runtime.has_mutation_receipt("p", sid, "r")
