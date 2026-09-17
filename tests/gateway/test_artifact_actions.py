"""Tests for artifact backend intent invocation (V1 slice).

Covers:
- invoke() happy paths: non-destructive handler runs inline
- invoke() destructive handler: needs_confirmation + challenge issued
- confirm() happy path: challenge consumed, handler runs
- confirm() expired / wrong artifact: fails gracefully
- conflict: stale revision returns conflict without running handler
- idempotency: second invoke with same key returns cached result
- unsupported: unknown binding_id or unregistered intent
- tombstone handler: _deleted set in content, artifact updated
- refresh handler: returns succeeded (with/without maintainer)
- _tombstone_entity: dataset / map / model path coverage
"""

import json
import time

import pytest


@pytest.fixture()
def artifact_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    return tmp_path


@pytest.fixture(autouse=True)
def _clear_caches():
    """Reset in-memory stores between tests."""
    from tui_gateway import artifact_actions as aa
    aa._pending_challenges.clear()
    aa._idempotency_cache.clear()
    yield
    aa._pending_challenges.clear()
    aa._idempotency_cache.clear()


# ── helpers ────────────────────────────────────────────────────────────────


def _make_artifact(artifact_home, artifact_id="test-art", kind="dataset", actions=None):
    from tui_gateway import artifact_store as store
    content = json.dumps({"key": "name", "rows": [{"name": "Alice"}, {"name": "Bob"}]})
    stored = store.set_artifact(
        artifact_id=artifact_id,
        kind=kind,
        content=content,
        title="Test",
        updated_by="test",
        actions=actions,
    )
    return stored


# ── invoke ─────────────────────────────────────────────────────────────────


def test_invoke_non_destructive_handler_runs_inline(artifact_home):
    from tui_gateway import artifact_actions as aa

    actions = [{"type": "intent", "id": "do-refresh", "label": "Refresh",
                "intent": "artifact.refresh", "presentation": {"role": "normal"}}]
    stored = _make_artifact(artifact_home, actions=actions)

    result = aa.invoke(
        artifact_id="test-art",
        artifact_rev=stored["rev"],
        binding_id="do-refresh",
        entity_ref="",
        idempotency_key="key-1",
    )
    assert result["status"] == "succeeded"


def test_invoke_destructive_returns_needs_confirmation(artifact_home):
    from tui_gateway import artifact_actions as aa

    actions = [{"type": "intent", "id": "del-row", "label": "Delete",
                "intent": "artifact.entity.tombstone",
                "presentation": {"role": "destructive"}}]
    stored = _make_artifact(artifact_home, actions=actions)

    result = aa.invoke(
        artifact_id="test-art",
        artifact_rev=stored["rev"],
        binding_id="del-row",
        entity_ref="dataset/alice",
        idempotency_key="key-2",
    )
    assert result["status"] == "needs_confirmation"
    assert "challenge" in result
    assert "prompt" in result
    # §0.1: prompt must lead with the server-resolved intent name, not the
    # artifact-authored label ("Delete"). An artifact could label a destructive
    # binding "Refresh" to trick the user into confirming without knowing the
    # real operation; the intent name is resolved server-side and is trusted.
    prompt = result["prompt"]
    assert prompt.startswith("artifact.entity.tombstone"), (
        f"confirmation prompt must lead with intent name; got: {prompt!r}"
    )
    assert "Delete" not in prompt.split("\n")[0], (
        "artifact-authored label must not appear on the first line of the prompt"
    )


def test_confirmation_prompt_label_mismatch_shown_as_secondary(artifact_home):
    """A label that differs from the intent name appears as secondary text only."""
    from tui_gateway import artifact_actions as aa

    actions = [{"type": "intent", "id": "sneaky", "label": "Refresh",
                "intent": "artifact.entity.tombstone",
                "presentation": {"role": "destructive"}}]
    _make_artifact(artifact_home, actions=actions)

    result = aa.invoke(
        artifact_id="test-art",
        artifact_rev=1,
        binding_id="sneaky",
        entity_ref="alice",
        idempotency_key="key-label-test",
    )
    prompt = result["prompt"]
    lines = prompt.splitlines()
    # Intent name leads
    assert lines[0].startswith("artifact.entity.tombstone")
    # Artifact label present somewhere but not on line 0
    assert any("Refresh" in line for line in lines[1:]), (
        "artifact label should appear as secondary text"
    )


def test_invoke_conflict_stale_revision(artifact_home):
    from tui_gateway import artifact_actions as aa, artifact_store as store

    actions = [{"type": "intent", "id": "do-refresh", "label": "Refresh",
                "intent": "artifact.refresh", "presentation": {"role": "normal"}}]
    stored = _make_artifact(artifact_home, actions=actions)
    store.set_artifact("test-art", "dataset",
                       json.dumps({"key": "name", "rows": []}),
                       updated_by="other")  # bumps rev to 2

    result = aa.invoke(
        artifact_id="test-art",
        artifact_rev=stored["rev"],  # stale: still rev 1
        binding_id="do-refresh",
        entity_ref="",
        idempotency_key="key-3",
    )
    assert result["status"] == "conflict"


def test_invoke_unknown_binding_returns_unsupported(artifact_home):
    from tui_gateway import artifact_actions as aa

    stored = _make_artifact(artifact_home, actions=[])

    result = aa.invoke(
        artifact_id="test-art",
        artifact_rev=stored["rev"],
        binding_id="no-such-binding",
        entity_ref="",
        idempotency_key="key-4",
    )
    assert result["status"] == "unsupported"


def test_invoke_unregistered_intent_returns_unsupported(artifact_home):
    from tui_gateway import artifact_actions as aa

    actions = [{"type": "intent", "id": "custom", "label": "Custom",
                "intent": "my.custom.intent.not.registered",
                "presentation": {"role": "normal"}}]
    stored = _make_artifact(artifact_home, actions=actions)

    result = aa.invoke(
        artifact_id="test-art",
        artifact_rev=stored["rev"],
        binding_id="custom",
        entity_ref="",
        idempotency_key="key-5",
    )
    assert result["status"] == "unsupported"


def test_invoke_idempotency_key_returns_cached_result(artifact_home):
    from tui_gateway import artifact_actions as aa

    actions = [{"type": "intent", "id": "do-refresh", "label": "Refresh",
                "intent": "artifact.refresh", "presentation": {"role": "normal"}}]
    stored = _make_artifact(artifact_home, actions=actions)

    r1 = aa.invoke(
        artifact_id="test-art",
        artifact_rev=stored["rev"],
        binding_id="do-refresh",
        entity_ref="",
        idempotency_key="same-key",
    )
    # Mutate the artifact to bump rev — if idempotency works we still get r1
    from tui_gateway import artifact_store as store
    store.set_artifact("test-art", "dataset",
                       json.dumps({"key": "name", "rows": []}), updated_by="bump")

    r2 = aa.invoke(
        artifact_id="test-art",
        artifact_rev=stored["rev"] + 99,  # would conflict if not cached
        binding_id="do-refresh",
        entity_ref="",
        idempotency_key="same-key",
    )
    assert r1 == r2


# ── confirm ────────────────────────────────────────────────────────────────


def test_confirm_destructive_after_challenge(artifact_home):
    from tui_gateway import artifact_actions as aa

    actions = [{"type": "intent", "id": "del-row", "label": "Delete",
                "intent": "artifact.entity.tombstone",
                "presentation": {"role": "destructive"}}]
    stored = _make_artifact(artifact_home, actions=actions)

    invoke_result = aa.invoke(
        artifact_id="test-art",
        artifact_rev=stored["rev"],
        binding_id="del-row",
        entity_ref="alice",
        idempotency_key="key-c1",
    )
    assert invoke_result["status"] == "needs_confirmation"

    confirm_result = aa.confirm(
        artifact_id="test-art",
        challenge=invoke_result["challenge"],
    )
    assert confirm_result["status"] == "succeeded"


def test_confirm_expired_challenge_fails(artifact_home):
    from tui_gateway import artifact_actions as aa

    actions = [{"type": "intent", "id": "del-row", "label": "Delete",
                "intent": "artifact.entity.tombstone",
                "presentation": {"role": "destructive"}}]
    stored = _make_artifact(artifact_home, actions=actions)

    invoke_result = aa.invoke(
        artifact_id="test-art",
        artifact_rev=stored["rev"],
        binding_id="del-row",
        entity_ref="alice",
        idempotency_key="key-c2",
    )
    challenge = invoke_result["challenge"]

    # Manually expire the challenge by back-dating its expiry.
    aa._pending_challenges[challenge]["expires"] = time.monotonic() - 1
    result = aa.confirm(artifact_id="test-art", challenge=challenge)
    assert result["status"] == "failed"


def test_confirm_wrong_artifact_fails(artifact_home):
    from tui_gateway import artifact_actions as aa

    actions = [{"type": "intent", "id": "del-row", "label": "Delete",
                "intent": "artifact.entity.tombstone",
                "presentation": {"role": "destructive"}}]
    stored = _make_artifact(artifact_home, actions=actions)

    invoke_result = aa.invoke(
        artifact_id="test-art",
        artifact_rev=stored["rev"],
        binding_id="del-row",
        entity_ref="alice",
        idempotency_key="key-c3",
    )
    result = aa.confirm(
        artifact_id="other-artifact",  # wrong artifact
        challenge=invoke_result["challenge"],
    )
    assert result["status"] == "failed"


# ── tombstone handler ──────────────────────────────────────────────────────


def test_tombstone_handler_marks_dataset_row(artifact_home):
    from tui_gateway import artifact_actions as aa, artifact_store as store

    stored = _make_artifact(artifact_home)
    result = aa._handle_tombstone(
        artifact_id="test-art", binding_id="", entity_ref="alice"
    )
    assert result["status"] == "succeeded"

    updated = store.get_artifact("test-art")
    rows = json.loads(updated["content"])["rows"]
    alice = next(r for r in rows if r.get("name") == "Alice")
    assert alice["_deleted"] is True


def test_tombstone_handler_marks_map_marker(artifact_home):
    from tui_gateway import artifact_actions as aa, artifact_store as store

    content = json.dumps({"markers": [{"label": "Ekkamai", "lat": 13.72, "lon": 100.58}]})
    store.set_artifact("map-art", "map", content, updated_by="test")

    result = aa._handle_tombstone(
        artifact_id="map-art", binding_id="", entity_ref="ekkamai"
    )
    assert result["status"] == "succeeded"

    updated = store.get_artifact("map-art")
    markers = json.loads(updated["content"])["markers"]
    assert markers[0]["_deleted"] is True


def test_tombstone_handler_marks_model_entity(artifact_home):
    from tui_gateway import artifact_actions as aa, artifact_store as store

    content = json.dumps({
        "entities": {
            "issues": {"key": "id", "items": [{"id": "ISS-1"}, {"id": "ISS-2"}]}
        }
    })
    store.set_artifact("model-art", "model", content, updated_by="test")

    result = aa._handle_tombstone(
        artifact_id="model-art", binding_id="", entity_ref="issues/iss-1"
    )
    assert result["status"] == "succeeded"

    updated = store.get_artifact("model-art")
    items = json.loads(updated["content"])["entities"]["issues"]["items"]
    iss1 = next(i for i in items if i.get("id") == "ISS-1")
    assert iss1["_deleted"] is True


# ── refresh handler ────────────────────────────────────────────────────────


def test_refresh_with_no_maintainer(artifact_home):
    from tui_gateway import artifact_actions as aa

    _make_artifact(artifact_home)
    result = aa._handle_refresh(artifact_id="test-art", binding_id="", entity_ref="")
    assert result["status"] == "succeeded"
    assert "No maintainer" in result.get("message", "")


def test_refresh_with_maintainer_declared(artifact_home):
    from tui_gateway import artifact_actions as aa, artifact_store as store

    content = json.dumps({
        "maintainers": ["cron:abc123"],
        "key": "name", "rows": [],
    })
    store.set_artifact("test-art", "dataset", content, updated_by="test")

    result = aa._handle_refresh(artifact_id="test-art", binding_id="", entity_ref="")
    assert result["status"] == "succeeded"
    assert "maintainer" in result.get("message", "").lower()


# ── session-spawn handler ──────────────────────────────────────────────────


class _FakeMethods(dict):
    """Stand-in for server._methods that records session.create / prompt.submit
    calls and hands back a fixed live session_id — no real runtime spun up.

    ``session.create`` returns both the short runtime ``session_id`` and the
    long ``stored_session_id`` (the stable database key) so the spawn handler's
    id selection can be exercised. Pass ``stored_session_id=None`` to model an
    older runtime that omits the database id (fallback path)."""

    def __init__(self, session_id="abc12345",
                 stored_session_id="20260101_000000_abcdef"):
        super().__init__()
        self.session_id = session_id
        self.stored_session_id = stored_session_id
        self.created = []
        self.submitted = []
        self["session.create"] = self._create
        self["prompt.submit"] = self._submit

    def _create(self, rid, params):
        self.created.append(params)
        result = {"session_id": self.session_id}
        if self.stored_session_id is not None:
            result["stored_session_id"] = self.stored_session_id
        return {"result": result}

    def _submit(self, rid, params):
        self.submitted.append(params)
        return {"result": {"status": "streaming"}}


@pytest.fixture()
def fake_server_methods(monkeypatch):
    from tui_gateway import server
    fake = _FakeMethods()
    monkeypatch.setattr(server, "_methods", fake, raising=False)
    return fake


def test_session_spawn_returns_live_session_id(artifact_home, fake_server_methods):
    from tui_gateway import artifact_actions as aa

    actions = [{"type": "intent", "id": "investigate", "label": "Investigate",
                "intent": "artifact.session.spawn",
                "session_prompt": "Investigate this issue.",
                "presentation": {"role": "normal"}}]
    stored = _make_artifact(artifact_home, actions=actions)

    result = aa.invoke(
        artifact_id="test-art",
        artifact_rev=stored["rev"],
        binding_id="investigate",
        entity_ref="alice",
        idempotency_key="key-spawn-1",
    )
    assert result["status"] == "succeeded"
    # The stable database id flows back so the client can click through: its
    # navigation resolves against session.list rows, which carry only the
    # database id — never the short runtime id.
    assert result["session_id"] == "20260101_000000_abcdef"
    assert fake_server_methods.created, "a session should have been created"
    # …but the task is seeded against the short runtime id, the correct handle
    # for the in-memory prompt.submit dispatch.
    assert fake_server_methods.submitted[0]["session_id"] == "abc12345"


def test_session_spawn_falls_back_to_runtime_id_without_db_id(artifact_home, monkeypatch):
    """A runtime that returns no stored_session_id (older gateway) still yields
    a usable click-through id — the short runtime id."""
    from tui_gateway import server, artifact_actions as aa

    fake = _FakeMethods(stored_session_id=None)
    monkeypatch.setattr(server, "_methods", fake, raising=False)

    actions = [{"type": "intent", "id": "investigate", "label": "Investigate",
                "intent": "artifact.session.spawn",
                "session_prompt": "Investigate this issue.",
                "presentation": {"role": "normal"}}]
    stored = _make_artifact(artifact_home, actions=actions)

    result = aa.invoke(
        artifact_id="test-art",
        artifact_rev=stored["rev"],
        binding_id="investigate",
        entity_ref="alice",
        idempotency_key="key-spawn-fallback",
    )
    assert result["status"] == "succeeded"
    assert result["session_id"] == "abc12345"


def test_session_spawn_seeds_task_from_stored_entity_not_raw_ref(
    artifact_home, fake_server_methods
):
    """§0.2: the seeded task is composed from the resolved stored entity, and
    the raw client entity_ref is not spliced in as an instruction."""
    from tui_gateway import artifact_actions as aa

    actions = [{"type": "intent", "id": "investigate", "label": "Investigate",
                "intent": "artifact.session.spawn",
                "session_prompt": "Investigate this row.",
                "presentation": {"role": "normal"}}]
    stored = _make_artifact(artifact_home, actions=actions)

    aa.invoke(
        artifact_id="test-art",
        artifact_rev=stored["rev"],
        binding_id="investigate",
        entity_ref="alice",  # lookup key; matches row {"name": "Alice"}
        idempotency_key="key-spawn-2",
    )
    assert fake_server_methods.submitted, "an initial task should be seeded"
    task = fake_server_methods.submitted[0]["text"]
    assert "Investigate this row." in task           # author template
    assert "Alice" in task                            # resolved stored field
    assert "resolved from stored content" in task     # provenance marker


def test_session_spawn_includes_required_human_context(
    artifact_home, fake_server_methods
):
    from tui_gateway import artifact_actions as aa

    actions = [{"type": "intent", "id": "fix", "label": "Fix",
                "intent": "artifact.session.spawn",
                "session_prompt": "Resolve the open application questions.",
                "presentation": {"role": "normal", "context": "required"}}]
    stored = _make_artifact(artifact_home, actions=actions)

    with pytest.raises(ValueError, match="user_context is required"):
        aa.invoke(
            artifact_id="test-art", artifact_rev=stored["rev"],
            binding_id="fix", entity_ref="alice",
            idempotency_key="context-required",
        )

    result = aa.invoke(
        artifact_id="test-art", artifact_rev=stored["rev"],
        binding_id="fix", entity_ref="alice",
        idempotency_key="context-required",
        user_context="  Use my answer for the relocation question.  ",
    )

    assert result["status"] == "succeeded"
    task = fake_server_methods.submitted[0]["text"]
    assert "Human-provided context for this run" in task
    assert "Use my answer for the relocation question." in task


def test_user_context_is_bounded_and_session_only(artifact_home):
    from tui_gateway import artifact_actions as aa

    actions = [{"type": "intent", "id": "refresh", "intent": "artifact.refresh"}]
    stored = _make_artifact(artifact_home, actions=actions)

    with pytest.raises(ValueError, match="only for artifact.session.spawn"):
        aa.invoke(
            artifact_id="test-art", artifact_rev=stored["rev"],
            binding_id="refresh", entity_ref="", idempotency_key="wrong-intent",
            user_context="Do something unrelated.",
        )

    with pytest.raises(ValueError, match="4000 UTF-8 bytes"):
        aa.invoke(
            artifact_id="test-art", artifact_rev=stored["rev"],
            binding_id="refresh", entity_ref="", idempotency_key="too-large",
            user_context="x" * 4_001,
        )

    with pytest.raises(ValueError, match="4000 UTF-8 bytes"):
        aa.invoke(
            artifact_id="test-art", artifact_rev=stored["rev"],
            binding_id="refresh", entity_ref="", idempotency_key="whitespace-large",
            user_context=" " * 4_001,
        )

    with pytest.raises(ValueError, match="control character"):
        aa.invoke(
            artifact_id="test-art", artifact_rev=stored["rev"],
            binding_id="refresh", entity_ref="", idempotency_key="control-char",
            user_context="answer\u0085separator",
        )

    assert aa._normalize_user_context("line one\nline two") == "line one\nline two"


def test_human_context_participates_in_idempotency_scope(
    artifact_home, fake_server_methods
):
    from tui_gateway import artifact_actions as aa

    actions = [{"type": "intent", "id": "fix", "label": "Fix",
                "intent": "artifact.session.spawn",
                "session_prompt": "Resolve the open questions.",
                "presentation": {"role": "normal", "context": "optional"}}]
    stored = _make_artifact(artifact_home, actions=actions)

    for context in ("First answer", "First answer", "Corrected answer", None):
        result = aa.invoke(
            artifact_id="test-art", artifact_rev=stored["rev"],
            binding_id="fix", entity_ref="alice", idempotency_key="same-key",
            user_context=context,
        )
        assert result["status"] == "succeeded"

    assert len(fake_server_methods.created) == 3
    assert "First answer" in fake_server_methods.submitted[0]["text"]
    assert "Corrected answer" in fake_server_methods.submitted[1]["text"]


def test_context_idempotency_replays_complete_result_from_ledger(
    artifact_home, fake_server_methods
):
    from tui_gateway import artifact_actions as aa

    actions = [{"type": "intent", "id": "fix", "label": "Fix",
                "intent": "artifact.session.spawn",
                "presentation": {"role": "normal", "context": "optional"}}]
    stored = _make_artifact(artifact_home, actions=actions)
    kwargs = {
        "artifact_id": "test-art",
        "artifact_rev": stored["rev"],
        "binding_id": "fix",
        "entity_ref": "alice",
        "idempotency_key": "durable-context",
        "user_context": "Use my answer.",
    }

    first = aa.invoke(**kwargs)
    aa._idempotency_cache.clear()
    replay = aa.invoke(**kwargs)

    assert replay == first
    assert replay["session_id"] == "20260101_000000_abcdef"
    assert len(fake_server_methods.created) == 1


def test_destructive_session_preserves_context_through_confirmation(
    artifact_home, fake_server_methods
):
    from tui_gateway import artifact_actions as aa

    actions = [{"type": "intent", "id": "fix", "label": "Fix",
                "intent": "artifact.session.spawn.with_context",
                "session_prompt": "Resolve the open questions.",
                "presentation": {"role": "destructive", "context": "required"}}]
    stored = _make_artifact(artifact_home, actions=actions)

    invoked = aa.invoke(
        artifact_id="test-art", artifact_rev=stored["rev"],
        binding_id="fix", entity_ref="alice", idempotency_key="confirmed-context",
        user_context="My confirmed answer.",
    )
    assert invoked["status"] == "needs_confirmation"
    assert not fake_server_methods.created

    result = aa.confirm("test-art", invoked["challenge"])

    assert result["status"] == "succeeded"
    assert "My confirmed answer." in fake_server_methods.submitted[0]["text"]


def test_context_confirmation_rejects_changed_artifact_revision(
    artifact_home, fake_server_methods
):
    from tui_gateway import artifact_actions as aa, artifact_store as store

    actions = [{"type": "intent", "id": "fix", "label": "Fix",
                "intent": "artifact.session.spawn.with_context",
                "presentation": {"role": "destructive", "context": "required"}}]
    stored = _make_artifact(artifact_home, actions=actions)
    invoked = aa.invoke(
        artifact_id="test-art", artifact_rev=stored["rev"],
        binding_id="fix", entity_ref="alice", idempotency_key="changed-revision",
        user_context="Sensitive application answer.",
    )
    store.set_artifact(
        "test-art", "dataset", stored["content"], updated_by="changed",
        replace=True, actions=actions,
    )

    result = aa.confirm("test-art", invoked["challenge"])

    assert result["status"] == "conflict"
    assert not fake_server_methods.created


def test_action_invoke_rpc_forwards_human_context(monkeypatch):
    from tui_gateway import artifact_actions as aa, server

    captured = {}

    def fake_invoke(**kwargs):
        captured.update(kwargs)
        return {"status": "unsupported"}

    monkeypatch.setattr(aa, "invoke", fake_invoke)
    response = server._methods["artifact.action.invoke"]("rpc-context", {
        "artifact_id": "test-art",
        "artifact_rev": 3,
        "binding_id": "fix",
        "entity_ref": "alice",
        "idempotency_key": "rpc-key",
        "user_context": "My application answer.",
    })

    assert response["result"]["status"] == "unsupported"
    assert captured["user_context"] == "My application answer."


def test_session_spawn_unresolved_entity_fails_closed(
    artifact_home, fake_server_methods
):
    """An entity_ref that resolves to no stored entity must fail, not spawn a
    session pointed at an attacker-controlled string."""
    from tui_gateway import artifact_actions as aa

    actions = [{"type": "intent", "id": "investigate", "label": "Investigate",
                "intent": "artifact.session.spawn",
                "presentation": {"role": "normal"}}]
    stored = _make_artifact(artifact_home, actions=actions)

    result = aa.invoke(
        artifact_id="test-art",
        artifact_rev=stored["rev"],
        binding_id="investigate",
        entity_ref="nonesuch",  # not a row in the artifact
        idempotency_key="key-spawn-3",
    )
    assert result["status"] == "failed"
    assert not fake_server_methods.created, "no session should be created"


def test_session_spawn_artifact_scoped_needs_no_entity(
    artifact_home, fake_server_methods
):
    """With no entity_ref the intent is artifact-scoped and still spawns."""
    from tui_gateway import artifact_actions as aa

    actions = [{"type": "intent", "id": "summarize", "label": "Summarize",
                "intent": "artifact.session.spawn",
                "session_prompt": "Summarize this artifact.",
                "presentation": {"role": "normal"}}]
    stored = _make_artifact(artifact_home, actions=actions)

    result = aa.invoke(
        artifact_id="test-art",
        artifact_rev=stored["rev"],
        binding_id="summarize",
        entity_ref="",
        idempotency_key="key-spawn-4",
    )
    assert result["status"] == "succeeded"
    assert result["session_id"] == "20260101_000000_abcdef"


def test_session_spawn_destructive_requires_confirmation(
    artifact_home, fake_server_methods
):
    """Session spawning honors the confirmation gate like any other intent:
    a destructive binding must confirm before the session is created."""
    from tui_gateway import artifact_actions as aa

    actions = [{"type": "intent", "id": "purge", "label": "Purge",
                "intent": "artifact.session.spawn",
                "session_prompt": "Purge this row.",
                "presentation": {"role": "destructive"}}]
    stored = _make_artifact(artifact_home, actions=actions)

    invoke_result = aa.invoke(
        artifact_id="test-art",
        artifact_rev=stored["rev"],
        binding_id="purge",
        entity_ref="alice",
        idempotency_key="key-spawn-5",
    )
    assert invoke_result["status"] == "needs_confirmation"
    # No session created before confirmation.
    assert not fake_server_methods.created

    confirm_result = aa.confirm(
        artifact_id="test-art",
        challenge=invoke_result["challenge"],
    )
    assert confirm_result["status"] == "succeeded"
    assert confirm_result["session_id"] == "20260101_000000_abcdef"
    assert fake_server_methods.created, "session created only after confirm"


# ── actions persisted through artifact store ──────────────────────────────


def test_actions_persist_and_carry_forward(artifact_home):
    from tui_gateway import artifact_store as store

    actions = [{"type": "intent", "id": "refresh", "label": "Refresh",
                "intent": "artifact.refresh"}]
    stored = _make_artifact(artifact_home, actions=actions)
    assert stored.get("actions") == actions

    # Update without supplying actions — should carry forward
    updated = store.set_artifact(
        "test-art", "dataset",
        json.dumps({"key": "name", "rows": []}),
        updated_by="agent",
    )
    assert updated.get("actions") == actions


def test_actions_overwritten_when_supplied(artifact_home):
    from tui_gateway import artifact_store as store

    original_actions = [{"type": "delete"}]
    _make_artifact(artifact_home, actions=original_actions)

    new_actions = [{"type": "toggle", "field": "done"}]
    updated = store.set_artifact(
        "test-art", "dataset",
        json.dumps({"key": "name", "rows": []}),
        updated_by="agent",
        actions=new_actions,
    )
    assert updated.get("actions") == new_actions