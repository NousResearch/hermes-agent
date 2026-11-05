"""Contract suite for ``SessionRepository``.

Runs the real fixture backend end to end with the following validity predicate: records are seeded and observed through public protocol operations only, mutation tests assert pre-state, returned result, and post-state, refused operations prove non-mutation, and two-profile cases seed colliding domain ids with distinct sentinels in both scopes.

The behaviours this suite leaves out are documented in ``persistence/conversation/sessions.py``'s module docstring (durability across handle reopen, profile stamping and the key-namespace fence, the YOLO projection, message-derived listing facets including preview / min_message_count / MRU ordering / the message-blocks arms of the empty-session surfaces, ``compact_rows``, and unknown title-provenance strings, which ``TitleSource`` makes unrepresentable). The reopened-reset-parent child staying listable is held structurally by ``SessionLineageKind`` and still covered behaviorally below; ``set_auto_title`` rejecting USER provenance keeps the provenance ladder's behavioral arm tested.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from persistence import AuthorityRefusal, Conflict, PermanentStoreError, RetiredSessionError, SessionId, UnknownExecution
from persistence.conversation.sessions import (
    CANONICAL_BOT_CHAT_TITLE,
    MAX_TITLE_LENGTH,
    RECOVERABLE_END_REASONS,
    RETAG_GATE_PREFIX,
    RETAG_SOURCE_FROM,
    RETAG_SOURCE_TO,
    ArchiveChange,
    NewSession,
    ReplaceSessionConfig,
    SessionCountQuery,
    SessionLineageKind,
    SessionOrder,
    SessionQuery,
    SessionWindow,
    TitleConflict,
    TitleSource,
)
from tests.contracts.conftest import require_capability, require_failpoints
from tests.contracts.conversation._session_probes import is_retired
from tests.contracts._scopes import MEDIA_ROOT


# One representative per end_reason class the store distinguishes.
END_REASONS = [
    "cli_close", "compression", "session_reset", "branched",
    "agent_close", "ws_orphan_reap", "superseded_by_resume", "startup_orphan_reap",
    "session_switch", "idle", "tui_shutdown", "custom_reason",
]

NON_RECOVERABLE_END_REASONS = [
    "cli_close", "compression", "branched", "session_reset", "tui_shutdown", "custom_reason",
]


def _new(session_id: str, source: str = "cli", **kwargs) -> NewSession:
    return NewSession(session_id=SessionId(session_id), source=source, **kwargs)


def _ids(backend, scope, query: SessionQuery = SessionQuery(), limit: int = 100) -> set:
    store = backend.open(scope)
    return {entry.session.session_id for entry in store.sessions.list_sessions(query, SessionWindow(limit=limit))}


def _busy(refused) -> bool:
    return isinstance(refused.value, AuthorityRefusal) and refused.value.reason == "session_busy"


# ── Transition table ─────────────────────────────────────────────


def test_t1_create_returns_the_id_and_row_is_live(backend, scope_a):
    store_a = backend.open(scope_a)
    created = store_a.sessions.create(_new("sess-t1", source="cli", model="m1", cwd="/w1"))
    # CONTRACT: create returns the id and the row is immediately readable and live.
    assert created == "sess-t1"
    row = store_a.sessions.get(SessionId("sess-t1"))
    assert row is not None
    assert (row.source, row.model, row.cwd) == ("cli", "m1", "/w1")
    assert row.ended_at is None
    assert row.end_reason is None
    assert row.started_at is not None


def test_t2_duplicate_create_keeps_first_writers_fields(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.create(
        _new("sess-t2", source="cli", model="m1", cwd="/first", user_id="u1")
    )
    store_a.sessions.create(
        _new("sess-t2", source="telegram", model="m2", cwd="/second", user_id="u2")
    )
    row = store_a.sessions.get(SessionId("sess-t2"))
    # CONTRACT: duplicate create is an upsert that never overwrites what an earlier writer set.
    assert (row.model, row.cwd, row.user_id, row.source) == ("m1", "/first", "u1", "cli")
    # CONTRACT: a NULL field left by the first writer is filled by a later create.
    store_a.sessions.create(_new("sess-t2b"))
    store_a.sessions.create(_new("sess-t2b", model="late-model"))
    later = store_a.sessions.get(SessionId("sess-t2b"))
    assert later.model == "late-model"


@pytest.mark.parametrize("reason", END_REASONS)
def test_t3_end_stamps_reason(backend, scope_a, reason):
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("sess-t3"))
    store_a.sessions.end(SessionId("sess-t3"), reason)
    row = store_a.sessions.get(SessionId("sess-t3"))
    # CONTRACT: ending a live session stamps ended_at and the given end_reason.
    assert row.ended_at is not None
    assert row.end_reason == reason


def test_t4_repeated_end_first_reason_wins(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("sess-t4"))
    store_a.sessions.end(SessionId("sess-t4"), "compression")
    first = store_a.sessions.get(SessionId("sess-t4"))
    store_a.sessions.end(SessionId("sess-t4"), "cli_close")
    row = store_a.sessions.get(SessionId("sess-t4"))
    # CONTRACT: end on an already-ended row is a no-op; first reason and stamp survive.
    assert row.end_reason == "compression"
    assert row.ended_at == first.ended_at


@pytest.mark.parametrize("reason", END_REASONS)
def test_t5_reopen_clears_end_fields(backend, scope_a, reason):
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("sess-t5"))
    store_a.sessions.end(SessionId("sess-t5"), reason)
    store_a.sessions.reopen(SessionId("sess-t5"))
    row = store_a.sessions.get(SessionId("sess-t5"))
    # CONTRACT: reopen clears both end fields regardless of the prior reason.
    assert row.ended_at is None
    assert row.end_reason is None
    # CONTRACT: a reopened session can be ended again with a new reason.
    store_a.sessions.end(SessionId("sess-t5"), "cli_close")
    row = store_a.sessions.get(SessionId("sess-t5"))
    assert row.end_reason == "cli_close"


def test_t6_reopen_live_session_is_noop(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("sess-t6"))
    before = store_a.sessions.get(SessionId("sess-t6"))
    store_a.sessions.reopen(SessionId("sess-t6"))
    after = store_a.sessions.get(SessionId("sess-t6"))
    # CONTRACT: reopening a live session changes nothing observable.
    assert after == before


def test_t7_t8_end_and_reopen_missing_are_silent_noops(backend, scope_a):
    # CONTRACT: ending or reopening an unknown id neither raises nor creates a row.
    store_a = backend.open(scope_a)
    store_a.sessions.end(SessionId("sess-t7-missing"), "cli_close")
    assert store_a.sessions.get(SessionId("sess-t7-missing")) is None
    store_a.sessions.reopen(SessionId("sess-t8-missing"))
    assert store_a.sessions.get(SessionId("sess-t8-missing")) is None


def test_t9_promote_live_session_to_reset(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("sess-t9"))
    # CONTRACT: promoting a live row ends it with the reset reason and reports True.
    assert store_a.sessions.promote_to_reset(SessionId("sess-t9")) is True
    row = store_a.sessions.get(SessionId("sess-t9"))
    assert row.end_reason == "session_reset"
    assert row.ended_at is not None


@pytest.mark.parametrize("reason", sorted(RECOVERABLE_END_REASONS))
def test_t10_promote_overwrites_recoverable_end(backend, scope_a, reason):
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("sess-t10"))
    store_a.sessions.end(SessionId("sess-t10"), reason)
    before = store_a.sessions.get(SessionId("sess-t10"))
    # CONTRACT: a recoverable accidental end is overwritten by an intentional reset boundary with a fresh stamp.
    assert store_a.sessions.promote_to_reset(SessionId("sess-t10")) is True
    row = store_a.sessions.get(SessionId("sess-t10"))
    assert row.end_reason == "session_reset"
    assert row.ended_at >= before.ended_at


@pytest.mark.parametrize("reason", NON_RECOVERABLE_END_REASONS)
def test_t11_promote_preserves_non_recoverable_end(backend, scope_a, reason):
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("sess-t11"))
    store_a.sessions.end(SessionId("sess-t11"), reason)
    before = store_a.sessions.get(SessionId("sess-t11"))
    # CONTRACT: promotion overwrites exactly the recoverable set; any other reason survives and the call reports False.
    assert store_a.sessions.promote_to_reset(SessionId("sess-t11")) is False
    row = store_a.sessions.get(SessionId("sess-t11"))
    assert row.end_reason == reason
    assert row.ended_at == before.ended_at


def test_t12_promote_missing_id_returns_false(backend, scope_a):
    # CONTRACT: promotion of an unknown id is False without creating a row.
    store_a = backend.open(scope_a)
    assert store_a.sessions.promote_to_reset(SessionId("sess-t12-missing")) is False
    assert store_a.sessions.get(SessionId("sess-t12-missing")) is None


# ── Titles ───────────────────────────────────────────────────────────────────


def test_title_set_get_and_sanitize(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("sess-ti1"))
    # CONTRACT: a user title lands; whitespace collapses, control characters strip.
    assert store_a.sessions.set_title(SessionId("sess-ti1"), "  my \t proj\x00ect  ") is True
    assert store_a.sessions.get_title(SessionId("sess-ti1")) == "my project"
    assert store_a.sessions.get_title_source(SessionId("sess-ti1")) is TitleSource.USER
    # CONTRACT: an empty title clears both title and provenance.
    assert store_a.sessions.set_title(SessionId("sess-ti1"), "") is True
    assert store_a.sessions.get_title(SessionId("sess-ti1")) is None
    assert store_a.sessions.get_title_source(SessionId("sess-ti1")) is None


def test_title_length_cap_raises_and_writes_nothing(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("sess-ti2"))
    # CONTRACT: a title past the cap raises ValueError and writes nothing.
    with pytest.raises(ValueError):
        store_a.sessions.set_title(SessionId("sess-ti2"), "x" * (MAX_TITLE_LENGTH + 1))
    assert store_a.sessions.get_title(SessionId("sess-ti2")) is None


def test_title_provenance_ladder(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("sess-ti3"))
    sid = SessionId("sess-ti3")
    # CONTRACT: derived lands on an untitled row; llm upgrades derived; a second llm write is refused; user overwrites everything; autos never overwrite user.
    assert store_a.sessions.set_auto_title(sid, "derived name", source=TitleSource.DERIVED) is True
    assert store_a.sessions.set_auto_title(sid, "llm name", source=TitleSource.LLM) is True
    assert store_a.sessions.set_auto_title(sid, "llm again", source=TitleSource.LLM) is False
    assert store_a.sessions.get_title(sid) == "llm name"
    assert store_a.sessions.set_title(sid, "user name") is True
    assert store_a.sessions.set_auto_title(sid, "llm later", source=TitleSource.LLM) is False
    assert store_a.sessions.get_title(sid) == "user name"
    assert store_a.sessions.get_title_source(sid) is TitleSource.USER
    # CONTRACT: an automatic write may never claim user provenance (the enum makes arbitrary strings unrepresentable; this is the surviving invalid-source arm).
    with pytest.raises(ValueError):
        store_a.sessions.set_auto_title(sid, "bad", source=TitleSource.USER)


def test_title_uniqueness_conflict_leaves_loser_untitled(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("sess-ti4a"))
    store_a.sessions.create(_new("sess-ti4b"))
    store_a.sessions.set_title(SessionId("sess-ti4a"), "taken")
    # CONTRACT: an exact duplicate title on another session is refused and the loser stays untitled (the contract surfaces this as Conflict).
    with pytest.raises(TitleConflict) as conflict:
        store_a.sessions.set_title(SessionId("sess-ti4b"), "taken")
    assert isinstance(conflict.value, Conflict) and isinstance(conflict.value, ValueError)
    assert (conflict.value.title, conflict.value.holder) == ("taken", "sess-ti4a")
    assert store_a.sessions.get_title(SessionId("sess-ti4b")) is None


def test_title_lineage_lookups(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("sess-ti5a"))
    store_a.sessions.create(_new("sess-ti5b"))
    store_a.sessions.set_title(SessionId("sess-ti5a"), "proj")
    store_a.sessions.set_title(SessionId("sess-ti5b"), "proj #3")
    # CONTRACT: exact-title lookup returns the row; the next lineage title increments past the highest number; the unnumbered original counts as #1.
    found = store_a.sessions.get_by_title("proj")
    assert found.session_id == "sess-ti5a"
    assert store_a.sessions.next_title_in_lineage("proj") == "proj #4"
    assert store_a.sessions.next_title_in_lineage("proj #3") == "proj #4"
    assert store_a.sessions.next_title_in_lineage("unused base") == "unused base"
    # CONTRACT: title resolution prefers the latest numbered continuation over the exact match.
    assert store_a.sessions.resolve_by_title("proj") == "sess-ti5b"
    # CONTRACT: missing titles resolve to None.
    assert store_a.sessions.get_by_title("nope") is None
    assert store_a.sessions.resolve_by_title("nope") is None


def test_set_title_source_rewrites_provenance_only(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("sess-ti6"))
    store_a.sessions.set_auto_title(SessionId("sess-ti6"), "auto", source=TitleSource.DERIVED)
    # CONTRACT: provenance can be rewritten without touching the text; untitled rows refuse.
    assert store_a.sessions.set_title_source(SessionId("sess-ti6"), TitleSource.USER) is True
    assert store_a.sessions.get_title(SessionId("sess-ti6")) == "auto"
    assert store_a.sessions.get_title_source(SessionId("sess-ti6")) is TitleSource.USER
    store_a.sessions.create(_new("sess-ti6b"))
    assert store_a.sessions.set_title_source(SessionId("sess-ti6b"), TitleSource.USER) is False


# ── Metadata ─────────────────────────────────────────────────────────────────


def test_model_config_patch_merge_and_delete(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("sess-mc1", model_config={"a": 1}))
    sid = SessionId("sess-mc1")
    store_a.sessions.patch_model_config(sid, {"b": 2})
    # CONTRACT: patches merge key-wise into the stored config.
    assert store_a.sessions.get_model_config_value(sid, "a") == 1
    assert store_a.sessions.get_model_config_value(sid, "b") == 2
    # CONTRACT: a None value deletes the key; the default surfaces for missing keys.
    store_a.sessions.patch_model_config(sid, {"a": None})
    assert store_a.sessions.get_model_config_value(sid, "a", "gone") == "gone"
    # CONTRACT: patching a missing session neither raises nor creates a row.
    missing = SessionId("sess-mc1-missing")
    store_a.sessions.patch_model_config(missing, {"x": 1})
    assert store_a.sessions.get(missing) is None
    assert store_a.sessions.get_model_config_value(missing, "x") is None


def test_replace_session_config_and_update_model(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("sess-mc2", model="m1"))
    sid = SessionId("sess-mc2")
    store_a.sessions.replace_session_config(sid, ReplaceSessionConfig({"k": "v"}))
    row = store_a.sessions.get(sid)
    # CONTRACT: replace swaps model_config wholesale and keeps the model when none is given.
    assert store_a.sessions.get_model_config_value(sid, "k") == "v"
    assert row.model == "m1"
    store_a.sessions.update_model(sid, "m2", provider="prov1")
    row = store_a.sessions.get(sid)
    # CONTRACT: a model switch overwrites the model unconditionally and merges model and provider into model_config so resume recombines the route.
    assert row.model == "m2"
    assert store_a.sessions.get_model_config_value(sid, "model") == "m2"
    assert store_a.sessions.get_model_config_value(sid, "provider") == "prov1"


def test_update_model_writes_the_whole_route_and_clears_the_stale_endpoint(backend, scope_a):
    store_a = backend.open(scope_a)
    from persistence.gateway.routing import GatewayRuntime, gateway_runtime
    from persistence.conversation.sessions import RUNTIME_LOCK_KEY

    store_a.sessions.create(_new("sess-route", model="m1"))
    sid = SessionId("sess-route")
    store_a.sessions.patch_model_config(sid, {RUNTIME_LOCK_KEY: {"provider": "p", "model": "m1"}})
    store_a.sessions.update_model(
        sid, "m2", provider="custom", base_url="https://model.example/v1", api_mode="chat_completions")
    config = (store_a.sessions.get(sid)).model_config
    # CONTRACT: a switch with a provider stores the whole route top-level and as the gateway_runtime object, so resume on either reader recombines model, provider and endpoint.
    route = {"provider": "custom", "base_url": "https://model.example/v1", "api_mode": "chat_completions"}
    assert {key: config[key] for key in route} == route
    assert config["gateway_runtime"] == route
    assert gateway_runtime(config) == GatewayRuntime(**route)
    # CONTRACT: a model switch drops the runtime lock, whose route would otherwise override the switch.
    assert RUNTIME_LOCK_KEY not in config

    store_a.sessions.update_model(sid, "m3", provider="openrouter")
    config = (store_a.sessions.get(sid)).model_config
    # CONTRACT: switching to a provider without an endpoint deletes the previous one at top level and nulls it in gateway_runtime; it never survives to send the new model to the old host.
    assert "base_url" not in config and "api_mode" not in config
    assert config["gateway_runtime"] == {"provider": "openrouter", "base_url": None, "api_mode": None}
    assert gateway_runtime(config) == GatewayRuntime("openrouter")

    store_a.sessions.update_model(sid, "m4", base_url="https://ignored.example")
    config = (store_a.sessions.get(sid)).model_config
    # CONTRACT: without a provider the stored route is left untouched.
    assert config["model"] == "m4"
    assert config["gateway_runtime"] == {"provider": "openrouter", "base_url": None, "api_mode": None}
    assert "base_url" not in config

    store_a.sessions.update_model(sid, "")
    row = store_a.sessions.get(sid)
    # CONTRACT: the model column is overwritten even with an empty model, but an empty model never replaces the model recorded in model_config.
    assert row.model == ""
    assert row.model_config["model"] == "m4"


def test_update_cwd_and_git_generation(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("sess-cwd"))
    sid = SessionId("sess-cwd")
    gen1 = store_a.sessions.update_cwd(sid, "/w", git_branch="main", git_repo_root="/w")
    # CONTRACT: each cwd update returns a monotonically increasing generation.
    assert isinstance(gen1, int) and gen1 >= 1
    row = store_a.sessions.get(sid)
    assert (row.cwd, row.git_branch, row.git_repo_root) == ("/w", "main", "/w")
    # CONTRACT: a same-cwd update with empty git values never clobbers recorded git identity, and it claims a strictly newer generation.
    gen2 = store_a.sessions.update_cwd(sid, "/w")
    assert gen2 > gen1
    row = store_a.sessions.get(sid)
    assert (row.git_branch, row.git_repo_root) == ("main", "/w")
    # CONTRACT: a cwd CHANGE without git values clears the stale git identity.
    store_a.sessions.update_cwd(sid, "/elsewhere")
    row = store_a.sessions.get(sid)
    assert row.cwd == "/elsewhere"
    assert row.git_branch is None
    assert row.git_repo_root is None
    # CONTRACT: updating a missing session returns None and creates nothing.
    missing = SessionId("sess-cwd-missing")
    assert store_a.sessions.update_cwd(missing, "/w") is None
    assert store_a.sessions.get(missing) is None


def test_update_cwd_replace_git_meta_is_the_workspace_move(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("sess-move"))
    sid = SessionId("sess-move")
    gen1 = store_a.sessions.update_cwd(sid, "/w", git_branch="main", git_repo_root="/w")
    # CONTRACT: a same-cwd MOVE with empty git values clears the recorded git identity and still claims a newer generation.
    gen2 = store_a.sessions.update_cwd(sid, "/w", replace_git_meta=True)
    assert gen2 is not None and gen1 is not None and gen2 > gen1
    row = store_a.sessions.get(sid)
    assert (row.cwd, row.git_branch, row.git_repo_root) == ("/w", None, None)
    # CONTRACT: a same-cwd MOVE overwrites both fields with what it is given; the field it is not given is cleared, never kept.
    store_a.sessions.update_cwd(sid, "/w", git_branch="main", git_repo_root="/w")
    store_a.sessions.update_cwd(sid, "/w", git_branch="dev", replace_git_meta=True)
    row = store_a.sessions.get(sid)
    assert (row.git_branch, row.git_repo_root) == ("dev", None)
    # CONTRACT: a MOVE to another cwd behaves as a plain cwd change: given values land, and with none the identity is cleared.
    store_a.sessions.update_cwd(sid, "/v", git_branch="b", git_repo_root="/v", replace_git_meta=True)
    row = store_a.sessions.get(sid)
    assert (row.cwd, row.git_branch, row.git_repo_root) == ("/v", "b", "/v")
    store_a.sessions.update_cwd(sid, "/u", replace_git_meta=True)
    row = store_a.sessions.get(sid)
    assert (row.cwd, row.git_branch, row.git_repo_root) == ("/u", None, None)
    # CONTRACT: a MOVE of an unknown id returns None and creates nothing.
    missing = SessionId("sess-move-missing")
    assert store_a.sessions.update_cwd(missing, "/w", replace_git_meta=True) is None
    assert store_a.sessions.get(missing) is None


def test_publish_git_metadata_generation_fencing(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("sess-pub"))
    sid = SessionId("sess-pub")
    gen1 = store_a.sessions.update_cwd(sid, "/w")
    gen2 = store_a.sessions.update_cwd(sid, "/w")
    # CONTRACT: publishing against a superseded generation is refused; the current generation lands.
    assert store_a.sessions.publish_git_metadata(sid, "/w", gen1, git_branch="stale") is False
    row = store_a.sessions.get(sid)
    assert row.git_branch is None
    assert store_a.sessions.publish_git_metadata(sid, "/w", gen2, git_branch="fresh") is True
    row = store_a.sessions.get(sid)
    assert row.git_branch == "fresh"
    # CONTRACT: publishes with no non-empty git fields or an invalid generation are refused, leaving the published state intact.
    assert store_a.sessions.publish_git_metadata(sid, "/w", gen2) is False
    assert store_a.sessions.publish_git_metadata(sid, "/w", 0, git_branch="b") is False
    row = store_a.sessions.get(sid)
    assert row.git_branch == "fresh"


# ── Lineage ──────────────────────────────────────────────────────────────────


def test_child_inherits_parent_metadata_nulls_only(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("sess-par1", cwd="/repo", git_repo_root="/repo"))
    store_a.sessions.create(
        _new("sess-kid1", parent_session_id=SessionId("sess-par1"))
    )
    row = store_a.sessions.get(SessionId("sess-kid1"))
    # CONTRACT: a child's NULL cwd/git backfill from the parent at creation.
    assert row.cwd == "/repo"
    assert row.git_repo_root == "/repo"
    store_a.sessions.create(
        _new("sess-kid2", parent_session_id=SessionId("sess-par1"), cwd="/own")
    )
    row = store_a.sessions.get(SessionId("sess-kid2"))
    # CONTRACT: an explicit child value is never overwritten by inheritance.
    assert row.cwd == "/own"


def test_compression_fork_inherits_routing_columns(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.create(
        _new(
            "sess-par3", source="telegram", session_key="agent:main:r", chat_id="c1",
            chat_type="private", thread_id="t1", user_id="u1",
        ),
    )
    store_a.sessions.end(SessionId("sess-par3"), "compression")
    store_a.sessions.create(
        _new("sess-fork3", source="telegram", parent_session_id=SessionId("sess-par3"))
    )
    row = store_a.sessions.get(SessionId("sess-fork3"))
    # CONTRACT: a compression fork inherits the gateway routing columns so a crash cannot strand it unroutable.
    assert (row.session_key, row.chat_id, row.chat_type, row.thread_id, row.user_id) == (
        "agent:main:r", "c1", "private", "t1", "u1",
    )
    # CONTRACT: a non-compression child does NOT inherit routing columns.
    store_a.sessions.create(
        _new("sess-par4", source="telegram", session_key="agent:main:s", chat_id="c2")
    )
    store_a.sessions.create(
        _new("sess-kid5", parent_session_id=SessionId("sess-par4"))
    )
    row = store_a.sessions.get(SessionId("sess-kid5"))
    assert row.session_key is None
    assert row.chat_id is None


# ── Archive, pin, hide, read ─────────────────────────────────────────────────


def _make_compression_pair(backend, scope, root_id: str, tip_id: str, source: str = "cli"):
    store = backend.open(scope)
    store.sessions.create(_new(root_id, source=source))
    store.sessions.end(SessionId(root_id), "compression")
    store.sessions.create(
        _new(tip_id, source=source, parent_session_id=SessionId(root_id))
    )


def test_archive_propagates_across_compression_lineage(backend, scope_a):
    store_a = backend.open(scope_a)
    _make_compression_pair(backend, scope_a, "sess-ar-root", "sess-ar-tip")
    # CONTRACT: archiving either end of a compression lineage flips the whole lineage; unarchive mirrors it.
    change = store_a.sessions.set_archived(SessionId("sess-ar-tip"), True)
    # CONTRACT: a real flip to archived names every archived row of the lineage for the caller's archive notification.
    assert change.changed is True and set(change.archived_ids) == {"sess-ar-root", "sess-ar-tip"}
    assert (store_a.sessions.get(SessionId("sess-ar-root"))).archived is True
    assert (store_a.sessions.get(SessionId("sess-ar-tip"))).archived is True
    assert store_a.sessions.set_archived(SessionId("sess-ar-root"), False) == ArchiveChange(changed=True)
    assert (store_a.sessions.get(SessionId("sess-ar-root"))).archived is False
    assert (store_a.sessions.get(SessionId("sess-ar-tip"))).archived is False
    # CONTRACT: archiving a missing id reports False.
    assert store_a.sessions.set_archived(SessionId("sess-ar-missing"), True) == ArchiveChange(changed=False)


def test_archive_does_not_cross_non_compression_edges(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("sess-ar2-par"))
    store_a.sessions.create(
        _new("sess-ar2-kid", parent_session_id=SessionId("sess-ar2-par"))
    )
    store_a.sessions.set_archived(SessionId("sess-ar2-par"), True)
    # CONTRACT: lineage propagation follows compression edges only; an ordinary child is untouched.
    assert (store_a.sessions.get(SessionId("sess-ar2-par"))).archived is True
    assert (store_a.sessions.get(SessionId("sess-ar2-kid"))).archived is False


def test_pin_clears_hidden_and_propagates(backend, scope_a):
    store_a = backend.open(scope_a)
    _make_compression_pair(backend, scope_a, "sess-pin-root", "sess-pin-tip")
    store_a.sessions.set_hidden(SessionId("sess-pin-root"), True)
    assert (store_a.sessions.get(SessionId("sess-pin-tip"))).hidden is True
    # CONTRACT: pinning propagates across the lineage and clears hidden so the pinned row is visible.
    assert store_a.sessions.set_pinned(SessionId("sess-pin-root"), True) is True
    for sid in ("sess-pin-root", "sess-pin-tip"):
        row = store_a.sessions.get(SessionId(sid))
        assert row.pinned is True
        assert row.hidden is False
    # CONTRACT: unpinning does not re-hide.
    store_a.sessions.set_pinned(SessionId("sess-pin-root"), False)
    row = store_a.sessions.get(SessionId("sess-pin-root"))
    assert row.pinned is False
    assert row.hidden is False


def test_read_watermark_via_listing(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("sess-read"))

    def unread_flag() -> bool:
        listed = store_a.sessions.list_sessions(SessionQuery(), SessionWindow(limit=100))
        return {e.session.session_id: e.unread for e in listed}["sess-read"]

    # CONTRACT: a never-tracked watermark reads as read.
    assert unread_flag() is False
    # CONTRACT: an explicit unread mark surfaces as unread in listings.
    assert store_a.sessions.set_read(SessionId("sess-read"), False) is True
    assert unread_flag() is True
    # CONTRACT: re-marking read clears the unread state.
    assert store_a.sessions.set_read(SessionId("sess-read"), True) is True
    assert unread_flag() is False


def test_unarchive_recoverable_session(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("sess-rec1"))
    store_a.sessions.end(SessionId("sess-rec1"), "agent_close")
    store_a.sessions.set_archived(SessionId("sess-rec1"), True)
    # CONTRACT: a row archived after a recoverable accidental end is resurrected: unarchived, end stamp cleared.
    assert store_a.sessions.unarchive_recoverable(SessionId("sess-rec1")) is True
    row = store_a.sessions.get(SessionId("sess-rec1"))
    assert row.archived is False
    assert row.ended_at is None
    assert row.end_reason is None
    # CONTRACT: a deliberately ended archived row is left alone: still archived, end stamp and reason intact.
    store_a.sessions.create(_new("sess-rec2"))
    store_a.sessions.end(SessionId("sess-rec2"), "session_reset")
    store_a.sessions.set_archived(SessionId("sess-rec2"), True)
    before = store_a.sessions.get(SessionId("sess-rec2"))
    assert store_a.sessions.unarchive_recoverable(SessionId("sess-rec2")) is False
    after = store_a.sessions.get(SessionId("sess-rec2"))
    assert after.archived is True
    assert after.end_reason == "session_reset"
    assert after.ended_at == before.ended_at
    # CONTRACT: a non-archived row and a missing id both report False.
    store_a.sessions.create(_new("sess-rec3"))
    assert store_a.sessions.unarchive_recoverable(SessionId("sess-rec3")) is False
    assert store_a.sessions.unarchive_recoverable(SessionId("sess-rec-missing")) is False


def _canonical_bot_chat(store, session_id: str, *, hidden: bool = True) -> SessionId:
    sid = SessionId(session_id)
    store.sessions.create(_new(session_id, source="desktop"))
    assert store.sessions.set_title(sid, CANONICAL_BOT_CHAT_TITLE)
    store.sessions.set_hidden(sid, hidden)
    return sid


@pytest.mark.parametrize("hidden", [True, False])
def test_archive_retires_the_canonical_bot_chat_name(backend, scope_a, hidden):
    store_a = backend.open(scope_a)
    sid = _canonical_bot_chat(store_a, "sess-bc-retire", hidden=hidden)
    # CONTRACT: a deliberate archive releases the canonical title and un-hides the row in the same write.
    assert store_a.sessions.set_archived(sid, True).changed is True
    row = store_a.sessions.get(sid)
    assert (row.archived, row.title, row.title_source, row.hidden) == (True, None, None, False)
    assert store_a.sessions.get_by_title(CANONICAL_BOT_CHAT_TITLE) is None
    assert store_a.sessions.resolve_by_title(CANONICAL_BOT_CHAT_TITLE) is None
    # CONTRACT: the name is free for a fresh canonical chat; unarchiving the old row never takes it back.
    fresh = _canonical_bot_chat(store_a, "sess-bc-fresh")
    assert store_a.sessions.get_by_title(CANONICAL_BOT_CHAT_TITLE).session_id == fresh
    assert store_a.sessions.set_archived(sid, False).changed is True
    assert (store_a.sessions.get(sid).title, store_a.sessions.get(sid).hidden) == (None, False)


@pytest.mark.parametrize("already_archived", [False, True])
def test_archive_of_a_reaped_bot_chat_clears_its_recoverable_end(backend, scope_a, already_archived):
    store_a = backend.open(scope_a)
    sid = _canonical_bot_chat(store_a, "sess-bc-reaped")
    store_a.sessions.end(sid, "ws_orphan_reap")
    ended_at = store_a.sessions.get(sid).ended_at
    if already_archived:
        store_a.maintenance.auto_archive_lineage(sid)
        # CONTRACT: the idle sweep's archive keeps the name: the row still resolves for resurrection.
        assert store_a.sessions.get_by_title(CANONICAL_BOT_CHAT_TITLE).session_id == sid
    # CONTRACT: the user's archive retires a reaped row too: the recoverable end reason is cleared, ended_at kept.
    assert store_a.sessions.set_archived(sid, True).changed is True
    row = store_a.sessions.get(sid)
    assert (row.archived, row.title, row.end_reason, row.ended_at) == (True, None, None, ended_at)
    assert store_a.sessions.get_by_title(CANONICAL_BOT_CHAT_TITLE) is None
    assert store_a.sessions.unarchive_recoverable(sid) is False


def test_archived_titles_other_than_bot_chat_still_resolve(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("sess-bc-plain"))
    store_a.sessions.set_title(SessionId("sess-bc-plain"), "My project")
    store_a.sessions.set_archived(SessionId("sess-bc-plain"), True)
    # CONTRACT: only the canonical title retires on archive; any other archived title keeps resolving.
    assert store_a.sessions.get_by_title("My project").session_id == "sess-bc-plain"
    assert store_a.sessions.resolve_by_title("My project") == "sess-bc-plain"
    # CONTRACT: a released canonical title never falls through to a numbered sibling.
    sid = _canonical_bot_chat(store_a, "sess-bc-canon")
    store_a.sessions.create(_new("sess-bc-sibling", source="desktop"))
    store_a.sessions.set_title(SessionId("sess-bc-sibling"), f"{CANONICAL_BOT_CHAT_TITLE} #2")
    store_a.sessions.set_archived(sid, True)
    assert store_a.sessions.resolve_by_title(CANONICAL_BOT_CHAT_TITLE) is None


# ── Delete, including parent/child propagation ───────────────────────────────


def test_delete_cascades_delegate_children_only(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("sess-del-par"))
    store_a.sessions.create(
        _new(
            "sess-del-sub",
            parent_session_id=SessionId("sess-del-par"),
            lineage_kind=SessionLineageKind.DELEGATE,
        ),
    )
    store_a.sessions.create(
        _new(
            "sess-del-branch",
            parent_session_id=SessionId("sess-del-par"),
            lineage_kind=SessionLineageKind.BRANCH,
        ),
    )
    # CONTRACT: the preview lists exactly the parent plus its recursive delegate children.
    targets = store_a.sessions.delete_targets(SessionId("sess-del-par"))
    assert targets == ("sess-del-par", "sess-del-sub")
    assert store_a.sessions.delete(SessionId("sess-del-par"), media_root=MEDIA_ROOT) == ("sess-del-par", "sess-del-sub")
    # CONTRACT: delegate children cascade with the parent.
    assert store_a.sessions.get(SessionId("sess-del-par")) is None
    assert store_a.sessions.get(SessionId("sess-del-sub")) is None
    # CONTRACT: branch children survive, orphaned.
    branch = store_a.sessions.get(SessionId("sess-del-branch"))
    assert branch is not None
    assert branch.parent_session_id is None


def test_delete_cascades_recursive_delegate_chain(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("sess-del2"))
    store_a.sessions.create(
        _new(
            "sess-del2-a",
            parent_session_id=SessionId("sess-del2"),
            lineage_kind=SessionLineageKind.DELEGATE,
        ),
    )
    store_a.sessions.create(
        _new(
            "sess-del2-b",
            parent_session_id=SessionId("sess-del2-a"),
            lineage_kind=SessionLineageKind.DELEGATE,
        ),
    )
    # CONTRACT: deleting the root returns every removed id; the delegate cascade is recursive through grandchildren.
    assert store_a.sessions.delete(SessionId("sess-del2"), media_root=MEDIA_ROOT) == ("sess-del2", "sess-del2-a", "sess-del2-b")
    for sid in ("sess-del2", "sess-del2-a", "sess-del2-b"):
        assert store_a.sessions.get(SessionId(sid)) is None


def test_delete_with_stale_expected_ids_fails_closed(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("sess-del3"))
    store_a.sessions.create(
        _new(
            "sess-del3-sub",
            parent_session_id=SessionId("sess-del3"),
            lineage_kind=SessionLineageKind.DELEGATE,
        ),
    )
    # CONTRACT: when the expected id set no longer matches the live cascade the delete refuses and removes nothing.
    assert store_a.sessions.delete(
        SessionId("sess-del3"), expected_ids=[SessionId("sess-del3")], media_root=MEDIA_ROOT
    ) == ()
    assert store_a.sessions.get(SessionId("sess-del3")) is not None
    assert store_a.sessions.get(SessionId("sess-del3-sub")) is not None
    # CONTRACT: a matching expected set proceeds.
    assert store_a.sessions.delete(
        SessionId("sess-del3"),
        expected_ids=[SessionId("sess-del3"), SessionId("sess-del3-sub")], media_root=MEDIA_ROOT,
    ) == ("sess-del3", "sess-del3-sub")
    assert store_a.sessions.get(SessionId("sess-del3")) is None


def test_delete_missing_removes_nothing_and_targets_empty(backend, scope_a):
    # CONTRACT: deleting a missing id removes nothing; its target preview is empty.
    store_a = backend.open(scope_a)
    assert store_a.sessions.delete(SessionId("sess-del-missing"), media_root=MEDIA_ROOT) == ()
    assert store_a.sessions.delete_targets(SessionId("sess-del-missing")) == ()


def test_delete_many_skips_unknown_ids(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("sess-blk1"))
    store_a.sessions.create(_new("sess-blk2"))
    # CONTRACT: bulk delete removes known rows, skips unknown ids, and reports the requested ids that existed and every row removed.
    outcome = store_a.sessions.delete_many(
        [SessionId("sess-blk1"), SessionId("sess-blk2"), SessionId("sess-blk-missing")], media_root=MEDIA_ROOT
    )
    assert (outcome.deleted, outcome.skipped_guarded, outcome.removed_ids) == (2, (), ("sess-blk1", "sess-blk2"))
    assert store_a.sessions.get(SessionId("sess-blk1")) is None
    assert store_a.sessions.get(SessionId("sess-blk2")) is None
    assert store_a.sessions.delete_many([], media_root=MEDIA_ROOT).deleted == 0
    assert store_a.sessions.delete_many([SessionId("sess-blk-missing")], media_root=MEDIA_ROOT).removed_ids == ()


def test_delete_if_empty(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("sess-emp1"))
    # CONTRACT: an empty untitled childless session deletes.
    assert store_a.sessions.delete_if_empty(SessionId("sess-emp1"), media_root=MEDIA_ROOT) == ("sess-emp1",)
    assert store_a.sessions.get(SessionId("sess-emp1")) is None
    # CONTRACT: a title blocks the empty-delete; the refused row and its title survive intact.
    store_a.sessions.create(_new("sess-emp3"))
    store_a.sessions.set_title(SessionId("sess-emp3"), "named")
    assert store_a.sessions.delete_if_empty(SessionId("sess-emp3"), media_root=MEDIA_ROOT) == ()
    assert store_a.sessions.get(SessionId("sess-emp3")) is not None
    assert store_a.sessions.get_title(SessionId("sess-emp3")) == "named"
    # CONTRACT: a child blocks the empty-delete; parent and child both survive.
    store_a.sessions.create(_new("sess-emp4"))
    store_a.sessions.create(
        _new("sess-emp4-kid", parent_session_id=SessionId("sess-emp4"))
    )
    assert store_a.sessions.delete_if_empty(SessionId("sess-emp4"), media_root=MEDIA_ROOT) == ()
    assert store_a.sessions.get(SessionId("sess-emp4")) is not None
    assert store_a.sessions.get(SessionId("sess-emp4-kid")) is not None
    # This test does not cover message-blocked deletion; persistence.conversation.messages.MessageRepository.append can seed that case.


def test_count_and_delete_empty_sessions(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("sess-swp1"))
    store_a.sessions.end(SessionId("sess-swp1"), "cli_close")
    store_a.sessions.create(_new("sess-swp3"))  # live and empty: not swept
    store_a.sessions.create(_new("sess-swp4"))  # ended but archived: not swept
    store_a.sessions.end(SessionId("sess-swp4"), "cli_close")
    store_a.sessions.set_archived(SessionId("sess-swp4"), True)
    # CONTRACT: only ended, empty, non-archived sessions count as empty; the sweep deletes exactly those.
    assert store_a.sessions.count_empty() == 1
    assert store_a.sessions.delete_empty(media_root=MEDIA_ROOT) == ("sess-swp1",)
    assert store_a.sessions.get(SessionId("sess-swp1")) is None
    assert store_a.sessions.get(SessionId("sess-swp3")) is not None
    assert store_a.sessions.get(SessionId("sess-swp4")) is not None


# ── Listing ──────────────────────────────────────────────────────────────────


def test_pinned_backfill_carries_archived_pins(backend, scope_a):
    """``include_pinned`` back-fills pins whatever their archive state: an archived pinned session rides on the listing though the query admits no archived rows, an archived unpinned one does not."""
    store_a = backend.open(scope_a)
    for sid in ("sess-bf-live", "sess-bf-pin", "sess-bf-old"):
        store_a.sessions.create(_new(sid))
    store_a.sessions.set_pinned(SessionId("sess-bf-pin"), True)
    store_a.sessions.set_archived(SessionId("sess-bf-pin"), True)
    store_a.sessions.set_archived(SessionId("sess-bf-old"), True)
    query = SessionQuery(include_pinned=True)
    expected = {"sess-bf-live", "sess-bf-pin"}
    assert _ids(backend, scope_a, query) == expected


VISIBLE_DEFAULT = {"sess-li-cli1", "sess-li-cli2", "sess-li-tg", "sess-li-ended"}


def _seed_listing(backend, scope) -> None:
    store = backend.open(scope)
    store.sessions.create(_new("sess-li-cli1"))
    store.sessions.create(_new("sess-li-cli2"))
    store.sessions.create(_new("sess-li-tg", source="telegram"))
    store.sessions.create(_new("sess-li-ended", cwd="/proj/a"))
    store.sessions.end(SessionId("sess-li-ended"), "cli_close")
    store.sessions.create(_new("sess-li-arch"))
    store.sessions.set_archived(SessionId("sess-li-arch"), True)
    store.sessions.create(_new("sess-li-hid"))
    store.sessions.set_hidden(SessionId("sess-li-hid"), True)


def test_list_default_excludes_archived_and_hidden(backend, scope_a):
    _seed_listing(backend, scope_a)
    # CONTRACT: the default listing hides archived and hidden rows only; ENDED visible sessions remain listed.
    assert _ids(backend, scope_a) == VISIBLE_DEFAULT


def test_list_archived_and_hidden_switches(backend, scope_a):
    _seed_listing(backend, scope_a)
    # CONTRACT: include_archived adds archived rows; archived_only restricts to them; include_hidden adds hidden rows.
    assert "sess-li-arch" in _ids(backend, scope_a, SessionQuery(include_archived=True))
    assert _ids(backend, scope_a, SessionQuery(archived_only=True)) == {"sess-li-arch"}
    assert "sess-li-hid" in _ids(backend, scope_a, SessionQuery(include_hidden=True))


def test_list_source_filters(backend, scope_a):
    _seed_listing(backend, scope_a)
    # CONTRACT: sources narrows to the given set; exclude_sources removes.
    assert _ids(backend, scope_a, SessionQuery(sources=("telegram",))) == {"sess-li-tg"}
    assert _ids(backend, scope_a, SessionQuery(sources=("cli", "telegram"))) == VISIBLE_DEFAULT
    assert _ids(backend, scope_a, SessionQuery(exclude_sources=("cli",))) == {"sess-li-tg"}
    # CONTRACT: source composes with the archive switches; both must hold at once.
    assert _ids(backend, scope_a, SessionQuery(sources=("cli",), archived_only=True)) == {"sess-li-arch"}
    assert _ids(backend, scope_a, SessionQuery(sources=("telegram",), archived_only=True)) == set()


def test_list_cwd_prefix_filter(backend, scope_a):
    _seed_listing(backend, scope_a)
    # CONTRACT: cwd_prefix admits rows whose cwd sits at or under the prefix, excluding unrelated cwds and rows without one.
    assert _ids(backend, scope_a, SessionQuery(cwd_prefix="/proj")) == {"sess-li-ended"}
    assert _ids(backend, scope_a, SessionQuery(cwd_prefix="/nowhere")) == set()


def test_list_session_key_filter(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("sess-sk1", source="telegram", session_key="agent:main:k1"))
    store_a.sessions.create(_new("sess-sk2", source="telegram", session_key="agent:main:k2"))
    # CONTRACT: session_key narrows to exact-key rows.
    assert _ids(backend, scope_a, SessionQuery(session_key="agent:main:k1")) == {"sess-sk1"}


def _windows(backend, scope, query: SessionQuery, limit: int, rows: int) -> list[list]:
    """The ``limit``-row windows covering ``rows`` listed rows, by increasing offset."""
    store = backend.open(scope)
    return [
        [entry.session.session_id for entry in store.sessions.list_sessions(query, SessionWindow(limit=limit, offset=offset))]
        for offset in range(0, rows, limit)
    ]


def test_list_windows_lose_and_duplicate_nothing(backend, scope_a):
    _seed_listing(backend, scope_a)
    collected = [sid for window in _windows(backend, scope_a, SessionQuery(), 2, len(_ids(backend, scope_a))) for sid in window]
    # CONTRACT: offset windows over an unchanged dataset lose nothing and duplicate nothing.
    assert len(collected) == len(set(collected))
    assert set(collected) == _ids(backend, scope_a)


def test_list_include_pinned_backfills_beyond_the_window(backend, scope_a):
    store_a = backend.open(scope_a)
    _seed_listing(backend, scope_a)
    store_a.sessions.set_pinned(SessionId("sess-li-cli1"), True)
    query = SessionQuery(include_pinned=True)
    rows = len(_ids(backend, scope_a))
    windows = _windows(backend, scope_a, query, 1, rows)
    # CONTRACT: every window carries the pinned row: in place in its own slice, back-filled after the slice elsewhere.
    assert all("sess-li-cli1" in window for window in windows)
    # CONTRACT: the slices themselves still lose and duplicate nothing; only the pin repeats.
    slices = [window[0] for window in windows]
    assert len(slices) == len(set(slices)) and set(slices) == _ids(backend, scope_a)
    assert all(window == [window[0]] or window == [window[0], "sess-li-cli1"] for window in windows)
    # CONTRACT: a window past the end still carries the pin.
    assert [e.session.session_id for e in store_a.sessions.list_sessions(query, SessionWindow(limit=1, offset=rows))] == ["sess-li-cli1"]


def test_count_at_least_counts_every_row(backend, scope_a):
    store_a = backend.open(scope_a)
    _seed_listing(backend, scope_a)
    total = store_a.sessions.count_sessions(SessionCountQuery(include_archived=True))
    # CONTRACT: the short-circuit existence probe takes no filters (archived and hidden rows count) and agrees with the totals in both polarities.
    assert total == 6
    assert store_a.sessions.count_at_least() is True
    assert store_a.sessions.count_at_least(total) is True
    assert store_a.sessions.count_at_least(total + 1) is False


def test_count_by_source_splits_by_archive_state(backend, scope_a):
    store_a = backend.open(scope_a)
    _seed_listing(backend, scope_a)
    # CONTRACT: per-source counts include hidden rows, exclude archived rows by default and split by archive state on request; largest first.
    default = store_a.sessions.count_by_source()
    assert dict(default) == {"cli": 4, "telegram": 1} and list(default) == ["cli", "telegram"]
    assert store_a.sessions.count_by_source(include_archived=True)["cli"] == 5
    archived_only = store_a.sessions.count_by_source(archived_only=True)
    assert archived_only["cli"] == 1
    assert "telegram" not in archived_only


def test_session_counts_match_listing(backend, scope_a):
    store_a = backend.open(scope_a)
    _seed_listing(backend, scope_a)
    # CONTRACT: counts include hidden rows, composing with every other listing switch.
    for count_query, list_query in (
        (SessionCountQuery(), SessionQuery(include_hidden=True)),
        (SessionCountQuery(sources=("telegram",)), SessionQuery(sources=("telegram",), include_hidden=True)),
        (SessionCountQuery(include_archived=True), SessionQuery(include_archived=True, include_hidden=True)),
        (SessionCountQuery(archived_only=True), SessionQuery(archived_only=True, include_hidden=True)),
    ):
        assert store_a.sessions.count_sessions(count_query) == len(_ids(backend, scope_a, list_query))


def test_list_title_and_id_queries(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("sess-q-alpha"))
    store_a.sessions.set_title(SessionId("sess-q-alpha"), "alpha project")
    store_a.sessions.create(_new("sess-q-beta"))
    store_a.sessions.set_title(SessionId("sess-q-beta"), "beta work")
    recent = SessionOrder.LAST_ACTIVE
    # CONTRACT: search_text matches titles and excludes non-matches.
    assert _ids(backend, scope_a, SessionQuery(search_text="alpha", order=recent)) == {"sess-q-alpha"}
    assert _ids(backend, scope_a, SessionQuery(search_text="no-such", order=recent)) == set()
    # CONTRACT: id_query matches session-id substrings, excluding non-matches.
    assert _ids(backend, scope_a, SessionQuery(id_query="q-beta", order=recent)) == {"sess-q-beta"}
    # CONTRACT: both filters apply in last-active order only.
    assert _ids(backend, scope_a, SessionQuery(search_text="no-such")) == {"sess-q-alpha", "sess-q-beta"}


def _search_ids(backend, scope, **filters) -> set:
    store = backend.open(scope)
    return {entry.session.session_id for entry in store.sessions.search_sessions(window=SessionWindow(limit=100), **filters)}


def test_search_sessions_workspace_key_admission(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("sess-ws-a", git_repo_root="/repo"))
    store_a.sessions.create(_new("sess-ws-b", cwd="/repo/sub"))
    store_a.sessions.create(_new("sess-ws-c", cwd="/other"))
    store_a.sessions.create(_new("sess-ws-tg", source="telegram", cwd="/repo/sub"))
    # CONTRACT: workspace_key admits rows whose git root equals the key or, lacking one, whose cwd sits at or under it.
    assert _search_ids(backend, scope_a, workspace_key="/repo") == {"sess-ws-a", "sess-ws-b", "sess-ws-tg"}
    # CONTRACT: the source filter composes with workspace_key, excluding other-source rows in the same workspace.
    assert _search_ids(backend, scope_a, sources=("cli",), workspace_key="/repo") == {"sess-ws-a", "sess-ws-b"}
    assert _search_ids(backend, scope_a, sources=("telegram",)) == {"sess-ws-tg"}


def test_list_children_hidden_unless_included(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("sess-ch-par"))
    store_a.sessions.create(
        _new(
            "sess-ch-sub",
            parent_session_id=SessionId("sess-ch-par"),
            lineage_kind=SessionLineageKind.DELEGATE,
        ),
    )
    store_a.sessions.create(
        _new(
            "sess-ch-branch",
            parent_session_id=SessionId("sess-ch-par"),
            lineage_kind=SessionLineageKind.BRANCH,
        ),
    )
    ids = _ids(backend, scope_a)
    # CONTRACT: the default listing hides delegate children but keeps user-visible branch children.
    assert "sess-ch-par" in ids
    assert "sess-ch-branch" in ids
    assert "sess-ch-sub" not in ids
    # CONTRACT: include_children surfaces everything.
    assert "sess-ch-sub" in _ids(backend, scope_a, SessionQuery(include_children=True))
    # CONTRACT: include_subagents re-admits delegate children alone, and the count agrees with the listing.
    with_subagents = SessionQuery(include_subagents=True)
    sub_ids = _ids(backend, scope_a, with_subagents)
    assert {"sess-ch-par", "sess-ch-branch", "sess-ch-sub"} <= sub_ids
    assert store_a.sessions.count_sessions(SessionCountQuery(include_subagents=True, exclude_children=True)) == len(sub_ids)
    assert store_a.sessions.count_sessions(SessionCountQuery(exclude_children=True)) == len(ids)
    # CONTRACT: a count includes children unless told otherwise.
    assert store_a.sessions.count_sessions(SessionCountQuery()) == 3


def test_list_projects_compression_root_to_tip(backend, scope_a):
    store_a = backend.open(scope_a)
    _make_compression_pair(backend, scope_a, "sess-cp-root", "sess-cp-tip")
    ids = [entry.session.session_id for entry in store_a.sessions.list_sessions(SessionQuery(), SessionWindow(limit=100))]
    # CONTRACT: a compression lineage surfaces as ONE row carrying the live tip's id; the ended root does not appear and the tip appears exactly once.
    assert ids.count("sess-cp-tip") == 1
    assert "sess-cp-root" not in ids
    # CONTRACT: project_compression_tips=False lists the raw root row instead of the projected tip.
    raw = _ids(backend, scope_a, SessionQuery(project_compression_tips=False))
    assert "sess-cp-root" in raw
    assert "sess-cp-tip" not in raw


def test_reset_lineage_child_stays_listed_after_parent_reopen(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("sess-parent-r", source="telegram", session_key="agent:main:k1"))
    store_a.sessions.end(SessionId("sess-parent-r"), "session_reset")
    store_a.sessions.create(
        _new(
            "sess-child-r", source="telegram",
            parent_session_id=SessionId("sess-parent-r"), session_key="agent:main:k1",
        ),
    )
    assert "sess-child-r" in _ids(backend, scope_a)
    store_a.sessions.reopen(SessionId("sess-parent-r"))
    # CONTRACT: a reset-boundary child stays in the default listing after the parent reopens (typed lineage makes SessionDB's marker inference unnecessary; the outcome is preserved).
    assert "sess-child-r" in _ids(backend, scope_a)


# ── Missing-ID returns ───────────────────────────────────────────────────────


def test_missing_id_read_surfaces(backend, scope_a):
    store_a = backend.open(scope_a)
    missing = SessionId("sess-nope")
    # CONTRACT: reads of unknown ids return None or empty values, never raise.
    assert store_a.sessions.get(missing) is None
    assert store_a.sessions.get_title(missing) is None
    assert store_a.sessions.get_title_source(missing) is None
    assert store_a.sessions.get_model_config_value(missing, "k", "dflt") == "dflt"
    assert store_a.sessions.resolve_id("sess-nope") is None


def test_missing_id_write_surfaces_report_failure(backend, scope_a):
    store_a = backend.open(scope_a)
    missing = SessionId("sess-nope")
    # CONTRACT: lifecycle writes against unknown ids report failure without creating rows.
    assert store_a.sessions.set_archived(missing, True).changed is False
    assert store_a.sessions.set_pinned(missing, True) is False
    assert store_a.sessions.set_hidden(missing, True) is False
    assert store_a.sessions.set_read(missing, True) is False
    assert store_a.sessions.set_title(missing, "t") is False
    assert store_a.sessions.set_auto_title(missing, "t", source=TitleSource.LLM) is False
    assert store_a.sessions.get(missing) is None


def test_resolve_session_id_prefixes(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("abc-one"))
    store_a.sessions.create(_new("abd-two"))
    # CONTRACT: exact ids resolve; a unique prefix resolves; an ambiguous prefix returns None.
    assert store_a.sessions.resolve_id("abc-one") == "abc-one"
    assert store_a.sessions.resolve_id("abd") == "abd-two"
    assert store_a.sessions.resolve_id("ab") is None


# ── Two-profile isolation through the public protocol ─────────────────────────


def test_two_scopes_colliding_ids_stay_isolated(backend, scope_a, scope_b):
    store_a = backend.open(scope_a)
    store_b = backend.open(scope_b)
    shared = SessionId("sess-shared")
    store_a.sessions.create(_new("sess-shared", model="sentinel-a"))
    store_b.sessions.create(_new("sess-shared", source="telegram", model="sentinel-b"))
    store_a.sessions.create(_new("sess-only-a"))
    # Positive reads per scope: each sees its own sentinel.
    row_a = store_a.sessions.get(shared)
    row_b = store_b.sessions.get(shared)
    assert row_a.model == "sentinel-a"
    assert row_b.model == "sentinel-b"
    # Cross-scope invisibility: a row seeded only in A is absent from B's reads and listings.
    assert store_b.sessions.get(SessionId("sess-only-a")) is None
    assert "sess-only-a" not in _ids(backend, scope_b)
    # Cross-scope mutation impotence: B's writes to the colliding id never reach A's row.
    store_b.sessions.end(shared, "cli_close")
    store_b.sessions.set_title(shared, "b-title")
    row_a = store_a.sessions.get(shared)
    assert row_a.ended_at is None
    assert row_a.title is None
    # B's delete removes only B's row.
    assert store_b.sessions.delete(shared, media_root=MEDIA_ROOT) == (shared,)
    assert store_a.sessions.get(shared) is not None


def test_two_scopes_title_namespaces_are_independent(backend, scope_a, scope_b):
    store_a = backend.open(scope_a)
    store_b = backend.open(scope_b)
    store_a.sessions.create(_new("sess-ta"))
    store_b.sessions.create(_new("sess-tb"))
    store_a.sessions.set_title(SessionId("sess-ta"), "shared name")
    # Title uniqueness is scoped: the same title lands cleanly in the other scope.
    assert store_b.sessions.set_title(SessionId("sess-tb"), "shared name") is True
    found_b = store_b.sessions.get_by_title("shared name")
    assert found_b.session_id == "sess-tb"
    found_a = store_a.sessions.get_by_title("shared name")
    assert found_a.session_id == "sess-ta"


# ── Retirement ───


def test_create_after_delete_raises_retired_session_error(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("sess-ret1", model="first-life"))
    assert store_a.sessions.delete(SessionId("sess-ret1"), media_root=MEDIA_ROOT) == ("sess-ret1",)
    # CONTRACT: the discriminating case for recreating a deleted id. A deleted id is
    # permanently retired; recreating it is neither absence nor a live-row collision.
    with pytest.raises(RetiredSessionError) as excinfo:
        store_a.sessions.create(_new("sess-ret1", model="second-life"))
    assert excinfo.value.session_id == SessionId("sess-ret1") and excinfo.value.reason == "not_found"
    # CONTRACT: the refused create resurrected nothing.
    assert store_a.sessions.get(SessionId("sess-ret1")) is None


def test_retirement_covers_cascaded_delegates_and_bulk_deletes(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("sess-ret2"))
    store_a.sessions.create(
        _new(
            "sess-ret2-sub",
            parent_session_id=SessionId("sess-ret2"),
            lineage_kind=SessionLineageKind.DELEGATE,
        ),
    )
    store_a.sessions.create(_new("sess-ret2-bulk"))
    assert store_a.sessions.delete(SessionId("sess-ret2"), media_root=MEDIA_ROOT) == ("sess-ret2", "sess-ret2-sub")
    assert store_a.sessions.delete_many([SessionId("sess-ret2-bulk")], media_root=MEDIA_ROOT).deleted == 1
    # CONTRACT: every removed id is retired, cascaded delegates and bulk rows included.
    for sid in ("sess-ret2", "sess-ret2-sub", "sess-ret2-bulk"):
        assert is_retired(backend, scope_a, SessionId(sid)) is True
        with pytest.raises(RetiredSessionError):
            store_a.sessions.create(_new(sid))


def test_delete_if_empty_retires_too(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("sess-ret3"))
    assert store_a.sessions.delete_if_empty(SessionId("sess-ret3"), media_root=MEDIA_ROOT) == ("sess-ret3",)
    assert is_retired(backend, scope_a, SessionId("sess-ret3")) is True
    with pytest.raises(RetiredSessionError):
        store_a.sessions.create(_new("sess-ret3"))


def test_retirement_visibility(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("sess-ret4"))
    # CONTRACT: a live row and an unknown id are not retired; only deletion retires.
    assert is_retired(backend, scope_a, SessionId("sess-ret4")) is False
    assert is_retired(backend, scope_a, SessionId("sess-never-existed")) is False
    assert store_a.sessions.delete(SessionId("sess-ret4"), media_root=MEDIA_ROOT) == ("sess-ret4",)
    assert is_retired(backend, scope_a, SessionId("sess-ret4")) is True


def test_delete_refuses_busy_session_untouched(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("sess-busy1", model="keep-me"))
    store_a.sessions.seed_ledger_admission(SessionId("sess-busy1"), status="queued")
    # CONTRACT: live ledger work refuses the delete with the authority's session_busy refusal.
    with pytest.raises(AuthorityRefusal) as excinfo:
        store_a.sessions.delete(SessionId("sess-busy1"), media_root=MEDIA_ROOT)
    assert _busy(excinfo)
    # CONTRACT: the refusal removed and retired nothing.
    row = store_a.sessions.get(SessionId("sess-busy1"))
    assert row is not None and row.model == "keep-me"


def test_delete_refuses_started_work_and_delete_if_empty_agrees(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("sess-busy2"))
    store_a.sessions.seed_ledger_admission(SessionId("sess-busy2"), status="started")
    with pytest.raises(AuthorityRefusal) as busy:
        store_a.sessions.delete(SessionId("sess-busy2"), media_root=MEDIA_ROOT)
    assert _busy(busy)
    # CONTRACT: an admitted-against session is not empty in the ledger sense either.
    with pytest.raises(AuthorityRefusal) as busy:
        store_a.sessions.delete_if_empty(SessionId("sess-busy2"), media_root=MEDIA_ROOT)
    assert _busy(busy)
    assert store_a.sessions.get(SessionId("sess-busy2")) is not None


def test_delete_refuses_unknown_work_with_the_barrier(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("sess-unk1"))
    store_a.sessions.seed_ledger_admission(SessionId("sess-unk1"), status="unknown")
    # CONTRACT: unknown work is its own fail-closed barrier, not a busy signal or an automatic requeue; explicit resolution or proof-verified worker adoption clears it.
    with pytest.raises(UnknownExecution):
        store_a.sessions.delete(SessionId("sess-unk1"), media_root=MEDIA_ROOT)
    assert store_a.sessions.get(SessionId("sess-unk1")) is not None


def test_delete_refusal_precedence_unknown_wins_over_busy(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("sess-mixed1"))
    store_a.sessions.seed_ledger_admission(SessionId("sess-mixed1"), status="queued")
    store_a.sessions.seed_ledger_admission(SessionId("sess-mixed1"), status="unknown")
    # CONTRACT: when one target holds both live and unknown work, the unknown
    # barrier wins (unknown is checked before busy per target).
    with pytest.raises(UnknownExecution):
        store_a.sessions.delete(SessionId("sess-mixed1"), media_root=MEDIA_ROOT)
    assert store_a.sessions.get(SessionId("sess-mixed1")) is not None


def test_delete_many_refuses_batch_on_busy_delegate_descendant(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("sess-cascade-par"))
    store_a.sessions.create(
        _new(
            "sess-cascade-sub",
            parent_session_id=SessionId("sess-cascade-par"),
            lineage_kind=SessionLineageKind.DELEGATE,
        ),
    )
    store_a.sessions.create(_new("sess-cascade-innocent"))
    store_a.sessions.seed_ledger_admission(SessionId("sess-cascade-sub"), status="queued")
    # CONTRACT: the bulk delete cascades delegate children, so a busy DESCENDANT
    # refuses the whole batch; nothing is removed or retired, the innocent row included.
    with pytest.raises(AuthorityRefusal) as busy:
        store_a.sessions.delete_many(
            [SessionId("sess-cascade-par"), SessionId("sess-cascade-innocent")], media_root=MEDIA_ROOT
        )
    assert _busy(busy)
    for sid in ("sess-cascade-par", "sess-cascade-sub", "sess-cascade-innocent"):
        assert store_a.sessions.get(SessionId(sid)) is not None


def test_delete_many_cascades_and_retires_delegate_descendants(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("sess-cascade2-par"))
    store_a.sessions.create(
        _new(
            "sess-cascade2-sub",
            parent_session_id=SessionId("sess-cascade2-par"),
            lineage_kind=SessionLineageKind.DELEGATE,
        ),
    )
    # CONTRACT: the count reports named rows; the cascaded delegate is removed
    # (and named in removed_ids) and retired all the same.
    outcome = store_a.sessions.delete_many([SessionId("sess-cascade2-par")], media_root=MEDIA_ROOT)
    assert (outcome.deleted, outcome.removed_ids) == (1, ("sess-cascade2-par", "sess-cascade2-sub"))
    for sid in ("sess-cascade2-par", "sess-cascade2-sub"):
        assert store_a.sessions.get(SessionId(sid)) is None
        assert is_retired(backend, scope_a, SessionId(sid)) is True


def test_delete_many_refuses_whole_batch_on_one_busy_target(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("sess-batch-ok"))
    store_a.sessions.create(_new("sess-batch-busy"))
    store_a.sessions.seed_ledger_admission(SessionId("sess-batch-busy"), status="queued")
    # CONTRACT: the bulk delete is one atomic transition; a busy target refuses the
    # whole batch and the innocent row survives.
    with pytest.raises(AuthorityRefusal) as busy:
        store_a.sessions.delete_many(
            [SessionId("sess-batch-ok"), SessionId("sess-batch-busy")], media_root=MEDIA_ROOT
        )
    assert _busy(busy)
    assert store_a.sessions.get(SessionId("sess-batch-ok")) is not None


def test_delete_empty_skips_busy_rows_instead_of_refusing(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("sess-sweep-idle"))
    store_a.sessions.end(SessionId("sess-sweep-idle"), "cli_close")
    store_a.sessions.create(_new("sess-sweep-busy"))
    store_a.sessions.end(SessionId("sess-sweep-busy"), "cli_close")
    store_a.sessions.seed_ledger_admission(SessionId("sess-sweep-busy"), status="queued")
    # CONTRACT: the sweep never aborts on one busy row; it removes the idle set and
    # leaves the busy row untouched and unretired (retire_prunable polarity).
    assert store_a.sessions.delete_empty(media_root=MEDIA_ROOT) == ("sess-sweep-idle",)
    assert store_a.sessions.get(SessionId("sess-sweep-idle")) is None
    assert is_retired(backend, scope_a, SessionId("sess-sweep-idle")) is True
    assert store_a.sessions.get(SessionId("sess-sweep-busy")) is not None


def test_delete_drops_routing_entries_atomically(backend, scope_a):
    store_a = backend.open(scope_a)
    now = datetime(2026, 9, 16, 12, 0, 0, tzinfo=UTC)
    store_a.sessions.create(_new("sess-routed"))
    store_a.sessions.create(_new("sess-route-survivor"))
    for key, sid in (("key-doomed", "sess-routed"), ("key-survivor", "sess-route-survivor")):
        store_a.routing.upsert_routing_entry(key, json.dumps({"session_key": key, "session_id": sid, "created_at": now.isoformat()}), routing_scope="ns-ret")
    assert store_a.sessions.delete(SessionId("sess-routed"), media_root=MEDIA_ROOT) == ("sess-routed",)
    # CONTRACT: route retirement rides the delete; no route resurrects the id,
    # and unrelated entries survive.
    entries = store_a.routing.load_routing_entries(routing_scope="ns-ret")
    assert "key-doomed" not in entries
    assert json.loads(entries["key-survivor"])["session_id"] == "sess-route-survivor"


def test_retirement_is_scoped(backend, scope_a, scope_b):
    store_a = backend.open(scope_a)
    store_b = backend.open(scope_b)
    shared = SessionId("sess-ret-shared")
    store_a.sessions.create(_new(str(shared)))
    assert store_a.sessions.delete(shared, media_root=MEDIA_ROOT) == (shared,)
    # Two-scope isolation: retirement in one scope neither marks nor blocks the other.
    assert is_retired(backend, scope_b, shared) is False
    store_b.sessions.create(_new(str(shared), model="b-life"))
    assert store_b.sessions.get(shared).model == "b-life"
    assert is_retired(backend, scope_a, shared) is True


def test_delete_retirement_publishes_atomically_under_injected_crash(backend, scope_a):
    store_a = backend.open(scope_a)
    require_failpoints(backend)
    from tests.contracts._reference._authority import InjectedFailure, authority_state

    now = datetime(2026, 9, 16, 12, 0, 0, tzinfo=UTC)
    store_a.sessions.create(_new("sess-crash1", model="survivor"))
    store_a.routing.upsert_routing_entry("key-crash", json.dumps({"session_key": "key-crash", "session_id": "sess-crash1", "created_at": now.isoformat()}), routing_scope="ns-crash")
    state = authority_state(backend, scope_a)
    state.failpoints.arm(state.PUBLISH)
    # A crash immediately before publication leaves NO partial transition:
    # the row, its route, and the unretired marker all still read committed state.
    with pytest.raises(InjectedFailure):
        store_a.sessions.delete(SessionId("sess-crash1"), media_root=MEDIA_ROOT)
    row = store_a.sessions.get(SessionId("sess-crash1"))
    assert row is not None and row.model == "survivor"
    entries = store_a.routing.load_routing_entries(routing_scope="ns-crash")
    assert "key-crash" in entries
    assert SessionId("sess-crash1") not in state.snapshot["retired"]
    # Exactly-once retry: the failpoint self-disarms and the same delete commits whole.
    assert store_a.sessions.delete(SessionId("sess-crash1"), media_root=MEDIA_ROOT) == ("sess-crash1",)
    assert store_a.sessions.get(SessionId("sess-crash1")) is None
    assert is_retired(backend, scope_a, SessionId("sess-crash1")) is True
    assert "key-crash" not in store_a.routing.load_routing_entries(routing_scope="ns-crash")


# ── persistence.conversation.sessions.SessionRepository.roster_sessions: the latest local chat ───────────────


def test_roster_latest_local_chat_admission_and_owner_filter(backend, scope_a):
    # Admissible rows FIRST, excluded rows LAST: with no activity watermark the
    # recency key is started_at, so any broken filter would make an excluded
    # row (newer) win. That makes each predicate individually discriminating.
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("sess-lls-alice", chat_id="local-a", user_id="alice"))
    store_a.sessions.create(_new("sess-lls-bob", chat_id="local-b", user_id="bob"))
    store_a.sessions.create(_new("sess-lls-platform", chat_id="telegram-99", user_id="alice"))
    store_a.sessions.create(_new("sess-lls-nochat", user_id="alice"))
    store_a.sessions.create(_new("sess-lls-archived", chat_id="local-arch", user_id="alice"))
    store_a.sessions.set_archived(SessionId("sess-lls-archived"), True)
    # CONTRACT: owner-filtered reads admit only local-prefixed, unarchived rows of that owner.
    alice = store_a.sessions.roster_sessions(owner_user_id="alice", owner=None).last_session
    assert alice is not None and alice.id == SessionId("sess-lls-alice")
    bob = store_a.sessions.roster_sessions(owner_user_id="bob", owner=None).last_session
    assert bob is not None and bob.id == SessionId("sess-lls-bob")
    # CONTRACT: an owner whose only rows are excluded reads None (carol has none at all;
    # alice's newer rows were all excluded above, proving each predicate bites).
    assert store_a.sessions.roster_sessions(owner_user_id="carol", owner=None).last_session is None
    # CONTRACT: the operator read (owner None) spans every owner and picks the most recent.
    operator = store_a.sessions.roster_sessions(owner_user_id=None, owner=None).last_session
    assert operator is not None and operator.id == SessionId("sess-lls-bob")


def test_roster_latest_local_chat_recency_is_newest_started(backend, scope_a):
    # Without an activity stamp the recency key collapses to started_at: latest creation wins. Ids chosen so the id tiebreak agrees with creation order on equal stamps.
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("sess-lls-1-old", chat_id="local-old", user_id="u1"))
    store_a.sessions.create(_new("sess-lls-2-new", chat_id="local-new", user_id="u1"))
    latest = store_a.sessions.roster_sessions(owner_user_id="u1", owner=None).last_session
    assert latest is not None and latest.id == SessionId("sess-lls-2-new")


def test_roster_latest_local_chat_skips_hidden_rows(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("sess-lls-shown", chat_id="local-s", user_id="u2"))
    store_a.sessions.create(_new("sess-lls-hidden", chat_id="local-h", user_id="u2"))
    store_a.sessions.set_hidden(SessionId("sess-lls-hidden"), True)
    # CONTRACT: the roster's last session is the newest chat a picker shows; a newer hidden row never wins.
    found = store_a.sessions.roster_sessions(owner_user_id="u2", owner=None).last_session
    assert found is not None and found.id == SessionId("sess-lls-shown")


def test_roster_refuses_on_a_store_closed_under_its_handle(backend, scope_a, scope_b):
    store_a, store_b = backend.open(scope_a), backend.open(scope_b)
    store_a.sessions.create(_new("sess-roster-closed", chat_id="local-c", user_id="u3"))
    store_b.sessions.create(_new("sess-roster-other", chat_id="local-o", user_id="u3"))
    assert store_a.sessions.roster_sessions(owner_user_id="u3", owner=None).last_session.id == SessionId("sess-roster-closed")
    backend.close_store_under(scope_a)
    # CONTRACT: the roster read of a store closed under its handle (a profile being deleted) refuses on every poll instead of opening it again.
    for _poll in range(2):
        with pytest.raises(PermanentStoreError):
            store_a.sessions.roster_sessions(owner_user_id="u3", owner=None)
    # CONTRACT: another profile's store reads on.
    assert store_b.sessions.roster_sessions(owner_user_id="u3", owner=None).last_session.id == SessionId("sess-roster-other")


def test_retirement_marker_survives_reopen(backend, scope_a):
    store_a = backend.open(scope_a)
    require_capability(backend, "reopen_durability")
    store_a.sessions.create(_new("sess-ret-reopen"))
    assert store_a.sessions.delete(SessionId("sess-ret-reopen"), media_root=MEDIA_ROOT) == ("sess-ret-reopen",)
    backend.reopen()
    # CONTRACT: retirement is durable; the fence outlives the handle, so a
    # late constructor after restart still cannot resurrect the id.
    assert is_retired(backend, scope_a, SessionId("sess-ret-reopen")) is True
    with pytest.raises(RetiredSessionError):
        store_a.sessions.create(_new("sess-ret-reopen"))


# ── persistence.conversation.sessions.SessionRepository.retag_worker_sessions ─────────────────────────────────────


def _source(backend, scope, session_id: str) -> str:
    store = backend.open(scope)
    row = store.sessions.get(SessionId(session_id))
    assert row is not None
    return row.source


def test_retag_rewrites_cli_rows_at_and_under_root_only(backend, scope_a):
    store_a = backend.open(scope_a)
    root = "/ws/kanban"
    store_a.sessions.create(_new("rt-exact", cwd=root))
    store_a.sessions.create(_new("rt-child", cwd=root + "/task-1"))
    store_a.sessions.create(_new("rt-deep", cwd=root + "/task-1/sub"))
    store_a.sessions.create(_new("rt-sibling", cwd=root + "x/other"))
    store_a.sessions.create(_new("rt-underscore", cwd=root + "_old/other"))
    store_a.sessions.create(_new("rt-elsewhere", cwd="/elsewhere"))
    store_a.sessions.create(_new("rt-nocwd"))
    store_a.sessions.create(_new("rt-gateway", source="telegram", cwd=root + "/task-2"))
    store_a.sessions.create(_new("rt-already", source=RETAG_SOURCE_TO, cwd=root + "/task-3"))
    # CONTRACT: the count is exactly the cli rows at or under the root (3 here); a trailing separator on the root is stripped first.
    assert store_a.sessions.retag_worker_sessions(root + "/") == 3
    for sid in ("rt-exact", "rt-child", "rt-deep"):
        assert _source(backend, scope_a, sid) == RETAG_SOURCE_TO
    # CONTRACT: `<root>x` and `<root>_old` are not descendants (the descendant arm requires the separator), and rows elsewhere or without a cwd are untouched.
    for sid in ("rt-sibling", "rt-underscore", "rt-elsewhere", "rt-nocwd"):
        assert _source(backend, scope_a, sid) == RETAG_SOURCE_FROM
    # CONTRACT: only source == "cli" rows change; other sources under the root are untouched.
    assert _source(backend, scope_a, "rt-gateway") == "telegram"
    assert _source(backend, scope_a, "rt-already") == RETAG_SOURCE_TO


def test_retag_is_once_per_root_and_gate_is_observable(backend, scope_a):
    store_a = backend.open(scope_a)
    root = "/ws/once"
    store_a.sessions.create(_new("rt-once-1", cwd=root + "/a"))
    assert store_a.state_meta.get(RETAG_GATE_PREFIX + root) is None
    assert store_a.sessions.retag_worker_sessions(root) == 1
    # CONTRACT: the gate lands under RETAG_GATE_PREFIX + normalized root in the same scope, readable through StateMetaRepository.
    assert store_a.state_meta.get(RETAG_GATE_PREFIX + root) == "1"
    store_a.sessions.create(_new("rt-once-2", cwd=root + "/b"))
    # CONTRACT: a repeat call returns 0 and touches no row, even though a new cli row appeared under the root since.
    assert store_a.sessions.retag_worker_sessions(root + "\\") == 0
    assert _source(backend, scope_a, "rt-once-2") == RETAG_SOURCE_FROM


def test_retag_zero_match_still_records_gate(backend, scope_a):
    store_a = backend.open(scope_a)
    root = "/ws/empty-at-first"
    assert store_a.sessions.retag_worker_sessions(root) == 0
    # CONTRACT: zero-match gating; the root is now permanently retagged for this scope.
    assert store_a.state_meta.get(RETAG_GATE_PREFIX + root) == "1"
    store_a.sessions.create(_new("rt-late", cwd=root + "/late"))
    assert store_a.sessions.retag_worker_sessions(root) == 0
    assert _source(backend, scope_a, "rt-late") == RETAG_SOURCE_FROM


def test_retag_honors_a_preexisting_gate(backend, scope_a):
    store_a = backend.open(scope_a)
    root = "/ws/pregated"
    store_a.sessions.create(_new("rt-pregated", cwd=root + "/a"))
    store_a.state_meta.set(RETAG_GATE_PREFIX + root, "1")
    # CONTRACT: a gate already reading "1" short-circuits before any row is touched.
    assert store_a.sessions.retag_worker_sessions(root) == 0
    assert _source(backend, scope_a, "rt-pregated") == RETAG_SOURCE_FROM
    # A gate holding any OTHER value does not short-circuit (the check is for "1").
    other = "/ws/othervalue"
    store_a.sessions.create(_new("rt-othervalue", cwd=other + "/a"))
    store_a.state_meta.set(RETAG_GATE_PREFIX + other, "0")
    assert store_a.sessions.retag_worker_sessions(other) == 1
    assert store_a.state_meta.get(RETAG_GATE_PREFIX + other) == "1"


def test_retag_empty_root_is_a_no_op_without_gate(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("rt-empty-root", cwd="/anything"))
    for root in ("", "/", "///", "\\"):
        # CONTRACT: an empty normalized root returns 0, retags nothing, and records no gate (the empty-prefix gate would otherwise brand every future root).
        assert store_a.sessions.retag_worker_sessions(root) == 0
    assert store_a.state_meta.get(RETAG_GATE_PREFIX) is None
    assert _source(backend, scope_a, "rt-empty-root") == RETAG_SOURCE_FROM


def test_retag_wildcards_in_root_are_literal(backend, scope_a):
    store_a = backend.open(scope_a)
    root = "/ws/w%ld_card"
    store_a.sessions.create(_new("rt-wild-literal", cwd=root + "/x"))
    store_a.sessions.create(_new("rt-wild-pattern", cwd="/ws/wXXld-card/x"))
    store_a.sessions.create(_new("rt-wild-pattern2", cwd="/ws/wildYcard/x"))
    # CONTRACT: `%` and `_` in the root are path characters, never patterns.
    assert store_a.sessions.retag_worker_sessions(root) == 1
    assert _source(backend, scope_a, "rt-wild-literal") == RETAG_SOURCE_TO
    assert _source(backend, scope_a, "rt-wild-pattern") == RETAG_SOURCE_FROM
    assert _source(backend, scope_a, "rt-wild-pattern2") == RETAG_SOURCE_FROM


def test_retag_exact_arm_is_case_sensitive(backend, scope_a):
    store_a = backend.open(scope_a)
    root = "/ws/Case"
    store_a.sessions.create(_new("rt-case-exact", cwd=root))
    store_a.sessions.create(_new("rt-case-other", cwd="/ws/case"))
    # CONTRACT: the exact arm compares character-for-character; a differently-cased cwd equal to the root is not the root.
    # (Descendant-arm case behavior is deliberately unpromised: QUIRK(like-ascii-case) on the SQLite adapter.)
    assert store_a.sessions.retag_worker_sessions(root) == 1
    assert _source(backend, scope_a, "rt-case-exact") == RETAG_SOURCE_TO
    assert _source(backend, scope_a, "rt-case-other") == RETAG_SOURCE_FROM


def test_retag_is_visibility_blind(backend, scope_a):
    store_a = backend.open(scope_a)
    root = "/ws/blind"
    store_a.sessions.create(_new("rt-hidden", cwd=root + "/h"))
    store_a.sessions.set_hidden(SessionId("rt-hidden"), True)
    store_a.sessions.create(_new("rt-archived", cwd=root + "/a"))
    store_a.sessions.set_archived(SessionId("rt-archived"), True)
    store_a.sessions.create(_new("rt-ended", cwd=root + "/e"))
    store_a.sessions.end(SessionId("rt-ended"), "cli_close")
    store_a.sessions.create(
        _new("rt-delegate", cwd=root + "/d", parent_session_id=SessionId("rt-ended"), lineage_kind=SessionLineageKind.DELEGATE),
    )
    # CONTRACT: hidden, archived, ended, and lineage-child rows under the root are retagged like any other; no listing admission applies.
    assert store_a.sessions.retag_worker_sessions(root) == 4
    for sid in ("rt-hidden", "rt-archived", "rt-ended", "rt-delegate"):
        assert _source(backend, scope_a, sid) == RETAG_SOURCE_TO


def test_retag_scope_isolation(backend, scope_a, scope_b):
    store_a = backend.open(scope_a)
    store_b = backend.open(scope_b)
    root = "/ws/scoped"
    store_a.sessions.create(_new("rt-scope-a", cwd=root + "/a"))
    store_b.sessions.create(_new("rt-scope-b", cwd=root + "/b"))
    assert store_a.sessions.retag_worker_sessions(root) == 1
    # CONTRACT: rows and the gate in another scope are neither read nor written.
    assert _source(backend, scope_b, "rt-scope-b") == RETAG_SOURCE_FROM
    assert store_b.state_meta.get(RETAG_GATE_PREFIX + root) is None
    assert store_a.state_meta.get(RETAG_GATE_PREFIX + root) == "1"
    # CONTRACT: scope_a's gate does not short-circuit scope_b's own first retag.
    assert store_b.sessions.retag_worker_sessions(root) == 1
    assert _source(backend, scope_b, "rt-scope-b") == RETAG_SOURCE_TO


# ── persistence.conversation.sessions.SessionRepository.replace_session_config ────────────────────────────────────


def test_replace_session_config_keeps_model_when_none(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("rsc-keep", model="m1", model_config={"old": 1, "gone": True}))
    sid = SessionId("rsc-keep")
    result = store_a.sessions.replace_session_config(sid, ReplaceSessionConfig({"fresh": "yes"}))
    # CONTRACT: a None-returning command.
    assert result is None
    row = store_a.sessions.get(sid)
    # CONTRACT: model=None keeps the stored model; the config is replaced wholesale (old keys gone, not merged).
    assert row.model == "m1"
    assert dict(row.model_config) == {"fresh": "yes"}
    assert store_a.sessions.get_model_config_value(sid, "gone", default="absent") == "absent"


def test_replace_session_config_overwrites_model_only_in_the_model_column(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("rsc-model", model="m1", model_config={"cwd": "/a"}))
    sid = SessionId("rsc-model")
    store_a.sessions.replace_session_config(sid, ReplaceSessionConfig({"cwd": "/b"}, model="m2"))
    row = store_a.sessions.get(sid)
    # CONTRACT: the model column changes and NOTHING is merged into model_config (unlike update_model).
    assert row.model == "m2"
    assert dict(row.model_config) == {"cwd": "/b"}
    assert store_a.sessions.get_model_config_value(sid, "model", default="absent") == "absent"


def test_replace_session_config_unknown_id_is_a_silent_no_op(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.replace_session_config(
        SessionId("rsc-missing"), ReplaceSessionConfig({"x": 1}, model="m9")
    )
    # CONTRACT: nothing is created.
    assert store_a.sessions.get(SessionId("rsc-missing")) is None


def test_replace_session_config_rejects_empty_model_before_the_store(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("rsc-empty", model="m1"))
    # CONTRACT: an empty-string model is an input-shape violation on the record itself.
    with pytest.raises(ValueError):
        ReplaceSessionConfig({}, model="")
    assert (store_a.sessions.get(SessionId("rsc-empty"))).model == "m1"


def test_replace_session_config_lineage_follows_the_written_markers(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("rsc-parent"))
    store_a.sessions.create(
        _new("rsc-delegate", parent_session_id=SessionId("rsc-parent"), lineage_kind=SessionLineageKind.DELEGATE, model_config={"seed": 1}),
    )
    store_a.sessions.create(_new("rsc-branch", parent_session_id=SessionId("rsc-parent"), lineage_kind=SessionLineageKind.BRANCH))
    assert store_a.sessions.delete_targets(SessionId("rsc-parent")) == (SessionId("rsc-parent"), SessionId("rsc-delegate"))
    store_a.sessions.replace_session_config(SessionId("rsc-delegate"), ReplaceSessionConfig({"after": 2}, model="m2"))
    store_a.sessions.replace_session_config(SessionId("rsc-branch"), ReplaceSessionConfig({}))
    delegate = store_a.sessions.get(SessionId("rsc-delegate"))
    branch = store_a.sessions.get(SessionId("rsc-branch"))
    # CONTRACT: the config is replaced whole, fork markers included; the lineage kind is read from the markers, so it is gone too.
    assert dict(delegate.model_config) == {"after": 2} and delegate.lineage_kind is SessionLineageKind.NONE
    assert dict(branch.model_config) == {} and branch.lineage_kind is SessionLineageKind.NONE
    assert delegate.parent_session_id == SessionId("rsc-parent")
    # CONTRACT: the delete cascade follows the markers: the former delegate no longer cascades with its parent.
    assert store_a.sessions.delete_targets(SessionId("rsc-parent")) == (SessionId("rsc-parent"),)
    # CONTRACT: a marker the replace writes is lineage like any other: it names the cascade root.
    store_a.sessions.create(_new("rsc-other"))
    store_a.sessions.replace_session_config(SessionId("rsc-delegate"), ReplaceSessionConfig({"_delegate_from": "rsc-other", "after": 3}))
    delegate = store_a.sessions.get(SessionId("rsc-delegate"))
    assert delegate.lineage_kind is SessionLineageKind.DELEGATE and delegate.lineage_origin_id == SessionId("rsc-other")
    assert store_a.sessions.delete_targets(SessionId("rsc-other")) == (SessionId("rsc-other"), SessionId("rsc-delegate"))


def test_replace_session_config_scope_isolation(backend, scope_a, scope_b):
    store_a = backend.open(scope_a)
    store_b = backend.open(scope_b)
    store_a.sessions.create(_new("rsc-shared", model="ma", model_config={"who": "a"}))
    store_b.sessions.create(_new("rsc-shared", model="mb", model_config={"who": "b"}))
    store_a.sessions.replace_session_config(SessionId("rsc-shared"), ReplaceSessionConfig({"who": "a2"}, model="ma2"))
    row_b = store_b.sessions.get(SessionId("rsc-shared"))
    # CONTRACT: the same id in another scope is untouched.
    assert row_b.model == "mb" and dict(row_b.model_config) == {"who": "b"}
    row_a = store_a.sessions.get(SessionId("rsc-shared"))
    assert row_a.model == "ma2" and dict(row_a.model_config) == {"who": "a2"}


def test_replace_session_config_after_usage_delta_lands_on_flushed_state(backend, scope_a):
    store_a = backend.open(scope_a)
    from persistence.conversation.usage import BillingRoute, TokenCounts, UsageDelta
    from tests.contracts.conversation.test_usage import accounting

    store_a.sessions.create(_new("rsc-usage", model="m1"))
    sid = SessionId("rsc-usage")
    store_a.usage.queue_usage(
        sid, UsageDelta(tokens=TokenCounts(input_tokens=3), api_call_count=1, route=BillingRoute(model="m1", billing_provider="p")),
    )
    store_a.sessions.replace_session_config(sid, ReplaceSessionConfig({"after": True}, model="m2"))
    row = store_a.sessions.get(sid)
    # CONTRACT: the accounting barrier holds: the replace lands after the queued delta, so the model overwrite is the final state and the delta's own route write never resurrects m1.
    assert row.model == "m2" and dict(row.model_config) == {"after": True}
    assert (accounting(store_a, sid)).tokens.input_tokens == 3


# ── persistence.conversation.sessions.SessionRepository.list_gateway_sessions ────────────


_GW_BASE = datetime(2026, 3, 1, 12, 0, 0, tzinfo=UTC)


def _gw_at(seconds: int) -> datetime:
    return _GW_BASE + timedelta(seconds=seconds)


def _seed_keyed(backend, scope, session_id: str, key: str | None, started: int, *,
                      source: str = "telegram", ended: int | None = None, parent: str | None = None) -> None:
    """Public seeding of a keyed row with an exact ``started_at``: the import writes the timestamp and parent edge; the create upsert fills the routing key (null-fill semantics)."""
    store = backend.open(scope)
    payload = {
        "id": session_id, "source": source, "started_at": _gw_at(started).timestamp(),
        "ended_at": _gw_at(ended).timestamp() if ended is not None else None,
        "end_reason": "cli_close" if ended is not None else None,
        "parent_session_id": parent, "messages": [],
    }
    result = store.maintenance.import_sessions([payload])
    assert result.ok and result.imported == 1, result
    if key is not None:
        store.sessions.create(_new(session_id, source=source, session_key=key))


def _gw_ids(backend, scope, **filters) -> list:
    store = backend.open(scope)
    return [entry.session.session_id for entry in store.sessions.list_gateway_sessions(**filters)]


# ── persistence.conversation.sessions.SessionRepository.descendants ───────────────────────────


def _desc_ids(backend, scope, root: str) -> set:
    store = backend.open(scope)
    return {row.session_id for row in store.sessions.descendants(SessionId(root))}


def test_descendants_include_root_and_every_reachable_row(backend, scope_a):
    # R(0) -> A(10) -> A1(100); R -> B(20); B -> B1(20, tie with B); C(5) is unrelated.
    store_a = backend.open(scope_a)
    _seed_keyed(backend, scope_a, "ds-R", None, 0)
    _seed_keyed(backend, scope_a, "ds-A", None, 10, parent="ds-R")
    _seed_keyed(backend, scope_a, "ds-A1", None, 100, parent="ds-A")
    _seed_keyed(backend, scope_a, "ds-B", None, 20, parent="ds-R")
    _seed_keyed(backend, scope_a, "ds-B1", None, 20, parent="ds-B")
    _seed_keyed(backend, scope_a, "ds-C", None, 5)
    rows = store_a.sessions.descendants(SessionId("ds-R"))
    # CONTRACT: the root itself, every child and grandchild, nothing unrelated, each once; the walk defines no order.
    assert sorted(row.session_id for row in rows) == sorted(SessionId(x) for x in ("ds-R", "ds-A", "ds-B", "ds-B1", "ds-A1"))
    # CONTRACT: full records, so the caller can read the parent edge and started_at it walks on.
    by_id = {row.session_id: row for row in rows}
    assert by_id[SessionId("ds-A1")].parent_session_id == SessionId("ds-A")
    assert by_id[SessionId("ds-B")].started_at == by_id[SessionId("ds-B1")].started_at
    # CONTRACT: a subtree root reads its own closure only.
    assert _desc_ids(backend, scope_a, "ds-A") == {SessionId("ds-A"), SessionId("ds-A1")}
    assert _desc_ids(backend, scope_a, "ds-A1") == {SessionId("ds-A1")}
    # CONTRACT: an unknown root reads as an empty tuple.
    assert store_a.sessions.descendants(SessionId("ds-nope")) == ()


def test_descendants_follow_continuation_edges_across_visibility(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("dl-root"))
    store_a.sessions.create(_new("dl-delegate", parent_session_id=SessionId("dl-root"), lineage_kind=SessionLineageKind.DELEGATE))
    store_a.sessions.create(_new("dl-branch", parent_session_id=SessionId("dl-root"), lineage_kind=SessionLineageKind.BRANCH))
    store_a.sessions.create(_new("dl-tool", source="tool", parent_session_id=SessionId("dl-root")))
    store_a.sessions.create(_new("dl-under-delegate", parent_session_id=SessionId("dl-delegate")))
    store_a.sessions.create(_new("dl-plain", parent_session_id=SessionId("dl-root")))
    store_a.sessions.end(SessionId("dl-root"), "compression")
    store_a.sessions.create(_new("dl-fork", parent_session_id=SessionId("dl-root")))
    store_a.sessions.create(_new("dl-archived", parent_session_id=SessionId("dl-root")))
    store_a.sessions.set_archived(SessionId("dl-archived"), True)
    store_a.sessions.set_hidden(SessionId("dl-plain"), True)
    store_a.sessions.end(SessionId("dl-fork"), "cli_close")
    # CONTRACT: plain and compression children count, archived/hidden/ended included; delegate, branch and tool-owned children are excluded and never walked through.
    assert _desc_ids(backend, scope_a, "dl-root") == {
        SessionId(x) for x in ("dl-root", "dl-plain", "dl-fork", "dl-archived")
    }
    # CONTRACT: an excluded row still reads its own continuation closure as a root.
    assert _desc_ids(backend, scope_a, "dl-delegate") == {SessionId("dl-delegate"), SessionId("dl-under-delegate")}
    # CONTRACT: unlike delete_targets, the set is not the delegate cascade.
    assert store_a.sessions.delete_targets(SessionId("dl-root")) == (SessionId("dl-root"), SessionId("dl-delegate"))


def test_descendants_exclude_a_reset_child_of_the_same_routing_key(backend, scope_a):
    store_a = backend.open(scope_a)
    _seed_keyed(backend, scope_a, "dr-root", "agent:main:telegram:dm:1", 0)
    store_a.sessions.end(SessionId("dr-root"), "session_reset")
    _seed_keyed(backend, scope_a, "dr-reset", "agent:main:telegram:dm:1", 10, parent="dr-root")
    _seed_keyed(backend, scope_a, "dr-other-key", "agent:main:telegram:dm:2", 20, parent="dr-root")
    # CONTRACT: a child riding its parent's key after a reset-boundary end is a reset child, not a continuation.
    assert _desc_ids(backend, scope_a, "dr-root") == {SessionId("dr-root"), SessionId("dr-other-key")}


def test_descendants_terminate_and_deduplicate_on_a_parent_cycle(backend, scope_a):
    """A cycle can only enter through the import door (mutual parents in one payload). Backends that detach the closing edge keep the FIRST entry as the root; backends that store the cycle report the same set from that root, deduplicated."""
    store_a = backend.open(scope_a)
    def _header(sid: str, parent: str, started: int) -> dict:
        return {"id": sid, "source": "cli", "started_at": _gw_at(started).timestamp(), "parent_session_id": parent, "messages": []}

    result = store_a.maintenance.import_sessions([_header("dc-A", "dc-B", 10), _header("dc-B", "dc-A", 20)])
    assert result.ok and result.imported == 2 and result.detached in (0, 1)
    rows = store_a.sessions.descendants(SessionId("dc-A"))
    ids = [row.session_id for row in rows]
    # CONTRACT: the walk terminates and each row is reported exactly once.
    assert sorted(ids) == [SessionId("dc-A"), SessionId("dc-B")]
    # CONTRACT: closure property: every non-root row's parent is in the set (whichever edge survived import).
    assert all(row.session_id == SessionId("dc-A") or row.parent_session_id in set(ids) for row in rows)


def test_descendants_scope_isolation(backend, scope_a, scope_b):
    store_a = backend.open(scope_a)
    store_b = backend.open(scope_b)
    store_a.sessions.create(_new("di-root"))
    store_a.sessions.create(_new("di-child", parent_session_id=SessionId("di-root")))
    store_b.sessions.create(_new("di-root"))
    store_b.sessions.create(_new("di-other", parent_session_id=SessionId("di-root")))
    # CONTRACT: edges never cross scopes; the same root id reads its own scope's closure only.
    assert _desc_ids(backend, scope_a, "di-root") == {SessionId("di-root"), SessionId("di-child")}
    assert _desc_ids(backend, scope_b, "di-root") == {SessionId("di-root"), SessionId("di-other")}



def test_gateway_listing_is_newest_row_per_routing_key(backend, scope_a):
    store_a = backend.open(scope_a)
    _seed_keyed(backend, scope_a, "gw-k1-old", "agent:main:k1", 10)
    _seed_keyed(backend, scope_a, "gw-k1-new", "agent:main:k1", 20)
    _seed_keyed(backend, scope_a, "gw-k2", "agent:main:k2", 15)
    _seed_keyed(backend, scope_a, "gw-nokey", None, 30)
    entries = store_a.sessions.list_gateway_sessions()
    # CONTRACT: rows at the greatest started_at per routing key win, including ties; rows without a key never appear.
    assert {e.session.session_id for e in entries} == {SessionId("gw-k1-new"), SessionId("gw-k2")}
    winner = next(e for e in entries if e.session.session_id == "gw-k1-new")
    # CONTRACT: the entry carries the full record and the last_active facet (started_at when nothing else exists).
    assert winner.session.session_key == "agent:main:k1" and winner.session.started_at == _gw_at(20).timestamp()
    assert winner.last_active == _gw_at(20).timestamp()


def test_gateway_listing_newest_ended_row_suppresses_its_key(backend, scope_a):
    store_a = backend.open(scope_a)
    _seed_keyed(backend, scope_a, "gw-sup-live", "agent:main:sup", 10)
    _seed_keyed(backend, scope_a, "gw-sup-ended", "agent:main:sup", 20, ended=25)
    _seed_keyed(backend, scope_a, "gw-other", "agent:main:other", 5)
    # CONTRACT: the winner is picked BEFORE the active filter, so a key whose newest row has ended lists nothing under active_only even though an older live row exists.
    assert _gw_ids(backend, scope_a, active_only=True) == [SessionId("gw-other")]
    # CONTRACT: without active_only the ended winner is listed and the older live loser still is not.
    assert set(_gw_ids(backend, scope_a, active_only=False)) == {SessionId("gw-sup-ended"), SessionId("gw-other")}
    # CONTRACT: ending through the lifecycle verb has the same effect as an imported end stamp.
    store_a.sessions.end(SessionId("gw-other"), "cli_close")
    assert _gw_ids(backend, scope_a, active_only=True) == []


def test_gateway_listing_keeps_started_at_ties(backend, scope_a):
    _seed_keyed(backend, scope_a, "gw-tie-a", "agent:main:tie", 10)
    _seed_keyed(backend, scope_a, "gw-tie-b", "agent:main:tie", 10)
    _seed_keyed(backend, scope_a, "gw-tie-older", "agent:main:tie", 9)
    # CONTRACT: equal-started_at rows of one key are all winners; the strictly older row is not.
    assert set(_gw_ids(backend, scope_a)) == {SessionId("gw-tie-a"), SessionId("gw-tie-b")}


def test_gateway_listing_admits_hidden_archived_and_child_winners(backend, scope_a):
    store_a = backend.open(scope_a)
    _seed_keyed(backend, scope_a, "gw-parent", None, 1)
    _seed_keyed(backend, scope_a, "gw-hidden", "agent:main:h", 10)
    _seed_keyed(backend, scope_a, "gw-archived", "agent:main:a", 10)
    _seed_keyed(backend, scope_a, "gw-child", "agent:main:c", 10, parent="gw-parent")
    assert store_a.sessions.set_hidden(SessionId("gw-hidden"), True)
    assert store_a.sessions.set_archived(SessionId("gw-archived"), True).changed
    # CONTRACT: visibility flags and lineage do not gate the gateway listing; the SessionQuery admission rules are not this query.
    assert set(_gw_ids(backend, scope_a)) == {SessionId("gw-hidden"), SessionId("gw-archived"), SessionId("gw-child")}
    child = store_a.sessions.get(SessionId("gw-child"))
    assert child.parent_session_id == SessionId("gw-parent")


def test_gateway_listing_platform_filter_folds_ascii_case(backend, scope_a):
    _seed_keyed(backend, scope_a, "gw-tg", "agent:main:p1", 10, source="telegram")
    _seed_keyed(backend, scope_a, "gw-Tg", "agent:main:p2", 10, source="Telegram")
    _seed_keyed(backend, scope_a, "gw-dc", "agent:main:p3", 10, source="discord")
    # CONTRACT: platform matches source under ASCII case folding, on the winners.
    assert set(_gw_ids(backend, scope_a, platform="TELEGRAM")) == {SessionId("gw-tg"), SessionId("gw-Tg")}
    assert _gw_ids(backend, scope_a, platform="discord") == [SessionId("gw-dc")]
    assert _gw_ids(backend, scope_a, platform="slack") == []
    # CONTRACT: an empty platform is no filter.
    assert set(_gw_ids(backend, scope_a, platform="")) == set(_gw_ids(backend, scope_a))
    # CONTRACT: the platform filter runs AFTER the winner pick: a discord row that lost its key to a newer telegram row is not resurrected by filtering on discord.
    _seed_keyed(backend, scope_a, "gw-p3-newer", "agent:main:p3", 20, source="telegram")
    assert _gw_ids(backend, scope_a, platform="discord") == []


def test_gateway_listing_last_active_is_newest_transcript_row_of_any_visibility(backend, scope_a):
    store_a = backend.open(scope_a)
    from persistence.conversation.messages import NewMessage

    def _m(seconds: int, role: str = "user") -> NewMessage:
        return NewMessage(role=role, content=f"m{seconds}", timestamp=_gw_at(seconds).timestamp())

    for sid, key in (("gw-la-quiet", "la1"), ("gw-la-newer", "la2"), ("gw-la-older", "la3"), ("gw-la-compacted", "la4"), ("gw-la-rewound", "la5")):
        _seed_keyed(backend, scope_a, sid, f"agent:main:{key}", 100)
    store_a.messages.append(SessionId("gw-la-newer"), _m(500))
    store_a.messages.append(SessionId("gw-la-older"), _m(5))
    store_a.messages.append(SessionId("gw-la-compacted"), _m(900))
    store_a.messages.compact(SessionId("gw-la-compacted"), [_m(200, role="assistant")])
    store_a.messages.append(SessionId("gw-la-rewound"), _m(300))
    rewind_target = store_a.messages.append(SessionId("gw-la-rewound"), _m(950))
    assert store_a.messages.deactivate_message(SessionId("gw-la-rewound"), rewind_target) is True
    entries = store_a.sessions.list_gateway_sessions()
    facet = {e.session.session_id: e.last_active for e in entries}
    # CONTRACT: no transcript -> started_at; a newer row -> its timestamp.
    assert facet[SessionId("gw-la-quiet")] == _gw_at(100).timestamp()
    assert facet[SessionId("gw-la-newer")] == _gw_at(500).timestamp()
    # CONTRACT: a transcript older than started_at yields a last_active OLDER than started_at (the source's outcome, not smoothed).
    assert facet[SessionId("gw-la-older")] == _gw_at(5).timestamp()
    # CONTRACT: rows of every visibility count: the compaction-archived original at 900 and the rewound row at 950 still set their facets.
    assert facet[SessionId("gw-la-compacted")] == _gw_at(900).timestamp()
    assert facet[SessionId("gw-la-rewound")] == _gw_at(950).timestamp()
    # CONTRACT: order is last_active descending.
    assert [e.session.session_id for e in entries] == [
        SessionId("gw-la-rewound"), SessionId("gw-la-compacted"), SessionId("gw-la-newer"),
        SessionId("gw-la-quiet"), SessionId("gw-la-older"),
    ]


def test_gateway_listing_last_active_counts_imported_transcripts_and_mixed_offsets(backend, scope_a):
    """Rows that entered through the import door are stored transcript rows like any other; an offset-bearing timestamp orders by instant, not by its textual form."""
    store_a = backend.open(scope_a)
    from datetime import timezone

    from persistence.conversation.messages import NewMessage
    plus_two = timezone(timedelta(hours=2))
    imported_row = {"role": "user", "content": "imported", "timestamp": _gw_at(3600).astimezone(plus_two).timestamp()}
    assert store_a.maintenance.import_sessions([{"id": "gw-imp", "source": "telegram", "started_at": _gw_at(0).timestamp(), "messages": [imported_row]}]).imported == 1
    store_a.sessions.create(_new("gw-imp", source="telegram", session_key="agent:main:imp"))
    # A second key whose facet is the UTC instant 13:30 written as +02:00 (15:30+02:00); textually it sorts above 14:00+00:00, chronologically below it.
    _seed_keyed(backend, scope_a, "gw-earlier", "agent:main:e", 0)
    _seed_keyed(backend, scope_a, "gw-later", "agent:main:l", 0)
    store_a.messages.append(SessionId("gw-earlier"), NewMessage(role="user", content="e", timestamp=_gw_at(5400).astimezone(plus_two).timestamp()))
    store_a.messages.append(SessionId("gw-later"), NewMessage(role="user", content="l", timestamp=_gw_at(7200).timestamp()))
    entries = store_a.sessions.list_gateway_sessions()
    # CONTRACT: the imported row is a stored transcript row, so it is the facet.
    assert {e.session.session_id: e.last_active for e in entries}[SessionId("gw-imp")] == _gw_at(3600).timestamp()
    # CONTRACT: ordering is by instant.
    assert [e.session.session_id for e in entries] == [SessionId("gw-later"), SessionId("gw-earlier"), SessionId("gw-imp")]


def test_gateway_listing_leaves_last_active_ties_unordered(backend, scope_a):
    store_a = backend.open(scope_a)
    from persistence.conversation.messages import NewMessage

    for i in range(7):
        _seed_keyed(backend, scope_a, f"gw-pg-{i}", f"agent:main:pg{i}", 10)
    # The two message-bearing rows share last_active and started_at; they lead the listing in no defined order.
    for sid in ("gw-pg-2", "gw-pg-5"):
        store_a.messages.append(SessionId(sid), NewMessage(role="user", content="x", timestamp=_gw_at(50).timestamp()))
    full = _gw_ids(backend, scope_a)
    assert set(full[:2]) == {SessionId("gw-pg-5"), SessionId("gw-pg-2")} and len(full) == len(set(full)) == 7


def test_gateway_listing_scope_isolation(backend, scope_a, scope_b):
    store_a = backend.open(scope_a)
    _seed_keyed(backend, scope_a, "gw-iso", "agent:main:iso", 10, source="telegram")
    _seed_keyed(backend, scope_b, "gw-iso", "agent:main:iso", 20, source="discord", ended=25)
    # CONTRACT: the same id and key in another scope is neither listed nor able to win the key.
    assert _gw_ids(backend, scope_a) == [SessionId("gw-iso")]
    assert _gw_ids(backend, scope_b) == []
    assert _gw_ids(backend, scope_b, active_only=False) == [SessionId("gw-iso")]
    assert store_a.sessions.list_gateway_sessions()[0].session.source == "telegram"


# ── Exclusive create ─────────────────────────────────


def test_create_exclusive_writes_the_bare_row_once(backend, scope_a):
    store_a = backend.open(scope_a)
    from persistence.conversation.sessions import SessionExists

    config = {"browser_model_lock": {"provider": "p", "model": "m"}}
    record = store_a.sessions.create_exclusive(
        SessionId("sess-ex1"), source="api_server", model="m", model_config=config, system_prompt="be brief", title="  Ops\tboard  ")
    # CONTRACT: the creating transaction returns the stored row.
    assert store_a.sessions.get(SessionId("sess-ex1")) == record
    # CONTRACT: the row carries the given bare fields and the SANITIZED title; title_source stays unset; visible by default; no activity yet.
    assert (record.session_id, record.source, record.model) == ("sess-ex1", "api_server", "m")
    # CONTRACT: created_source and profile_name stay unset.
    assert (record.created_source, record.profile_name) == (None, None)
    assert dict(record.model_config) == config and record.system_prompt == "be brief"
    assert record.title == "Ops board" and record.title_source is None and record.hidden is False
    summary = store_a.sessions.summary(SessionId("sess-ex1"))
    assert (summary.session.message_count, summary.preview, summary.lineage_root_id) == (0, "", None)
    assert summary.last_active == record.started_at
    # CONTRACT: a second exclusive create of the id raises SessionExists carrying the stored row, and writes nothing.
    with pytest.raises(SessionExists) as exists:
        store_a.sessions.create_exclusive(SessionId("sess-ex1"), source="other", title="Else")
    assert exists.value.existing == record and isinstance(exists.value, Conflict)
    assert store_a.sessions.get(SessionId("sess-ex1")) == record
    # CONTRACT: an absent or all-whitespace title leaves the row untitled.
    assert store_a.sessions.create_exclusive(SessionId("sess-ex2"), source="bot_room", title="   ").title is None
    bare = store_a.sessions.create_exclusive(SessionId("sess-ex3"), source="cli")
    assert bare.title is None and bare.hidden is False and dict(bare.model_config) == {} and bare.system_prompt is None
    # CONTRACT: an oversized title is an input error and writes nothing.
    with pytest.raises(ValueError):
        store_a.sessions.create_exclusive(SessionId("sess-ex4"), source="cli", title="x" * (MAX_TITLE_LENGTH + 1))
    assert store_a.sessions.get(SessionId("sess-ex4")) is None


def test_create_exclusive_refuses_a_held_title_before_writing(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("sess-holder"))
    store_a.sessions.set_title(SessionId("sess-holder"), "Group: r1")
    # CONTRACT: the sanitized title held by another session is TitleConflict naming the holder; no row is created.
    with pytest.raises(TitleConflict) as conflict:
        store_a.sessions.create_exclusive(SessionId("sess-ex-late"), source="bot_room", title=" Group:  r1 ")
    assert (conflict.value.title, conflict.value.holder) == ("Group: r1", "sess-holder")
    assert store_a.sessions.get(SessionId("sess-ex-late")) is None
    # CONTRACT: existence is checked before the title, so an existing id with a held title is SessionExists.
    from persistence.conversation.sessions import SessionExists
    with pytest.raises(SessionExists):
        store_a.sessions.create_exclusive(SessionId("sess-holder"), source="cli", title="Group: r1")


def test_create_exclusive_refuses_a_retired_id(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("sess-ex-ret"))
    assert store_a.sessions.delete(SessionId("sess-ex-ret"), media_root=MEDIA_ROOT) == ("sess-ex-ret",)
    with pytest.raises(RetiredSessionError) as plain:
        store_a.sessions.create(_new("sess-ex-ret"))
    # CONTRACT: the exclusive create checks the retirement fence too, answering session_deleted where a plain create answers not_found; nothing is written.
    with pytest.raises(RetiredSessionError) as exclusive:
        store_a.sessions.create_exclusive(SessionId("sess-ex-ret"), source="cli")
    assert (plain.value.reason, exclusive.value.reason) == ("not_found", "session_deleted")
    assert store_a.sessions.get(SessionId("sess-ex-ret")) is None
