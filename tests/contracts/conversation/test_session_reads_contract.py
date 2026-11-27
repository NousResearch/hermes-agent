"""Contract cases for the ``SessionRepository`` verbs for queue-row retirement, the canonical Bot Chat title, ownership and scope reads, title holders, config-key scans, the empty-candidate probe, the picker and live-listing rows, the roster, the bounded search, and the store-wide firsts.

Every case seeds through public protocol operations and observes through public reads; the rules are the ``persistence/conversation/sessions.py`` docstrings.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from persistence import SessionId
from persistence.conversation.messages import NewMessage, TranscriptRead, TranscriptView
from persistence.conversation.sessions import (
    CANONICAL_BOT_CHAT_TITLE,
    CanonicalTitleLocked,
    DeclaredScopeIdentity,
    NewSession,
    SessionCountQuery,
    SessionLineageKind,
    SessionListRow,
    SessionQuery,
    SessionWindow,
    TitleConflict,
    TitleSource,
)
from tests.contracts.authority.test_owner_rule_contract import OWNER, RULE, STRANGER
from tests.contracts.conftest import require_capability
from tests.contracts._scopes import MEDIA_ROOT

T0 = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)


def _new(session_id: str, source: str = "cli", **kwargs) -> NewSession:
    return NewSession(session_id=SessionId(session_id), source=source, **kwargs)


def _msg(role: str, content: str, i: int = 0, **kwargs) -> NewMessage:
    return NewMessage(role=role, content=content, timestamp=(T0 + timedelta(seconds=i)).timestamp(), **kwargs)


def _chain(backend, scope, root: str, tip: str, **root_fields) -> None:
    store = backend.open(scope)
    store.sessions.create(_new(root, **root_fields))
    store.sessions.end(SessionId(root), "compression")
    store.sessions.create(_new(tip, parent_session_id=SessionId(root)))


# ── queued prompt rows ────────────────────────────────────────────────────────


def test_retire_undrained_queue_rows_deactivates_marked_user_rows_only(backend, scope_a):
    store_a = backend.open(scope_a)
    sid = SessionId("rq-1")
    store_a.sessions.create(_new("rq-1"))
    queued = {"_queued_prompt": 1}
    store_a.messages.append_batch(sid, [
        _msg("user", "asked", 0),
        _msg("user", "queued one", 1, display_metadata=queued),
        _msg("assistant", "answer", 2, display_metadata=queued),
        _msg("user", "queued two", 3, display_metadata=queued),
    ])
    store_a.sessions.end(sid, "tui_close")
    before = store_a.sessions.get(sid)
    # CONTRACT: the still-active user rows carrying the queued-prompt marker are deactivated and counted; the end stamp is untouched.
    assert store_a.sessions.retire_undrained_queue_rows(sid) == 2
    active = [row.content for row in store_a.messages.read_transcript(sid, TranscriptRead(view=TranscriptView.ACTIVE))]
    assert active == ["asked", "answer"]
    assert store_a.sessions.get(sid).end_reason == "tui_close"
    # CONTRACT: a deactivation keeps the counters where the inserts put them; a second call finds nothing.
    after = store_a.sessions.get(sid)
    assert (after.message_count, after.tool_call_count) == (before.message_count, before.tool_call_count) == (4, 0)
    assert store_a.sessions.retire_undrained_queue_rows(sid) == 0
    assert store_a.sessions.retire_undrained_queue_rows(SessionId("rq-missing")) == 0


def test_reopen_retires_undrained_queue_rows_before_clearing_the_end(backend, scope_a):
    store_a = backend.open(scope_a)
    sid = SessionId("rq-2")
    store_a.sessions.create(_new("rq-2"))
    store_a.messages.append_batch(sid, [_msg("user", "asked", 0), _msg("user", "late", 1, display_metadata={"_queued_prompt": 1})])
    store_a.sessions.end(sid, "tui_close")
    store_a.sessions.reopen(sid)
    # CONTRACT: reopen drops the never-drained prompt and clears the end stamp.
    assert [row.content for row in store_a.messages.read_transcript(sid, TranscriptRead(view=TranscriptView.ACTIVE))] == ["asked"]
    assert store_a.sessions.get(sid).ended_at is None


# ── the canonical Bot Chat title ──────────────────────────────────────────────


def _canonical(backend, scope, sid: str) -> SessionId:
    store = backend.open(scope)
    store.sessions.create(_new(sid, chat_id=f"local-{sid}", user_id="u1"))
    assert store.sessions.set_title(SessionId(sid), CANONICAL_BOT_CHAT_TITLE)
    assert store.sessions.set_hidden(SessionId(sid), True)
    return SessionId(sid)


def test_hidden_canonical_bot_chat_keeps_its_title(backend, scope_a):
    store_a = backend.open(scope_a)
    sid = _canonical(backend, scope_a, "bc-1")
    # CONTRACT: a user rename raises CanonicalTitleLocked, an automatic title is refused, and pinning does not unhide it.
    with pytest.raises(CanonicalTitleLocked):
        store_a.sessions.set_title(sid, "Renamed")
    assert store_a.sessions.set_auto_title(sid, "auto", source=TitleSource.LLM) is False
    assert store_a.sessions.set_pinned(sid, True) is True
    row = store_a.sessions.get(sid)
    assert (row.title, row.hidden, row.pinned) == (CANONICAL_BOT_CHAT_TITLE, True, True)
    # CONTRACT: the canonical title resolves exactly, never to a numbered sibling.
    store_a.sessions.create(_new("bc-numbered"))
    store_a.sessions.set_title(SessionId("bc-numbered"), CANONICAL_BOT_CHAT_TITLE + " #2")
    assert store_a.sessions.resolve_by_title(CANONICAL_BOT_CHAT_TITLE) == sid


def test_retitle_canonical_bot_chat_moves_the_title_off_the_row(backend, scope_a):
    store_a = backend.open(scope_a)
    sid = _canonical(backend, scope_a, "bc-2")
    # CONTRACT: while the row still holds the canonical title, the new title lands with user provenance and the row is unhidden.
    assert store_a.sessions.retitle_canonical_bot_chat(sid, "Old bot chat") is True
    row = store_a.sessions.get(sid)
    assert (row.title, row.title_source, row.hidden) == ("Old bot chat", TitleSource.USER, False)
    # CONTRACT: once the title moved, the compare-and-swap refuses and writes nothing; a missing row refuses too.
    assert store_a.sessions.retitle_canonical_bot_chat(sid, "Again") is False
    assert store_a.sessions.get_title(sid) == "Old bot chat"
    assert store_a.sessions.retitle_canonical_bot_chat(SessionId("bc-missing"), "x") is False


# ── ownership and scope identity ──────────────────────────────────────────────


def test_owns_session_reads_the_owner_stamp(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("own-mine"))
    store_a.sessions.create(_new("own-other", profile_name="someone-else"))
    assert store_a.maintenance.import_sessions([{"id": "own-unset", "source": "cli", "started_at": T0.timestamp(), "messages": []}]).imported == 1
    # CONTRACT: a row is owned when its owner is the scope's profile or unset; another profile's row and a missing row are not.
    assert store_a.sessions.get(SessionId("own-mine")).profile_name == scope_a.name
    assert store_a.sessions.owns_session(SessionId("own-mine")) is True
    assert store_a.sessions.owns_session(SessionId("own-unset")) is True
    assert store_a.sessions.owns_session(SessionId("own-other")) is False
    assert store_a.sessions.owns_session(SessionId("own-missing")) is False


def test_is_kanban_owned_reads_the_resume_lineage(backend, scope_a):
    store_a = backend.open(scope_a)
    _chain(backend, scope_a, "kb-root", "kb-tip", source="kanban")
    store_a.sessions.create(_new("kb-branch", parent_session_id=SessionId("kb-root"), lineage_kind=SessionLineageKind.BRANCH))
    store_a.sessions.create(_new("kb-plain"))
    store_a.sessions.create(_new("kb-retag", cwd="/ws/kb/task"))
    store_a.sessions.retag_worker_sessions("/ws/kb")
    # CONTRACT: a kanban row anywhere on the resume lineage (the session and its compression ancestors) owns it.
    assert store_a.sessions.is_kanban_owned(SessionId("kb-root")) is True
    assert store_a.sessions.is_kanban_owned(SessionId("kb-tip")) is True
    assert store_a.sessions.is_kanban_owned(SessionId("kb-retag")) is True
    # CONTRACT: a fork child stands alone; a plain row and a missing row are not kanban-owned.
    assert store_a.sessions.is_kanban_owned(SessionId("kb-branch")) is False
    assert store_a.sessions.is_kanban_owned(SessionId("kb-plain")) is False
    assert store_a.sessions.is_kanban_owned(SessionId("kb-missing")) is False


def test_declared_scope_identity_names_fork_children(backend, scope_a):
    store_a = backend.open(scope_a)
    _chain(backend, scope_a, "ds-root", "ds-continuation", source=" telegram ")
    for sid, kwargs in (
        ("ds-branch", dict(lineage_kind=SessionLineageKind.BRANCH)),
        ("ds-delegate", dict(lineage_kind=SessionLineageKind.DELEGATE)),
        ("ds-tool", dict(source="tool")),
    ):
        store_a.sessions.create(_new(sid, parent_session_id=SessionId("ds-root"), **kwargs))
    # CONTRACT: branch, delegate and tool children of their parent are fork children; the source is stripped.
    assert store_a.sessions.declared_scope_identity(SessionId("ds-branch")) == DeclaredScopeIdentity(True, "cli")
    assert store_a.sessions.declared_scope_identity(SessionId("ds-delegate")) == DeclaredScopeIdentity(True, "cli")
    assert store_a.sessions.declared_scope_identity(SessionId("ds-tool")) == DeclaredScopeIdentity(True, "tool")
    # CONTRACT: a compression continuation and a root are not; a missing row reads (False, "").
    assert store_a.sessions.declared_scope_identity(SessionId("ds-continuation")) == DeclaredScopeIdentity(False, "cli")
    assert store_a.sessions.declared_scope_identity(SessionId("ds-root")) == DeclaredScopeIdentity(False, "telegram")
    assert store_a.sessions.declared_scope_identity(SessionId("ds-missing")) == DeclaredScopeIdentity(False, "")


# ── titles and config keys ────────────────────────────────────────────────────


def test_title_holder_names_another_session_with_the_exact_title(backend, scope_a, scope_b):
    store_a = backend.open(scope_a)
    store_b = backend.open(scope_b)
    store_a.sessions.create(_new("th-a"))
    store_a.sessions.create(_new("th-b"))
    store_a.sessions.set_title(SessionId("th-a"), "Plan")
    store_b.sessions.create(_new("th-other"))
    store_b.sessions.set_title(SessionId("th-other"), "Elsewhere")
    # CONTRACT: the holder of exactly that title other than the excluded id, else None; other scopes never hold it.
    assert store_a.sessions.title_holder("Plan", excluding=SessionId("th-b")) == "th-a"
    assert store_a.sessions.title_holder("Plan", excluding=SessionId("th-a")) is None
    assert store_a.sessions.title_holder("plan", excluding=SessionId("th-b")) is None
    assert store_a.sessions.title_holder("Elsewhere", excluding=SessionId("th-b")) is None


def test_automatic_titles_rank_below_user_and_unsourced_titles(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.create_exclusive(SessionId("at-unsourced"), source="cli", title="Unsourced")
    store_a.sessions.create(_new("at-free"))
    store_a.sessions.create(_new("at-empty"))
    # CONTRACT: a title without provenance ranks as a user title; an automatic write never replaces it.
    assert store_a.sessions.set_auto_title(SessionId("at-unsourced"), "Auto", source=TitleSource.LLM) is False
    assert store_a.sessions.get_title(SessionId("at-unsourced")) == "Unsourced"
    # CONTRACT: an automatic title held by another session raises TitleConflict naming the holder; an oversized one is a ValueError.
    with pytest.raises(TitleConflict) as conflict:
        store_a.sessions.set_auto_title(SessionId("at-free"), "Unsourced", source=TitleSource.DERIVED)
    assert conflict.value.holder == "at-unsourced"
    with pytest.raises(ValueError):
        store_a.sessions.set_auto_title(SessionId("at-free"), "x" * 101, source=TitleSource.DERIVED)
    assert store_a.sessions.get_title(SessionId("at-free")) is None
    # CONTRACT: an empty automatic title over an untitled row lands as a clear (True, still untitled).
    assert store_a.sessions.set_auto_title(SessionId("at-empty"), "", source=TitleSource.DERIVED) is True
    assert (store_a.sessions.get_title(SessionId("at-empty")), store_a.sessions.get_title_source(SessionId("at-empty"))) == (None, None)


def test_session_ids_with_model_config_key_newest_first(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("ck-1", model_config={"onboarding": {"step": 1}}))
    store_a.sessions.create(_new("ck-2"))
    store_a.sessions.create(_new("ck-3", model_config={"other": 1}))
    store_a.sessions.create(_new("ck-null", model_config={"onboarding": None}))
    store_a.sessions.patch_model_config(SessionId("ck-2"), {"onboarding": "done"})
    # CONTRACT: rows whose config carries a non-null value under the key, newest started first.
    assert store_a.sessions.session_ids_with_model_config_key("onboarding") == ("ck-2", "ck-1")
    assert store_a.sessions.session_ids_with_model_config_key("absent") == ()


# ── empty candidates and picker rows ──────────────────────────────────────────


def test_is_empty_candidate_is_untitled_messageless_and_childless(backend, scope_a):
    store_a = backend.open(scope_a)
    for sid in ("ec-empty", "ec-titled", "ec-talked", "ec-parent", "ec-leased"):
        store_a.sessions.create(_new(sid))
    store_a.sessions.set_title(SessionId("ec-titled"), "named")
    store_a.messages.append(SessionId("ec-talked"), _msg("user", "hi"))
    store_a.sessions.create(_new("ec-child", parent_session_id=SessionId("ec-parent")))
    # CONTRACT: untitled, no messages, no children; a title, a message or a child disqualifies; a missing row is no candidate.
    assert store_a.sessions.is_empty_candidate(SessionId("ec-empty")) is True
    assert store_a.sessions.is_empty_candidate(SessionId("ec-titled")) is False
    assert store_a.sessions.is_empty_candidate(SessionId("ec-talked")) is False
    assert store_a.sessions.is_empty_candidate(SessionId("ec-parent")) is False
    assert store_a.sessions.is_empty_candidate(SessionId("ec-missing")) is False
    # CONTRACT: unlike delete_if_empty, the probe does not consult the lease guard.
    require_capability(backend, "transactional_claims")
    assert store_a.coordination.try_acquire_turn_lease(session_id=SessionId("ec-leased"), holder="live", ttl_seconds=300.0)
    assert store_a.sessions.is_empty_candidate(SessionId("ec-leased")) is True
    assert store_a.sessions.delete_if_empty(SessionId("ec-leased"), media_root=MEDIA_ROOT) == ()
    assert store_a.sessions.get(SessionId("ec-leased")) is not None


def test_list_row_is_the_picker_columns_of_the_row(backend, scope_a):
    store_a = backend.open(scope_a)
    sid = SessionId("lr-1")
    store_a.sessions.create(_new("lr-1", source="telegram", model="m1"))
    store_a.sessions.set_title(sid, "Picker")
    store_a.messages.append_batch(sid, [_msg("user", "q", 0), _msg("assistant", "a", 1)])
    store_a.sessions.set_hidden(sid, True)
    store_a.sessions.end(sid, "cli_close")
    row = store_a.sessions.get(sid)
    # CONTRACT: the picker columns read what the full row reads, hidden and ended rows included; a missing row reads None.
    assert store_a.sessions.list_row(sid) == SessionListRow(
        session_id=sid, source="telegram", title="Picker", started_at=row.started_at, ended_at=row.ended_at,
        message_count=2, model="m1", hidden=True, archived=False,
    )
    assert store_a.sessions.list_row(SessionId("lr-missing")) is None


# ── the live listing ──────────────────────────────────────────────────────────


def test_live_listing_facts_rows_and_pending_admissions(backend, scope_a):
    store_a = backend.open(scope_a)
    require_capability(backend, "durable_admissions")
    for sid in ("ll-idle", "ll-busy"):
        store_a.sessions.create(_new(sid, model="m"))
    epoch = backend.begin_epoch(scope_a, "instance-contract")
    first = store_a.admissions.admit(epoch=epoch, principal_id="p", session_id=SessionId("ll-busy"), request_id="r1", payload={"input": "a"})[0]
    second = store_a.admissions.admit(epoch=epoch, principal_id="p", session_id=SessionId("ll-busy"), request_id="r2", payload={"input": "b"})[0]
    facts = store_a.sessions.live_listing_facts([SessionId("ll-idle"), SessionId("ll-busy"), SessionId("ll-missing")])
    # CONTRACT: rows for the ids that exist only, with the listing columns and the runtime counters.
    assert set(facts.rows) == {"ll-idle", "ll-busy"}
    idle = facts.rows[SessionId("ll-idle")]
    record = store_a.sessions.get(SessionId("ll-idle"))
    assert (idle.id, idle.source, idle.model, idle.started_at, idle.ended_at, idle.message_count) == ("ll-idle", "cli", "m", record.started_at, None, 0)
    assert (idle.runtime_revision, idle.runtime_generation, idle.target_ended_at) == (record.runtime_revision, record.runtime_generation, None)
    # CONTRACT: pending carries each id's queued, started or unknown admissions in admission order.
    assert set(facts.pending) == {"ll-busy"}
    refs = facts.pending[SessionId("ll-busy")]
    assert [(ref.admission_id, ref.status) for ref in refs] == [(first.admission_id, "queued"), (second.admission_id, "queued")]
    assert refs[0].seq < refs[1].seq
    claimed = store_a.admissions.claim(epoch=epoch, session_id=SessionId("ll-busy"))
    assert claimed is not None
    refs = store_a.sessions.live_listing_facts([SessionId("ll-busy")]).pending[SessionId("ll-busy")]
    assert [ref.status for ref in refs] == ["started", "queued"]


# ── store-wide firsts ─────────────────────────────────────────────────────────


def test_first_session_started_at_and_longest_session(backend, scope_a, scope_b):
    # CONTRACT: an empty store has neither.
    store_a = backend.open(scope_a)
    store_b = backend.open(scope_b)
    assert store_a.sessions.first_session_started_at() is None
    assert store_a.sessions.longest_session_id() is None
    for sid in ("fl-1", "fl-2", "fl-3"):
        store_a.sessions.create(_new(sid))
    store_b.sessions.create(_new("fl-b"))
    # CONTRACT: with no transcript rows anywhere, the query still names a session of the scope, one with zero messages.
    assert store_a.sessions.longest_session_id() in {"fl-1", "fl-2", "fl-3"}
    store_a.messages.append_batch(SessionId("fl-2"), [_msg("user", "a", 0), _msg("assistant", "b", 1)])
    rows = [store_a.messages.append(SessionId("fl-3"), _msg("user", str(i), i)) for i in range(3)]
    store_b.messages.append_batch(SessionId("fl-b"), [_msg("user", str(i), i) for i in range(9)])
    # CONTRACT: the earliest started_at of the scope; the session with the most transcript rows of any visibility.
    assert store_a.sessions.first_session_started_at() == min(store_a.sessions.get(SessionId(s)).started_at for s in ("fl-1", "fl-2", "fl-3"))
    assert store_a.sessions.longest_session_id() == "fl-3"
    assert store_a.messages.deactivate_message(SessionId("fl-3"), rows[1]) is True
    assert store_a.sessions.longest_session_id() == "fl-3"


# ── the Bots roster ───────────────────────────────────────────────────────────


def test_roster_reads_the_canonical_chat_its_tip_and_the_freshest_worker(backend, scope_a):
    store_a = backend.open(scope_a)
    _canonical(backend, scope_a, "ro-bot")
    store_a.sessions.end(SessionId("ro-bot"), "compression")
    store_a.sessions.create(_new("ro-bot-tip", parent_session_id=SessionId("ro-bot")))
    store_a.sessions.set_title(SessionId("ro-bot-tip"), "Tip title")
    long_line = "word " * 30
    store_a.messages.append_batch(SessionId("ro-bot-tip"), [_msg("user", "hi", 0), _msg("assistant", long_line, 1), _msg("tool", "", 2)])
    store_a.sessions.create(_new("ro-worker-old", source="kanban"))
    store_a.sessions.create(_new("ro-worker", source="tool"))
    store_a.sessions.set_title(SessionId("ro-worker"), "Job")
    store_a.sessions.create(_new("ro-worker-hidden", source="kanban"))
    store_a.sessions.set_hidden(SessionId("ro-worker-hidden"), True)
    roster = store_a.sessions.roster_sessions(owner_user_id="u1", owner=None)
    canonical = roster.canonical_session
    # CONTRACT: the canonical Bot Chat is read through its compression tip: the tip's title and counters, the root's title, the tip's latest dialogue line cut at 80 characters.
    assert (canonical.id, canonical.resolved_id, canonical.title, canonical.root_title) == ("ro-bot", "ro-bot-tip", "Tip title", CANONICAL_BOT_CHAT_TITLE)
    assert canonical.preview == " ".join(long_line.split())[:80] + "..." and canonical.message_count == 3
    # CONTRACT: the freshest visible worker, any owner.
    assert (roster.worker_session.id, roster.worker_session.source, roster.worker_session.title) == ("ro-worker", "tool", "Job")
    # CONTRACT: another owner sees no canonical chat; the operator (None) sees it; an archived canonical chat is gone.
    assert store_a.sessions.roster_sessions(owner_user_id="u2", owner=None).canonical_session is None
    assert store_a.sessions.roster_sessions(owner_user_id=None, owner=None).canonical_session.id == "ro-bot"
    store_a.sessions.set_archived(SessionId("ro-bot"), True)
    assert store_a.sessions.roster_sessions(owner_user_id="u1", owner=None).canonical_session is None


def test_roster_reads_the_canonical_chats_owner_through_the_owner_rule(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("ro-login", chat_id="local-ro-login", user_id="basic:admin"))
    assert store_a.sessions.set_title(SessionId("ro-login"), CANONICAL_BOT_CHAT_TITLE)
    # CONTRACT: the canonical Bot Chat is the reader's when its stored owner belongs to the reader under the owner rule (a sign-in stamp is the host owner's); the newest local chat still matches the stored owner exactly; without a rule only the equal stamp does.
    roster = store_a.sessions.roster_sessions(owner_user_id=OWNER, owner=RULE)
    assert roster.canonical_session.id == "ro-login" and roster.last_session is None
    assert store_a.sessions.roster_sessions(owner_user_id=STRANGER, owner=RULE).canonical_session is None
    assert store_a.sessions.roster_sessions(owner_user_id=OWNER, owner=None).canonical_session is None


# ── the bounded search ────────────────────────────────────────────────────────


def _local_row(backend, scope, sid: str, surface: str) -> None:
    """A ``source='local'`` row whose creating surface is ``surface``: a later create fills the unset creation provenance."""
    store = backend.open(scope)
    store.sessions.create(_new(sid, source="local"))
    store.sessions.create(_new(sid, source=surface))


def test_local_rows_match_sources_by_their_creating_surface(backend, scope_a):
    store_a = backend.open(scope_a)
    _local_row(backend, scope_a, "lc-desktop", "desktop")
    _local_row(backend, scope_a, "lc-tui", "tui")
    store_a.sessions.create(_new("lc-cli"))
    row = store_a.sessions.get(SessionId("lc-desktop"))
    assert (row.source, row.created_source) == ("local", "desktop")

    def search(**filters):
        return {e.session.session_id for e in store_a.sessions.search_sessions(**filters)}

    # CONTRACT: with local_created_sources a local row matches exactly when its creating surface is listed, even when "local" is in sources.
    assert search(sources=("cli",), local_created_sources=("desktop",)) == {"lc-cli", "lc-desktop"}
    assert search(sources=("local",), local_created_sources=("desktop",)) == {"lc-desktop"}
    assert search(sources=("local",)) == {"lc-desktop", "lc-tui"}
    listed = store_a.sessions.list_sessions(SessionQuery(sources=("cli",), local_created_sources=("tui",)), SessionWindow(limit=10))
    assert {e.session.session_id for e in listed} == {"lc-cli", "lc-tui"}
    assert store_a.sessions.count_sessions(SessionCountQuery(sources=("local",), local_created_sources=("tui",))) == 1


def test_search_sessions_skips_launch_stubs_on_request(backend, scope_a):
    store_a = backend.open(scope_a)
    for sid in ("st-stub", "st-talked", "st-admitted"):
        store_a.sessions.create(_new(sid))
    store_a.messages.append(SessionId("st-talked"), _msg("user", "hi"))
    store_a.sessions.seed_ledger_admission(SessionId("st-admitted"), status="queued")
    every = [e.session.session_id for e in store_a.sessions.search_sessions()]
    # CONTRACT: every row by default, most recently active first; skip_launch_stubs drops rows with neither messages nor an admission.
    assert set(every) == {"st-stub", "st-talked", "st-admitted"}
    assert {e.session.session_id for e in store_a.sessions.search_sessions(skip_launch_stubs=True)} == {"st-talked", "st-admitted"}
    assert len(store_a.sessions.search_sessions(window=SessionWindow(limit=1, offset=1))) == 1


def test_search_by_id_ranks_exact_then_prefix_then_substring(backend, scope_a):
    store_a = backend.open(scope_a)
    for sid in ("x-abc", "abc-1", "abc"):
        store_a.sessions.create(_new(sid))
    _chain(backend, scope_a, "abcroot", "zz-tip")
    # CONTRACT: exact matches first, then prefixes, then substrings; a compression root matches through its chain and lists as its tip.
    ids = [e.session.session_id for e in store_a.sessions.search_by_id("ABC")]
    assert ids[0] == "abc" and set(ids[1:3]) == {"abc-1", "zz-tip"} and ids[3] == "x-abc"
    assert [e.session.session_id for e in store_a.sessions.search_by_id("abc", limit=1)] == ["abc"]
    # CONTRACT: blank text or a non-positive limit lists nothing.
    assert store_a.sessions.search_by_id("  ") == ()
    assert store_a.sessions.search_by_id("abc", limit=0) == ()


# ── deletes over compression chains ───────────────────────────────────────────


def test_delete_with_the_compression_chain_takes_every_segment(backend, scope_a):
    store_a = backend.open(scope_a)
    _chain(backend, scope_a, "dc-root", "dc-tip")
    store_a.sessions.create(_new("dc-other"))
    # CONTRACT: include_compression_chain removes the whole chain, whichever segment is named.
    assert set(store_a.sessions.delete(SessionId("dc-tip"), include_compression_chain=True, media_root=MEDIA_ROOT)) == {"dc-root", "dc-tip"}
    assert store_a.sessions.get(SessionId("dc-root")) is None and store_a.sessions.get(SessionId("dc-other")) is not None


def test_delete_with_the_compression_chain_keeps_a_guarded_chain(backend, scope_a):
    store_a = backend.open(scope_a)
    require_capability(backend, "transactional_claims")
    _chain(backend, scope_a, "dg-root", "dg-tip")
    assert store_a.coordination.try_acquire_turn_lease(session_id=SessionId("dg-tip"), holder="live", ttl_seconds=300.0)
    # CONTRACT: the chain variant takes the bulk path: a guarded chain is kept and the result is empty instead of a refusal.
    assert store_a.sessions.delete(SessionId("dg-root"), refuse_guarded=True, include_compression_chain=True, media_root=MEDIA_ROOT) == ()
    assert store_a.sessions.get(SessionId("dg-root")) is not None and store_a.sessions.get(SessionId("dg-tip")) is not None


# ── working directory ─────────────────────────────────────────────────────────


def test_update_cwd_same_cwd_overwrites_only_the_given_git_field(backend, scope_a):
    store_a = backend.open(scope_a)
    sid = SessionId("cw-same")
    store_a.sessions.create(_new("cw-same"))
    store_a.sessions.update_cwd(sid, "/w", git_branch="main", git_repo_root="/w")
    store_a.sessions.update_cwd(sid, "/w", git_branch="dev")
    row = store_a.sessions.get(sid)
    # CONTRACT: on the same cwd a captured value overwrites its field and the other keeps its stored value.
    assert (row.git_branch, row.git_repo_root) == ("dev", "/w")
    assert store_a.sessions.update_cwd(sid, "") is None
