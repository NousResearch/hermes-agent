"""Contract cases for ``LineageRepository``.

Every case runs against every registered backend. Fixtures build lineage shapes through the sessions and messages contract verbs, with lineage publication, coordination leases and direct backdating in the compression-write cases below: a compression continuation is ``end(parent, "compression")`` followed by ``create(child, parent_session_id=parent)``; a fork is ``create(..., lineage_kind=BRANCH|DELEGATE)`` or ``source="tool"``; a reset continuation is ``promote_to_reset(parent)`` followed by a child on the same ``session_key``.
"""

from __future__ import annotations

from itertools import pairwise

import time
from datetime import datetime, timedelta, UTC
from typing import Optional

import pytest

from persistence import CompressionBusy, CompressionPublishRefused, Conflict, MessageId, RowVisibility, SessionId
from persistence.conversation.lineage import CompressionChild, LineageRepository
from persistence.conversation.messages import NewMessage, RowDisposition, TranscriptRead, TranscriptView
from persistence.conversation.sessions import AUTOMATIC_END_REASONS, END_REASON_ORPHANED_COMPRESSION, NewSession, SessionLineageKind
from persistence.conversation.usage import UsageDelta
from tests.contracts._concurrency import run_concurrently
from tests.contracts._waiting import wait_past

# The age past which an undriven continuation counts as orphaned (the protocol's seven days).
_ORPHAN_AGE = timedelta(days=7)
_T0 = datetime(2026, 9, 11, 10, tzinfo=UTC).timestamp()


def _new(session_id: str, source: str = "cli", **kwargs) -> NewSession:
    return NewSession(session_id=SessionId(session_id), source=source, **kwargs)


def _msg(content: str, i: int = 0) -> NewMessage:
    return NewMessage(role="user", content=content, timestamp=_T0 + i)


def _continue(backend, scope, parent: str, child: str, **kwargs) -> None:
    """End ``parent`` by compression and create ``child`` as its continuation."""
    store = backend.open(scope)
    store.sessions.end(SessionId(parent), "compression")
    store.sessions.create(_new(child, parent_session_id=SessionId(parent), **kwargs))


def _chain(backend, scope, *ids: str) -> None:
    """A straight compression chain ``ids[0] -> ids[1] -> ...``."""
    store = backend.open(scope)
    store.sessions.create(_new(ids[0]))
    for parent, child in pairwise(ids):
        _continue(backend, scope, parent, child)


# ── protocol + input shape ────────────────────────────────────────────────────


def test_repository_satisfies_protocol(backend, scope_a):
    # CONTRACT: the assembled attribute presents every verb of the family.
    assert isinstance(backend.open(scope_a).lineage, LineageRepository)


def test_empty_id_projects_like_an_unknown_id(backend, scope_a):
    repo, empty = backend.open(scope_a).lineage, SessionId("")
    # CONTRACT: an empty id is read like any unrecorded session: it projects to itself, its lineage is empty, it is no fork and nothing is published above it.
    assert (repo.tip(empty), repo.conversation_root(empty), repo.resolve_resume_target(empty)) == (empty, empty, empty)
    assert (repo.lineage(empty), repo.chain(empty), repo.ancestor_display_prefix(empty)) == ((), (), ())
    assert repo.is_fork_child(empty) is False


# ── unknown ids project to themselves ─────────────────────────────────────────


def test_unknown_id_projects_to_itself(backend, scope_a):
    store_a = backend.open(scope_a)
    ghost = SessionId("lin-ghost")
    # CONTRACT: the projections of a session nobody recorded are the session itself; lineage is empty; not a fork.
    assert store_a.lineage.tip(ghost) == ghost
    assert store_a.lineage.conversation_root(ghost) == ghost
    assert store_a.lineage.resolve_resume_target(ghost) == ghost
    assert store_a.lineage.lineage(ghost) == ()
    assert store_a.lineage.is_fork_child(ghost) is False


def test_session_without_continuation_is_its_own_tip_root_and_lineage(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("lin-solo"))
    solo = SessionId("lin-solo")
    # CONTRACT: no children means every projection is the session itself and the lineage is the singleton.
    assert store_a.lineage.tip(solo) == solo
    assert store_a.lineage.conversation_root(solo) == solo
    assert store_a.lineage.lineage(solo) == (solo,)
    assert store_a.lineage.resolve_resume_target(solo) == solo


# ── tip ───────────────────────────────────────────────────────────────────────


def test_tip_follows_the_compression_chain_from_any_segment(backend, scope_a):
    store_a = backend.open(scope_a)
    _chain(backend, scope_a, "lin-t-root", "lin-t-mid", "lin-t-tip")
    tip = SessionId("lin-t-tip")
    # CONTRACT: the tip is the last segment of the chain from wherever the walk starts; the tip is its own tip.
    assert store_a.lineage.tip(SessionId("lin-t-root")) == tip
    assert store_a.lineage.tip(SessionId("lin-t-mid")) == tip
    assert store_a.lineage.tip(tip) == tip


def test_tip_is_visibility_blind(backend, scope_a):
    store_a = backend.open(scope_a)
    _chain(backend, scope_a, "lin-v-root", "lin-v-tip")
    assert store_a.sessions.set_hidden(SessionId("lin-v-tip"), True) is True
    # CONTRACT: a hidden tip is still the tip (visibility is not lineage).
    assert store_a.lineage.tip(SessionId("lin-v-root")) == SessionId("lin-v-tip")
    assert store_a.sessions.set_archived(SessionId("lin-v-tip"), True).changed is True
    # CONTRACT: an archived tip is still the tip.
    assert store_a.lineage.tip(SessionId("lin-v-root")) == SessionId("lin-v-tip")


def test_tip_never_steps_into_a_fork_or_tool_child(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("lin-f-root"))
    store_a.sessions.end(SessionId("lin-f-root"), "compression")
    root = SessionId("lin-f-root")
    store_a.sessions.create(_new("lin-f-branch", parent_session_id=root, lineage_kind=SessionLineageKind.BRANCH))
    store_a.sessions.create(_new("lin-f-delegate", parent_session_id=root, lineage_kind=SessionLineageKind.DELEGATE))
    store_a.sessions.create(_new("lin-f-tool", source="tool", parent_session_id=root))
    # CONTRACT: branch, delegate and tool children of a compressed parent are not continuations; the parent stays the tip.
    assert store_a.lineage.tip(root) == root
    store_a.sessions.create(_new("lin-f-cont", parent_session_id=root))
    # CONTRACT: the plain child is the continuation even though it was created after the forks.
    assert store_a.lineage.tip(root) == SessionId("lin-f-cont")


def test_tip_requires_the_parent_to_have_compressed(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("lin-p-root"))
    store_a.sessions.create(_new("lin-p-kid", parent_session_id=SessionId("lin-p-root")))
    # CONTRACT: a plain child of a live parent is not a continuation.
    assert store_a.lineage.tip(SessionId("lin-p-root")) == SessionId("lin-p-root")
    store_a.sessions.end(SessionId("lin-p-root"), "idle")
    # CONTRACT: nor is a child of a parent that ended for any reason other than compression.
    assert store_a.lineage.tip(SessionId("lin-p-root")) == SessionId("lin-p-root")
    store_a.sessions.reopen(SessionId("lin-p-root"))
    store_a.sessions.end(SessionId("lin-p-root"), "compression")
    # CONTRACT: continuation is read at query time: once the parent's compression end lands, the pre-existing child projects.
    assert store_a.lineage.tip(SessionId("lin-p-root")) == SessionId("lin-p-kid")


def test_tip_prefers_a_child_that_compressed_then_a_live_child_over_a_stale_ended_sibling(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("lin-r-root"))
    _continue(backend, scope_a, "lin-r-root", "lin-r-stale")
    store_a.sessions.end(SessionId("lin-r-stale"), "ws_orphan_reap")
    store_a.sessions.create(_new("lin-r-live", parent_session_id=SessionId("lin-r-root")))
    # CONTRACT: a still-open continuation beats an ended sibling that did not compress, whatever their order.
    assert store_a.lineage.tip(SessionId("lin-r-root")) == SessionId("lin-r-live")
    store_a.sessions.create(_new("lin-r-older", parent_session_id=SessionId("lin-r-root")))
    store_a.sessions.end(SessionId("lin-r-older"), "compression")
    store_a.sessions.create(_new("lin-r-older-next", parent_session_id=SessionId("lin-r-older")))
    # CONTRACT: a continuation that itself compressed beats a live sibling, and the walk continues through it.
    assert store_a.lineage.tip(SessionId("lin-r-root")) == SessionId("lin-r-older-next")


def test_tip_among_equal_candidates_picks_the_most_recently_active(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("lin-a-root"))
    _continue(backend, scope_a, "lin-a-root", "lin-a-first")
    store_a.sessions.create(_new("lin-a-second", parent_session_id=SessionId("lin-a-root")))
    # CONTRACT: two open continuations without transcript rows: the newer start wins.
    assert store_a.lineage.tip(SessionId("lin-a-root")) == SessionId("lin-a-second")
    store_a.messages.append(SessionId("lin-a-first"), NewMessage(
        role="user", content="late", timestamp=datetime(2100, 1, 1, tzinfo=UTC).timestamp()))
    # CONTRACT: a newer transcript row makes the older-started sibling the most recently active, and the tip.
    assert store_a.lineage.tip(SessionId("lin-a-root")) == SessionId("lin-a-first")


# ── chain ─────────────────────────────────────────────────────────────────────


def test_chain_walks_forward_from_the_segment_to_the_tip(backend, scope_a):
    store_a = backend.open(scope_a)
    _chain(backend, scope_a, "lin-ch-root", "lin-ch-mid", "lin-ch-tip")
    root, mid, tip = SessionId("lin-ch-root"), SessionId("lin-ch-mid"), SessionId("lin-ch-tip")
    # CONTRACT: the walk starts at the session itself (no ancestors) and its last element is the tip.
    assert store_a.lineage.chain(root) == (root, mid, tip)
    assert store_a.lineage.chain(mid) == (mid, tip)
    assert store_a.lineage.chain(tip) == (tip,)
    # CONTRACT: an unknown id is its own one-element chain.
    assert store_a.lineage.chain(SessionId("lin-ch-ghost")) == (SessionId("lin-ch-ghost"),)


def test_chain_follows_the_preferred_continuation_where_lineage_takes_the_oldest(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("lin-cp-root"))
    _continue(backend, scope_a, "lin-cp-root", "lin-cp-first")
    store_a.sessions.create(_new("lin-cp-second", parent_session_id=SessionId("lin-cp-root")))
    root = SessionId("lin-cp-root")
    # CONTRACT: two open continuations: the chain steps into the most recently active one, as tip does, while lineage descends through the oldest.
    assert store_a.lineage.chain(root) == (root, SessionId("lin-cp-second"))
    assert store_a.lineage.chain(root)[-1] == store_a.lineage.tip(root)
    assert store_a.lineage.lineage(root) == (root, SessionId("lin-cp-first"))


# ── lineage ───────────────────────────────────────────────────────────────────


def test_lineage_is_root_first_chain_order_from_any_segment(backend, scope_a):
    store_a = backend.open(scope_a)
    _chain(backend, scope_a, "lin-l-root", "lin-l-mid", "lin-l-tip")
    expected = (SessionId("lin-l-root"), SessionId("lin-l-mid"), SessionId("lin-l-tip"))
    # CONTRACT: the same chain, root first, whichever segment is asked.
    for sid in expected:
        assert store_a.lineage.lineage(sid) == expected


def test_lineage_of_a_fork_child_is_itself_and_a_fork_starts_its_own_lineage(backend, scope_a):
    store_a = backend.open(scope_a)
    _chain(backend, scope_a, "lin-lf-root", "lin-lf-tip")
    store_a.sessions.create(_new(
        "lin-lf-branch", parent_session_id=SessionId("lin-lf-tip"), lineage_kind=SessionLineageKind.BRANCH))
    # CONTRACT: a fork child's lineage is the singleton; the parent chain does not include it.
    assert store_a.lineage.lineage(SessionId("lin-lf-branch")) == (SessionId("lin-lf-branch"),)
    assert store_a.lineage.lineage(SessionId("lin-lf-root")) == (SessionId("lin-lf-root"), SessionId("lin-lf-tip"))
    assert store_a.lineage.is_fork_child(SessionId("lin-lf-branch")) is True


def test_lineage_off_chain_segment_reads_itself(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("lin-o-root"))
    _continue(backend, scope_a, "lin-o-root", "lin-o-first")
    store_a.sessions.create(_new("lin-o-second", parent_session_id=SessionId("lin-o-root")))
    # CONTRACT: the descent takes the OLDEST continuation; the newer sibling is not on the computed chain and reads itself.
    assert store_a.lineage.lineage(SessionId("lin-o-root")) == (SessionId("lin-o-root"), SessionId("lin-o-first"))
    assert store_a.lineage.lineage(SessionId("lin-o-second")) == (SessionId("lin-o-second"),)


# ── conversation_root ─────────────────────────────────────────────────────────


def test_conversation_root_crosses_every_edge_kind(backend, scope_a):
    store_a = backend.open(scope_a)
    _chain(backend, scope_a, "lin-c-root", "lin-c-tip")
    store_a.sessions.create(_new(
        "lin-c-delegate", source="tool", parent_session_id=SessionId("lin-c-tip"), lineage_kind=SessionLineageKind.DELEGATE))
    store_a.sessions.create(_new(
        "lin-c-branch", parent_session_id=SessionId("lin-c-delegate"), lineage_kind=SessionLineageKind.BRANCH))
    root = SessionId("lin-c-root")
    # CONTRACT: compression, delegate and branch edges all lead to the same stable root.
    for sid in ("lin-c-tip", "lin-c-delegate", "lin-c-branch"):
        assert store_a.lineage.conversation_root(SessionId(sid)) == root
    # CONTRACT: the root is unchanged when there is no recorded parent.
    assert store_a.lineage.conversation_root(root) == root


# ── is_fork_child ─────────────────────────────────────────────────────────────


def test_is_fork_child_classifies_intent_not_edge_presence(backend, scope_a):
    store_a = backend.open(scope_a)
    _chain(backend, scope_a, "lin-k-root", "lin-k-cont")
    store_a.sessions.create(_new(
        "lin-k-branch", parent_session_id=SessionId("lin-k-cont"), lineage_kind=SessionLineageKind.BRANCH))
    store_a.sessions.create(_new(
        "lin-k-delegate", parent_session_id=SessionId("lin-k-cont"), lineage_kind=SessionLineageKind.DELEGATE))
    store_a.sessions.create(_new("lin-k-tool", source="tool", parent_session_id=SessionId("lin-k-cont")))
    store_a.sessions.create(_new("lin-k-plain", parent_session_id=SessionId("lin-k-cont")))
    repo = backend.open(scope_a).lineage
    # CONTRACT: branch, delegate and tool-sourced children are forks.
    assert repo.is_fork_child(SessionId("lin-k-branch")) is True
    assert repo.is_fork_child(SessionId("lin-k-delegate")) is True
    assert repo.is_fork_child(SessionId("lin-k-tool")) is True
    # CONTRACT: a compression continuation, a plain child and a root are not.
    assert repo.is_fork_child(SessionId("lin-k-cont")) is False
    assert repo.is_fork_child(SessionId("lin-k-plain")) is False
    assert repo.is_fork_child(SessionId("lin-k-root")) is False


def test_continuation_inheriting_parent_config_is_not_a_fork(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("lin-i-root"))
    store_a.sessions.create(_new(
        "lin-i-delegate", parent_session_id=SessionId("lin-i-root"), lineage_kind=SessionLineageKind.DELEGATE,
        model_config={"model": "m"}))
    delegate = store_a.sessions.get(SessionId("lin-i-delegate"))
    store_a.sessions.end(SessionId("lin-i-delegate"), "compression")
    store_a.sessions.create(_new(
        "lin-i-cont", parent_session_id=SessionId("lin-i-delegate"), model_config=dict(delegate.model_config)))
    # CONTRACT: a copied delegate marker does not make a branch, but excludes the child from the compression chain.
    assert store_a.lineage.is_fork_child(SessionId("lin-i-cont")) is False
    assert store_a.lineage.tip(SessionId("lin-i-delegate")) == SessionId("lin-i-delegate")


# ── resolve_resume_target ─────────────────────────────────────────────────────


def test_resume_target_is_the_deepest_descendant_holding_rows(backend, scope_a):
    store_a = backend.open(scope_a)
    _chain(backend, scope_a, "lin-rs-root", "lin-rs-tip")
    store_a.messages.append(SessionId("lin-rs-tip"), _msg("in the tip"))
    # CONTRACT: resuming the compacted root lands on the tip that holds the messages.
    assert store_a.lineage.resolve_resume_target(SessionId("lin-rs-root")) == SessionId("lin-rs-tip")
    store_a.sessions.create(_new("lin-rs-deeper", parent_session_id=SessionId("lin-rs-tip")))
    # CONTRACT: an empty plain child does not move the target...
    assert store_a.lineage.resolve_resume_target(SessionId("lin-rs-root")) == SessionId("lin-rs-tip")
    store_a.messages.append(SessionId("lin-rs-deeper"), _msg("newer"))
    # CONTRACT: ...but once it holds a row the DEEPEST holder wins, past a parent that was never compressed.
    assert store_a.lineage.resolve_resume_target(SessionId("lin-rs-root")) == SessionId("lin-rs-deeper")


def test_resume_target_counts_rows_of_every_visibility(backend, scope_a):
    store_a = backend.open(scope_a)
    _chain(backend, scope_a, "lin-rv-root", "lin-rv-tip")
    store_a.messages.append(SessionId("lin-rv-tip"), _msg("only row"))
    assert (store_a.messages.compact(SessionId("lin-rv-tip"), [])).active_count == 0
    records = store_a.messages.read_transcript(SessionId("lin-rv-tip"), TranscriptRead(view=TranscriptView.ALL))
    assert {r.visibility for r in records} == {RowVisibility.COMPACTION_ARCHIVED}
    assert store_a.messages.read_transcript(SessionId("lin-rv-tip")) == ()
    # CONTRACT: a session whose rows are all archived by compaction still "holds rows" for the resume walk.
    assert store_a.lineage.resolve_resume_target(SessionId("lin-rv-root")) == SessionId("lin-rv-tip")


def test_resume_target_never_enters_forks_or_reset_continuations(backend, scope_a):
    store_a = backend.open(scope_a)
    _chain(backend, scope_a, "lin-rf-root", "lin-rf-tip")
    store_a.messages.append(SessionId("lin-rf-tip"), _msg("tip"))
    tip = SessionId("lin-rf-tip")
    store_a.sessions.create(_new("lin-rf-branch", parent_session_id=tip, lineage_kind=SessionLineageKind.BRANCH))
    store_a.messages.append(SessionId("lin-rf-branch"), _msg("branch"))
    store_a.sessions.create(_new("lin-rf-tool", source="tool", parent_session_id=tip))
    store_a.messages.append(SessionId("lin-rf-tool"), _msg("tool"))
    # CONTRACT: branch and tool children never hijack the resume target.
    assert store_a.lineage.resolve_resume_target(SessionId("lin-rf-root")) == tip
    store_a.sessions.create(_new("lin-rt-parent", session_key="key-1"))
    store_a.messages.append(SessionId("lin-rt-parent"), _msg("before reset"))
    assert store_a.sessions.promote_to_reset(SessionId("lin-rt-parent")) is True
    store_a.sessions.create(_new("lin-rt-child", session_key="key-1", parent_session_id=SessionId("lin-rt-parent")))
    store_a.messages.append(SessionId("lin-rt-child"), _msg("after reset"))
    # CONTRACT: a reset continuation on the same routing key is a new conversation; the resume stays on the parent.
    assert store_a.lineage.resolve_resume_target(SessionId("lin-rt-parent")) == SessionId("lin-rt-parent")


def test_resume_target_honours_every_conversation_boundary_reason(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("lin-ri-parent", session_key="key-2"))
    store_a.messages.append(SessionId("lin-ri-parent"), _msg("before idle"))
    store_a.sessions.end(SessionId("lin-ri-parent"), "idle")
    store_a.sessions.create(_new("lin-ri-child", session_key="key-2", parent_session_id=SessionId("lin-ri-parent")))
    store_a.messages.append(SessionId("lin-ri-child"), _msg("after idle"))
    # CONTRACT: an idle boundary is a conversation boundary like a reset: the same-key child is not entered.
    assert store_a.lineage.resolve_resume_target(SessionId("lin-ri-parent")) == SessionId("lin-ri-parent")
    store_a.sessions.create(_new("lin-ri-other", session_key="key-3", parent_session_id=SessionId("lin-ri-parent")))
    store_a.messages.append(SessionId("lin-ri-other"), _msg("other key"))
    # CONTRACT: a child on a DIFFERENT key after the boundary is a plain continuation and is entered.
    assert store_a.lineage.resolve_resume_target(SessionId("lin-ri-parent")) == SessionId("lin-ri-other")


def test_resume_target_without_any_rows_is_the_tip(backend, scope_a):
    store_a = backend.open(scope_a)
    _chain(backend, scope_a, "lin-re-root", "lin-re-tip")
    # CONTRACT: nothing on the path holds a row: the tip itself (the input when nothing continued it).
    assert store_a.lineage.resolve_resume_target(SessionId("lin-re-root")) == SessionId("lin-re-tip")
    assert store_a.lineage.resolve_resume_target(SessionId("lin-re-tip")) == SessionId("lin-re-tip")


# ── scope isolation ───────────────────────────────────────────────────────────


def test_projections_never_cross_scopes(backend, scope_a, scope_b):
    store_a = backend.open(scope_a)
    store_b = backend.open(scope_b)
    _chain(backend, scope_a, "lin-s-root", "lin-s-tip")
    store_b.sessions.create(_new("lin-s-root"))
    root = SessionId("lin-s-root")
    # CONTRACT: the same id in another scope has its own singleton lineage.
    assert store_b.lineage.tip(root) == root
    assert store_b.lineage.lineage(root) == (root,)
    assert store_a.lineage.lineage(root) == (root, SessionId("lin-s-tip"))


# ── ancestor_display_prefix ───────────────────────────────────────────────────


def test_ancestor_display_prefix_is_the_ancestors_display_rows_root_first(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("lin-dp-root"))
    store_a.messages.append(SessionId("lin-dp-root"), _msg("root 1", 0))
    store_a.messages.append(SessionId("lin-dp-root"), _msg("root 2", 1))
    _continue(backend, scope_a, "lin-dp-root", "lin-dp-mid")
    store_a.messages.append(SessionId("lin-dp-mid"), _msg("mid 1", 2))
    _continue(backend, scope_a, "lin-dp-mid", "lin-dp-tip")
    store_a.messages.append(SessionId("lin-dp-tip"), _msg("tip 1", 3))
    prefix = store_a.lineage.ancestor_display_prefix(SessionId("lin-dp-tip"))
    # CONTRACT: every ancestor's rows, root first, insertion order; the session's own rows are not in the prefix.
    assert [r.content for r in prefix] == ["root 1", "root 2", "mid 1"]
    # CONTRACT: the prefix of a segment mid-chain is only what lies above it.
    assert [r.content for r in store_a.lineage.ancestor_display_prefix(SessionId("lin-dp-mid"))] == ["root 1", "root 2"]
    # CONTRACT: a root, and an unknown id, have no prefix.
    assert store_a.lineage.ancestor_display_prefix(SessionId("lin-dp-root")) == ()
    assert store_a.lineage.ancestor_display_prefix(SessionId("lin-dp-ghost")) == ()


def test_ancestor_display_prefix_includes_compaction_archived_rows_once(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("lin-dc-root"))
    store_a.messages.append(SessionId("lin-dc-root"), _msg("archived by compaction", 0))
    store_a.messages.compact(SessionId("lin-dc-root"), [_msg("summary", 1)])
    _continue(backend, scope_a, "lin-dc-root", "lin-dc-tip")
    prefix = store_a.lineage.ancestor_display_prefix(SessionId("lin-dc-tip"))
    # CONTRACT: the DISPLAY view of the ancestor: compaction history plus its live rows, each once.
    assert [r.content for r in prefix] == ["archived by compaction", "summary"]
    display = store_a.messages.read_transcript(SessionId("lin-dc-root"), TranscriptRead(view=TranscriptView.DISPLAY))
    assert [(r.content, r.visibility) for r in display] == [("archived by compaction", RowVisibility.COMPACTION_ARCHIVED), ("summary", RowVisibility.ACTIVE)]


def test_ancestor_display_prefix_drops_rows_the_session_carries_forward(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("lin-dt-root"))
    store_a.messages.append(SessionId("lin-dt-root"), _msg("older", 0))
    store_a.messages.append(SessionId("lin-dt-root"), _msg("protected tail", 1))
    _continue(backend, scope_a, "lin-dt-root", "lin-dt-tip")
    # The continuation carries the protected tail forward verbatim (same role, content, timestamp).
    store_a.messages.append(SessionId("lin-dt-tip"), _msg("protected tail", 1))
    prefix = store_a.lineage.ancestor_display_prefix(SessionId("lin-dt-tip"))
    # CONTRACT: the copied row shows where it lives now (the session), not in the prefix.
    assert [r.content for r in prefix] == ["older"]


def test_ancestor_display_prefix_is_empty_for_a_fork_and_a_child_of_a_live_parent(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("lin-db-root"))
    store_a.messages.append(SessionId("lin-db-root"), _msg("root", 0))
    store_a.sessions.create(_new(
        "lin-db-branch", parent_session_id=SessionId("lin-db-root"), lineage_kind=SessionLineageKind.BRANCH))
    store_a.messages.append(SessionId("lin-db-branch"), _msg("root", 0))
    # CONTRACT: a branch copied its transcript; it has no prefix.
    assert store_a.lineage.ancestor_display_prefix(SessionId("lin-db-branch")) == ()
    store_a.sessions.create(_new(
        "lin-db-delegate", parent_session_id=SessionId("lin-db-root"), lineage_kind=SessionLineageKind.DELEGATE))
    # CONTRACT: a delegate child reads alone too: the prefix walks only verified compression edges.
    assert store_a.lineage.ancestor_display_prefix(SessionId("lin-db-delegate")) == ()
    store_a.sessions.create(_new("lin-db-plain", parent_session_id=SessionId("lin-db-root")))
    # CONTRACT: so does a plain child of a parent that never compressed.
    assert store_a.lineage.ancestor_display_prefix(SessionId("lin-db-plain")) == ()


def test_ancestor_display_prefix_keeps_the_live_ancestor_copy_when_the_session_archived_its_own(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("lin-da-root"))
    store_a.messages.append(SessionId("lin-da-root"), _msg("carried", 0))
    _continue(backend, scope_a, "lin-da-root", "lin-da-kid")
    store_a.messages.append(SessionId("lin-da-kid"), _msg("carried", 0))
    store_a.messages.compact(SessionId("lin-da-kid"), [_msg("summary", 1)])
    prefix = store_a.lineage.ancestor_display_prefix(SessionId("lin-da-kid"))
    # CONTRACT: the identity's representative is the ACTIVE row, which lives in the ancestor, so the prefix shows it there.
    assert [r.content for r in prefix] == ["carried"]


# ═══════════════════════════════════════════════════════════════════════════════
# Compression write side: publish, reopen, finalize, counters.
# Fixtures shape rows through the sessions, messages, lineage and coordination contract verbs; backdating an existing row for finalize uses the reference's row dict, while an expired lease for reopen is acquired through the coordination contract and outlived.
# ═══════════════════════════════════════════════════════════════════════════════

def _child(session_id: str, **kwargs) -> CompressionChild:
    return CompressionChild(session_id=SessionId(session_id), source=kwargs.pop("source", "cli"), **kwargs)


def _publish(backend, scope, parent: str, child: CompressionChild, handoff, *, holder: Optional[str] = None, **kwargs):
    """Publish as the compressor does: fenced by ``holder``'s lease, refreshed first, when it holds one; unfenced otherwise."""
    store = backend.open(scope)
    fenced = holder is not None
    return store.lineage.publish_compression_child(SessionId(parent), child, handoff, compression_lock_holder=holder,
                                                     require_compression_lease=fenced, require_lease_refresh=fenced, **kwargs)


def _backdate_start(backend, scope, session_id: str, age: timedelta) -> None:
    """Fixture seam (not a contract path): make a row look ``age`` old. Import can set ``started_at`` on a new row; this helper changes an existing row through the reference's row dict."""
    backend.scope_store(scope)["sessions"][session_id]["started_at"] = datetime.now(UTC) - age


_SHORT_TTL = 0.02


def _expired_lease(backend, scope, resource: str, holder: str) -> str:
    """The holder of a lease that has expired without being released: acquired on a short TTL through the coordination contract, then outlived (the stalled-compressor shape)."""
    store = backend.open(scope)
    token = holder if store.coordination.try_acquire_compression_lock(session_id=SessionId(resource), holder=holder, ttl_seconds=_SHORT_TTL) else None
    assert token is not None
    wait_past(time.time() + _SHORT_TTL)
    return token


def _parent_with_history(backend, scope, parent: str, n: int = 3, **kwargs) -> list:
    store = backend.open(scope)
    store.sessions.create(_new(parent, **kwargs))
    return [store.messages.append(SessionId(parent), _msg(f"turn {i}", i)) for i in range(n)]


# ── publish: shape ─────────────────────────────────────────────────────────────


def test_publish_ends_the_parent_and_lands_the_child_with_the_handoff(backend, scope_a):
    store_a = backend.open(scope_a)
    _parent_with_history(backend, scope_a, "cp-p1", session_key="agent:main:k", chat_id="c1", chat_type="dm", thread_id="t1", user_id="u1", cwd="/work/a", git_repo_root="/work")
    store_a.sessions.update_cwd(SessionId("cp-p1"), "/work/a", git_branch="main", git_repo_root="/work")
    handoff = [_msg("summary", 10), _msg("protected tail", 11)]
    _publish(backend, scope_a, "cp-p1", _child("cp-c1", source="telegram", model="m-2", model_config={"k": 1}, system_prompt="SYS "), handoff,
    )
    parent = store_a.sessions.get(SessionId("cp-p1"))
    child = store_a.sessions.get(SessionId("cp-c1"))
    # CONTRACT: the parent is ended by compression; the child is its continuation with the parent's routing, user and git identity.
    assert parent.end_reason == "compression" and parent.ended_at is not None
    assert child.parent_session_id == SessionId("cp-p1") and child.lineage_kind is SessionLineageKind.NONE
    assert (child.session_key, child.chat_id, child.chat_type, child.thread_id, child.user_id) == ("agent:main:k", "c1", "dm", "t1", "u1")
    assert (child.cwd, child.git_branch, child.git_repo_root) == ("/work/a", "main", "/work")
    # CONTRACT: the child's own facts are stored as given.
    assert (child.source, child.model, dict(child.model_config)) == ("telegram", "m-2", {"k": 1})
    # CONTRACT: the handoff is the child's ACTIVE transcript, in order; totals match.
    rows = store_a.messages.read_transcript(SessionId("cp-c1"), TranscriptRead(view=TranscriptView.ACTIVE))
    assert [r.content for r in rows] == ["summary", "protected tail"]
    assert store_a.sessions.get(SessionId("cp-c1")).message_count == 2
    # CONTRACT: the lineage projections see the rotation.
    assert store_a.lineage.tip(SessionId("cp-p1")) == SessionId("cp-c1")
    assert store_a.lineage.lineage(SessionId("cp-c1")) == (SessionId("cp-p1"), SessionId("cp-c1"))
    assert store_a.lineage.is_fork_child(SessionId("cp-c1")) is False


def test_publish_child_cwd_falls_back_to_the_parent(backend, scope_a):
    store_a = backend.open(scope_a)
    _parent_with_history(backend, scope_a, "cp-cwd-p", cwd="/parent/cwd")
    _publish(backend, scope_a, "cp-cwd-p", _child("cp-cwd-c"), [_msg("s")])
    store_a.sessions.create(_new("cp-cwd-p2", cwd="/parent/cwd"))
    _publish(backend, scope_a, "cp-cwd-p2", _child("cp-cwd-c2", cwd="/own"), [_msg("s")])
    # CONTRACT: a None cwd inherits the parent's; an explicit one is kept.
    assert (store_a.sessions.get(SessionId("cp-cwd-c"))).cwd == "/parent/cwd"
    assert (store_a.sessions.get(SessionId("cp-cwd-c2"))).cwd == "/own"


_PIN = {"version": "v1", "tools": [{"type": "function", "function": {"name": "read_file", "parameters": {"type": "object"}}}]}


def test_publish_child_continues_the_parents_whole_tool_pin(backend, scope_a):
    store_a = backend.open(scope_a)
    _parent_with_history(backend, scope_a, "cp-pin-p")
    store_a.sessions.update_tool_pin(SessionId("cp-pin-p"), _PIN)
    _publish(backend, scope_a, "cp-pin-p", _child("cp-pin-c"), [_msg("s")])
    store_a.sessions.create(_new("cp-pin-bare"))
    _publish(backend, scope_a, "cp-pin-bare", _child("cp-pin-bare-c"), [_msg("s")])
    # CONTRACT: the child serves the parent's pin document unchanged; an unpinned parent yields an unpinned child.
    assert (store_a.sessions.get(SessionId("cp-pin-c"))).tool_pin == _PIN
    assert (store_a.sessions.get(SessionId("cp-pin-bare-c"))).tool_pin is None


def test_publish_clones_the_foreign_tail_after_the_watermark_up_to_the_ceiling(backend, scope_a):
    store_a = backend.open(scope_a)
    ids = _parent_with_history(backend, scope_a, "cp-wm", n=2)
    watermark = store_a.messages.active_watermark(SessionId("cp-wm"))
    assert watermark.last_active == ids[-1]
    foreign_a = store_a.messages.append(SessionId("cp-wm"), _msg("foreign a", 20))
    foreign_b = store_a.messages.append(SessionId("cp-wm"), _msg("foreign b", 21))
    store_a.messages.append(SessionId("cp-wm"), _msg("own flush", 22))
    _publish(backend, scope_a, "cp-wm", _child("cp-wm-c"), [_msg("summary", 30)], watermark=watermark.last_active, watermark_ceiling=foreign_b,
    )
    rows = store_a.messages.read_transcript(SessionId("cp-wm-c"), TranscriptRead(view=TranscriptView.ACTIVE))
    # CONTRACT: handoff first, then the rows in (watermark, ceiling] in parent order, under fresh ids; the flush above the ceiling is not cloned.
    assert [r.content for r in rows] == ["summary", "foreign a", "foreign b"]
    assert all(int(r.id) not in {int(foreign_a), int(foreign_b)} for r in rows)
    assert store_a.sessions.get(SessionId("cp-wm-c")).message_count == 3
    # CONTRACT: the originals stay active in the parent for lineage recovery.
    parent_rows = store_a.messages.read_transcript(SessionId("cp-wm"), TranscriptRead(view=TranscriptView.ACTIVE))
    assert [r.content for r in parent_rows] == ["turn 0", "turn 1", "foreign a", "foreign b", "own flush"]


def test_publish_returns_the_handoff_row_state_a_reflush_resolves_to(backend, scope_a):
    store_a = backend.open(scope_a)
    _parent_with_history(backend, scope_a, "cp-hr", n=1)
    watermark = store_a.messages.active_watermark(SessionId("cp-hr"))
    store_a.messages.append(SessionId("cp-hr"), _msg("foreign", 20))
    draft = NewMessage(role="assistant", content="draft", timestamp=_T0 + 31)
    published = _publish(backend, scope_a, "cp-hr", _child("cp-hr-c"), [_msg("summary", 30), draft], watermark=watermark.last_active,
    )
    rows = store_a.messages.read_transcript(SessionId("cp-hr-c"), TranscriptRead(view=TranscriptView.ACTIVE))
    # CONTRACT: one RowStamp per handoff row, in order, naming the child row it became; cloned tail rows are not reported.
    assert [r.content for r in rows] == ["summary", "draft", "foreign"]
    assert [p.id for p in published] == [rows[0].id, rows[1].id]
    assert [p.timestamp for p in published] == [_msg("summary", 30).timestamp, draft.timestamp]
    assert [p.message_uid for p in published] == [rows[0].message_uid, rows[1].message_uid] and all(published[i].message_uid for i in (0, 1))
    assert all(isinstance(p.db_row_snapshot, str) and p.db_row_snapshot for p in published)
    # CONTRACT: the reported ownership version is the child row's own: a re-flush handing it back rewrites that row in place.
    final = NewMessage(role="assistant", content="final", timestamp=draft.timestamp, repair_target=published[1].id,
                       db_row_snapshot=published[1].db_row_snapshot)
    result = store_a.messages.append_batch(SessionId("cp-hr-c"), [final])
    assert (result.outcomes[0].id, result.outcomes[0].disposition, result.inserted) == (published[1].id, RowDisposition.REPAIRED, 0)
    after = store_a.messages.read_transcript(SessionId("cp-hr-c"), TranscriptRead(view=TranscriptView.ACTIVE))
    assert [r.content for r in after] == ["summary", "final", "foreign"]


def test_publish_watermark_variants(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("cp-wv-empty"))
    empty_mark = store_a.messages.active_watermark(SessionId("cp-wv-empty"))
    assert empty_mark.last_active is None
    store_a.messages.append(SessionId("cp-wv-empty"), _msg("arrived", 1))
    _publish(backend, scope_a, "cp-wv-empty", _child("cp-wv-c1"), [_msg("s")], watermark=empty_mark.last_active or MessageId(0))
    # CONTRACT: watermark 0, the mark of an empty transcript, clones every active row that arrived since.
    assert [r.content for r in store_a.messages.read_transcript(SessionId("cp-wv-c1"), TranscriptRead(view=TranscriptView.ACTIVE))] == ["s", "arrived"]
    _parent_with_history(backend, scope_a, "cp-wv-none", n=2)
    _publish(backend, scope_a, "cp-wv-none", _child("cp-wv-c2"), [_msg("s")])
    # CONTRACT: no watermark clones nothing.
    assert [r.content for r in store_a.messages.read_transcript(SessionId("cp-wv-c2"), TranscriptRead(view=TranscriptView.ACTIVE))] == ["s"]
    ids = _parent_with_history(backend, scope_a, "cp-wv-unb", n=1)
    mark = store_a.messages.active_watermark(SessionId("cp-wv-unb"))
    store_a.messages.append(SessionId("cp-wv-unb"), _msg("late 1", 5))
    store_a.messages.append(SessionId("cp-wv-unb"), _msg("late 2", 6))
    _publish(backend, scope_a, "cp-wv-unb", _child("cp-wv-c3"), [_msg("s")], watermark=mark.last_active)
    # CONTRACT: no ceiling means unbounded.
    assert [r.content for r in store_a.messages.read_transcript(SessionId("cp-wv-c3"), TranscriptRead(view=TranscriptView.ACTIVE))] == ["s", "late 1", "late 2"]
    assert ids[0] < mark.last_active or ids[0] == mark.last_active


# ── publish: refusals ──────────────────────────────────────────────────────────


def test_publish_input_shape_refusals_touch_nothing(backend, scope_a):
    store_a = backend.open(scope_a)
    _parent_with_history(backend, scope_a, "cp-shape")
    # CONTRACT: an empty parent id is a missing parent and an empty handoff a refused publish, both the store's own refusal.
    with pytest.raises(CompressionPublishRefused, match="Compression parent not found"):
        _publish(backend, scope_a, "", _child("cp-shape-c"), [_msg("s")])
    with pytest.raises(CompressionPublishRefused, match="handoff must not be empty"):
        _publish(backend, scope_a, "cp-shape", _child("cp-shape-c"), [])
    with pytest.raises(ValueError):
        CompressionChild(session_id=SessionId(""), source="cli")
    with pytest.raises(ValueError):
        CompressionChild(session_id=SessionId("x"), source="")
    # CONTRACT: nothing was written.
    assert (store_a.sessions.get(SessionId("cp-shape"))).ended_at is None
    assert store_a.sessions.get(SessionId("cp-shape-c")) is None


def test_publish_unknown_parent_is_refused(backend, scope_a):
    store_a = backend.open(scope_a)
    with pytest.raises(CompressionPublishRefused, match="Compression parent not found: cp-ghost"):
        _publish(backend, scope_a, "cp-ghost", _child("cp-ghost-c"), [_msg("s")])
    assert store_a.sessions.get(SessionId("cp-ghost-c")) is None


def test_publish_heals_an_automatic_end_but_refuses_a_deliberate_one(backend, scope_a):
    store_a = backend.open(scope_a)
    automatic = sorted(AUTOMATIC_END_REASONS)[0]
    _parent_with_history(backend, scope_a, "cp-auto")
    store_a.sessions.end(SessionId("cp-auto"), automatic)
    _publish(backend, scope_a, "cp-auto", _child("cp-auto-c"), [_msg("s")])
    # CONTRACT: the automatic stamp is replaced by the compression boundary in the same transition.
    assert (store_a.sessions.get(SessionId("cp-auto"))).end_reason == "compression"
    assert store_a.lineage.tip(SessionId("cp-auto")) == SessionId("cp-auto-c")
    _parent_with_history(backend, scope_a, "cp-delib")
    store_a.sessions.end(SessionId("cp-delib"), "session_reset")
    with pytest.raises(CompressionPublishRefused, match="Compression parent already ended: cp-delib"):
        _publish(backend, scope_a, "cp-delib", _child("cp-delib-c"), [_msg("s")])
    # CONTRACT: a deliberate boundary fails closed with nothing written.
    assert (store_a.sessions.get(SessionId("cp-delib"))).end_reason == "session_reset"
    assert store_a.sessions.get(SessionId("cp-delib-c")) is None
    _parent_with_history(backend, scope_a, "cp-twice")
    _publish(backend, scope_a, "cp-twice", _child("cp-twice-c"), [_msg("s")])
    with pytest.raises(CompressionPublishRefused):
        _publish(backend, scope_a, "cp-twice", _child("cp-twice-c2"), [_msg("s")])
    # CONTRACT: a parent already rotated is a deliberate boundary; only one child exists.
    assert store_a.sessions.get(SessionId("cp-twice-c2")) is None


def test_publish_refuses_a_child_id_already_recorded(backend, scope_a):
    store_a = backend.open(scope_a)
    _parent_with_history(backend, scope_a, "cp-dup")
    store_a.sessions.create(_new("cp-dup-c"))
    with pytest.raises(Conflict):
        _publish(backend, scope_a, "cp-dup", _child("cp-dup-c"), [_msg("s")])
    # CONTRACT: nothing written: the parent stays live and the existing row is untouched.
    assert (store_a.sessions.get(SessionId("cp-dup"))).ended_at is None
    assert (store_a.sessions.get(SessionId("cp-dup-c"))).parent_session_id is None
    assert store_a.messages.read_transcript(SessionId("cp-dup-c"), TranscriptRead(view=TranscriptView.ALL)) == ()


def test_publish_is_scope_isolated(backend, scope_a, scope_b):
    store_a = backend.open(scope_a)
    _parent_with_history(backend, scope_a, "cp-scope")
    with pytest.raises(CompressionPublishRefused):
        _publish(backend, scope_b, "cp-scope", _child("cp-scope-c"), [_msg("s")])
    assert (store_a.sessions.get(SessionId("cp-scope"))).ended_at is None


# ── publish: lease fence ───────────────────────────────────────────────────────


def test_publish_under_a_lease_succeeds_for_the_holder_and_refuses_everyone_else(backend, scope_a):
    store_a = backend.open(scope_a)
    _parent_with_history(backend, scope_a, "cp-lease")
    token = "holder-a" if store_a.coordination.try_acquire_compression_lock(session_id=SessionId("cp-lease"), holder="holder-a", ttl_seconds=60.0) else None
    assert token is not None
    with pytest.raises(CompressionBusy):
        _publish(backend, scope_a, "cp-lease", _child("cp-lease-x"), [_msg("s")], holder="holder-b")
    # CONTRACT: the stranger wrote nothing.
    assert (store_a.sessions.get(SessionId("cp-lease"))).ended_at is None
    assert store_a.sessions.get(SessionId("cp-lease-x")) is None
    _publish(backend, scope_a, "cp-lease", _child("cp-lease-c"), [_msg("s")], holder=token)
    assert store_a.lineage.tip(SessionId("cp-lease")) == SessionId("cp-lease-c")
    # CONTRACT: the publish refreshed the holder's lease inside the transition (still held afterwards).
    assert store_a.coordination.get_compression_lock_holder(session_id=SessionId("cp-lease")) == "holder-a"


def test_publish_with_a_released_or_foreign_holder_is_lease_lost(backend, scope_a):
    store_a = backend.open(scope_a)
    _parent_with_history(backend, scope_a, "cp-rel")
    token = "holder-a" if store_a.coordination.try_acquire_compression_lock(session_id=SessionId("cp-rel"), holder="holder-a", ttl_seconds=60.0) else None
    store_a.coordination.release_compression_lock(session_id=SessionId("cp-rel"), holder=token)
    with pytest.raises(CompressionBusy):
        _publish(backend, scope_a, "cp-rel", _child("cp-rel-c"), [_msg("s")], holder=token)
    with pytest.raises(CompressionBusy):
        _publish(backend, scope_a, "cp-rel", _child("cp-rel-c"), [_msg("s")], holder="never")
    # CONTRACT: a lease held on another resource fences nothing here.
    assert store_a.coordination.try_acquire_compression_lock(session_id=SessionId("other"), holder="holder-a", ttl_seconds=60.0) is True
    with pytest.raises(CompressionBusy):
        _publish(backend, scope_a, "cp-rel", _child("cp-rel-c"), [_msg("s")], holder="holder-a")
    # CONTRACT: a lease is required unless the caller says otherwise, so a publish naming no holder is lease loss too.
    with pytest.raises(CompressionBusy, match="Compression lease lost before publication: cp-rel"):
        store_a.lineage.publish_compression_child(SessionId("cp-rel"), _child("cp-rel-c"), [_msg("s")])
    assert (store_a.sessions.get(SessionId("cp-rel"))).ended_at is None


def test_publish_revives_the_holders_expired_but_unclaimed_lease(backend, scope_a):
    store_a = backend.open(scope_a)
    _parent_with_history(backend, scope_a, "cp-exp")
    token = _expired_lease(backend, scope_a, "cp-exp", "holder-a")
    assert store_a.coordination.get_compression_lock_holder(session_id=SessionId("cp-exp")) is None
    # CONTRACT (anti-orphan rule): the refresh-then-verify fence lets the original holder publish as long as nobody reclaimed the lease.
    _publish(backend, scope_a, "cp-exp", _child("cp-exp-c"), [_msg("s")], holder=token)
    assert store_a.lineage.tip(SessionId("cp-exp")) == SessionId("cp-exp-c")
    _parent_with_history(backend, scope_a, "cp-exp2")
    stale = _expired_lease(backend, scope_a, "cp-exp2", "holder-a")
    reclaimed = "holder-b" if store_a.coordination.try_acquire_compression_lock(session_id=SessionId("cp-exp2"), holder="holder-b", ttl_seconds=60.0) else None
    assert reclaimed is not None
    with pytest.raises(CompressionBusy):
        _publish(backend, scope_a, "cp-exp2", _child("cp-exp2-c"), [_msg("s")], holder=stale)
    # CONTRACT: a reclaimed lease fences the stale holder out; the new holder may publish.
    _publish(backend, scope_a, "cp-exp2", _child("cp-exp2-c"), [_msg("s")], holder=reclaimed)
    # CONTRACT: without the refresh an expired lease is lease loss even for its holder.
    _parent_with_history(backend, scope_a, "cp-exp3")
    expired = _expired_lease(backend, scope_a, "cp-exp3", "holder-a")
    with pytest.raises(CompressionBusy):
        store_a.lineage.publish_compression_child(SessionId("cp-exp3"), _child("cp-exp3-c"), [_msg("s")], compression_lock_holder=expired)
    assert store_a.sessions.get(SessionId("cp-exp3-c")) is None


# ── publish: races (two contenders) ────────────────────────────────────────────


def test_two_unfenced_publishers_land_exactly_one_child(backend, scope_a):
    store_a = backend.open(scope_a)
    _parent_with_history(backend, scope_a, "cp-race-u")
    outcomes = run_concurrently(lambda: _publish(backend, scope_a, "cp-race-u", _child("cp-race-u-1"), [_msg("one")]), lambda: _publish(backend, scope_a, "cp-race-u", _child("cp-race-u-2"), [_msg("two")]), return_exceptions=True)
    # CONTRACT: exactly one child lands; the loser is refused (the parent is already ended) and wrote nothing.
    assert sorted(type(o).__name__ for o in outcomes) == ["CompressionPublishRefused", "tuple"]
    children = [c for c in ("cp-race-u-1", "cp-race-u-2") if store_a.sessions.get(SessionId(c)) is not None]
    assert len(children) == 1
    assert store_a.lineage.tip(SessionId("cp-race-u")) == SessionId(children[0])


def test_two_fenced_publishers_only_the_lease_holder_lands(backend, scope_a):
    store_a = backend.open(scope_a)
    _parent_with_history(backend, scope_a, "cp-race-f")
    token = "holder-a" if store_a.coordination.try_acquire_compression_lock(session_id=SessionId("cp-race-f"), holder="holder-a", ttl_seconds=60.0) else None
    outcomes = run_concurrently(lambda: _publish(backend, scope_a, "cp-race-f", _child("cp-race-f-a"), [_msg("a")], holder=token), lambda: _publish(backend, scope_a, "cp-race-f", _child("cp-race-f-b"), [_msg("b")], holder="holder-b"), return_exceptions=True)
    # CONTRACT: the holder lands, the other raises CompressionBusy, whatever the interleaving.
    assert isinstance(outcomes[0], tuple) and isinstance(outcomes[1], CompressionBusy)
    assert store_a.lineage.tip(SessionId("cp-race-f")) == SessionId("cp-race-f-a")
    assert store_a.sessions.get(SessionId("cp-race-f-b")) is None


def test_publisher_and_explicit_end_both_commit_orders(backend, scope_a):
    # Order 1: the end commits first: a deliberate boundary, the publish refuses and writes nothing.
    store_a = backend.open(scope_a)
    _parent_with_history(backend, scope_a, "cp-pe-1")
    store_a.sessions.end(SessionId("cp-pe-1"), "session_reset")
    with pytest.raises(CompressionPublishRefused):
        _publish(backend, scope_a, "cp-pe-1", _child("cp-pe-1-c"), [_msg("s")])
    assert store_a.sessions.get(SessionId("cp-pe-1-c")) is None
    assert (store_a.sessions.get(SessionId("cp-pe-1"))).end_reason == "session_reset"
    # Order 2: the publish commits first: the later end is the sessions contract's no-op on an ended row (first stamp kept).
    _parent_with_history(backend, scope_a, "cp-pe-2")
    _publish(backend, scope_a, "cp-pe-2", _child("cp-pe-2-c"), [_msg("s")])
    stamped = store_a.sessions.get(SessionId("cp-pe-2"))
    store_a.sessions.end(SessionId("cp-pe-2"), "session_reset")
    after = store_a.sessions.get(SessionId("cp-pe-2"))
    # CONTRACT
    assert (after.end_reason, after.ended_at) == ("compression", stamped.ended_at)
    assert store_a.lineage.tip(SessionId("cp-pe-2")) == SessionId("cp-pe-2-c")


def test_publisher_racing_an_explicit_end_lands_on_one_of_the_two_promised_states(backend, scope_a):
    store_a = backend.open(scope_a)
    _parent_with_history(backend, scope_a, "cp-race-e")
    outcomes = run_concurrently(lambda: _publish(backend, scope_a, "cp-race-e", _child("cp-race-e-c"), [_msg("s")]), lambda: store_a.sessions.end(SessionId("cp-race-e"), "session_reset"), return_exceptions=True)
    parent = store_a.sessions.get(SessionId("cp-race-e"))
    child = store_a.sessions.get(SessionId("cp-race-e-c"))
    # CONTRACT: the end never raises; the publish either lost to the end (Conflict, no child, reset stamp) or won (child present, compression stamp kept over the later end).
    assert outcomes[1] is None
    if isinstance(outcomes[0], CompressionPublishRefused):
        assert child is None and parent.end_reason == "session_reset"
    else:
        assert isinstance(outcomes[0], tuple) and child is not None and parent.end_reason == "compression"
        assert store_a.lineage.tip(SessionId("cp-race-e")) == SessionId("cp-race-e-c")


def test_a_refused_publish_leaves_the_lease_as_it_found_it(backend, scope_a):
    # The expired lease row admits the same holder's refresh, but publication refuses: the refresh must not persist.
    store_a = backend.open(scope_a)
    _parent_with_history(backend, scope_a, "cp-nt")
    store_a.sessions.create(_new("cp-nt-dup"))
    token = _expired_lease(backend, scope_a, "cp-nt", "holder-a")
    with pytest.raises(Conflict):
        _publish(backend, scope_a, "cp-nt", _child("cp-nt-dup"), [_msg("s")], holder=token)
    # CONTRACT: the expired lease is still expired (nobody holds it), the parent is still live, nothing was written.
    assert store_a.coordination.get_compression_lock_holder(session_id=SessionId("cp-nt")) is None
    assert (store_a.sessions.get(SessionId("cp-nt"))).ended_at is None
    assert store_a.messages.read_transcript(SessionId("cp-nt-dup"), TranscriptRead(view=TranscriptView.ALL)) == ()
    # A never-acquired token reports lease loss before parent state; neither refusal leaves a lease behind.
    _parent_with_history(backend, scope_a, "cp-nt2")
    store_a.sessions.end(SessionId("cp-nt2"), "session_reset")
    with pytest.raises(CompressionBusy):
        _publish(backend, scope_a, "cp-nt2", _child("cp-nt2-c"), [_msg("s")], holder="holder-a")
    assert store_a.coordination.get_compression_lock_holder(session_id=SessionId("cp-nt2")) is None
    assert store_a.sessions.get(SessionId("cp-nt2-c")) is None
    with pytest.raises(CompressionBusy):
        _publish(backend, scope_a, "cp-nt-ghost", _child("cp-nt-ghost-c"), [_msg("s")], holder="holder-a")
    assert store_a.coordination.get_compression_lock_holder(session_id=SessionId("cp-nt-ghost")) is None


def test_publish_child_inherits_an_empty_cwd_and_stores_its_config_as_given(backend, scope_a):
    store_a = backend.open(scope_a)
    _parent_with_history(backend, scope_a, "cp-mk", cwd="/parent")
    _publish(backend, scope_a, "cp-mk", _child("cp-mk-c", cwd="", model_config={"max_iterations": 3}), [_msg("s")])
    child = store_a.sessions.get(SessionId("cp-mk-c"))
    # CONTRACT: a continuation is never a fork; an empty cwd inherits like None; config is stored as given.
    assert child.lineage_kind is SessionLineageKind.NONE and store_a.lineage.is_fork_child(SessionId("cp-mk-c")) is False
    assert child.cwd == "/parent" and dict(child.model_config) == {"max_iterations": 3}
    assert store_a.lineage.tip(SessionId("cp-mk")) == SessionId("cp-mk-c")


# ── reopen ─────────────────────────────────────────────────────────────────────


def _orphaned_parent(backend, scope, parent: str) -> None:
    """A parent ended by compression whose handoff never landed (no child)."""
    store = backend.open(scope)
    _parent_with_history(backend, scope, parent)
    store.sessions.end(SessionId(parent), "compression")


def test_reopen_gives_the_parent_back_when_nothing_claims_the_lineage(backend, scope_a):
    store_a = backend.open(scope_a)
    _orphaned_parent(backend, scope_a, "cp-ro")
    assert store_a.lineage.reopen_orphaned_compression(SessionId("cp-ro")) is True
    row = store_a.sessions.get(SessionId("cp-ro"))
    # CONTRACT: the row is live again; a second reopen changes nothing.
    assert row.ended_at is None and row.end_reason is None
    assert store_a.lineage.reopen_orphaned_compression(SessionId("cp-ro")) is False
    assert [r.content for r in store_a.messages.read_transcript(SessionId("cp-ro"), TranscriptRead(view=TranscriptView.ACTIVE))] == ["turn 0", "turn 1", "turn 2"]


def test_reopen_refuses_every_other_state(backend, scope_a, scope_b):
    store_a = backend.open(scope_a)
    store_b = backend.open(scope_b)
    assert store_a.lineage.reopen_orphaned_compression(SessionId("")) is False
    assert store_a.lineage.reopen_orphaned_compression(SessionId("cp-ro-ghost")) is False
    _parent_with_history(backend, scope_a, "cp-ro-live")
    assert store_a.lineage.reopen_orphaned_compression(SessionId("cp-ro-live")) is False
    _parent_with_history(backend, scope_a, "cp-ro-reset")
    store_a.sessions.end(SessionId("cp-ro-reset"), "session_reset")
    # CONTRACT: only a compression stamp is recoverable.
    assert store_a.lineage.reopen_orphaned_compression(SessionId("cp-ro-reset")) is False
    assert (store_a.sessions.get(SessionId("cp-ro-reset"))).end_reason == "session_reset"
    _parent_with_history(backend, scope_a, "cp-ro-done")
    _publish(backend, scope_a, "cp-ro-done", _child("cp-ro-done-c"), [_msg("s")])
    # CONTRACT: a continuation child, live or ended, means another path owns the lineage.
    assert store_a.lineage.reopen_orphaned_compression(SessionId("cp-ro-done")) is False
    store_a.sessions.end(SessionId("cp-ro-done-c"), "session_reset")
    assert store_a.lineage.reopen_orphaned_compression(SessionId("cp-ro-done")) is False
    _orphaned_parent(backend, scope_a, "cp-ro-fork")
    store_a.sessions.create(_new("cp-ro-fork-b", parent_session_id=SessionId("cp-ro-fork"), lineage_kind=SessionLineageKind.BRANCH))
    store_a.sessions.create(_new("cp-ro-fork-t", source="tool", parent_session_id=SessionId("cp-ro-fork")))
    # CONTRACT: BRANCH / DELEGATE / tool-sourced children are not continuations and do not block recovery.
    assert store_a.lineage.reopen_orphaned_compression(SessionId("cp-ro-fork")) is True
    _orphaned_parent(backend, scope_a, "cp-ro-scope")
    # CONTRACT: scope isolation: the other scope does not see the orphan.
    assert store_b.lineage.reopen_orphaned_compression(SessionId("cp-ro-scope")) is False


def test_reopen_respects_a_live_lease_and_deletes_an_expired_one(backend, scope_a):
    store_a = backend.open(scope_a)
    _orphaned_parent(backend, scope_a, "cp-ro-lease")
    assert store_a.coordination.try_acquire_compression_lock(session_id=SessionId("cp-ro-lease"), holder="compressor", ttl_seconds=60.0) is True
    # CONTRACT: a live lease means a compressor still owns the handoff; recovery waits.
    assert store_a.lineage.reopen_orphaned_compression(SessionId("cp-ro-lease")) is False
    store_a.coordination.release_compression_lock(session_id=SessionId("cp-ro-lease"), holder="compressor")
    token = _expired_lease(backend, scope_a, "cp-ro-lease", "compressor")
    assert store_a.lineage.reopen_orphaned_compression(SessionId("cp-ro-lease")) is True
    # CONTRACT: the expired lease row is gone with the reopen, so the stale holder's refresh cannot revive it.
    assert store_a.coordination.refresh_compression_lock(session_id=SessionId("cp-ro-lease"), holder=token, ttl_seconds=60.0) is False
    assert store_a.coordination.get_compression_lock_holder(session_id=SessionId("cp-ro-lease")) is None


def test_two_reopeners_exactly_one_wins(backend, scope_a):
    store_a = backend.open(scope_a)
    _orphaned_parent(backend, scope_a, "cp-ro-race")
    outcomes = run_concurrently(lambda: store_a.lineage.reopen_orphaned_compression(SessionId("cp-ro-race")), lambda: store_a.lineage.reopen_orphaned_compression(SessionId("cp-ro-race")))
    assert sorted(outcomes) == [False, True]


def test_reopen_and_publish_both_commit_orders(backend, scope_a):
    # Order 1: the publish commits first against the compression stamp: refused, nothing written, and the reopen then succeeds.
    store_a = backend.open(scope_a)
    _orphaned_parent(backend, scope_a, "cp-rp-1")
    with pytest.raises(CompressionPublishRefused):
        _publish(backend, scope_a, "cp-rp-1", _child("cp-rp-1-c"), [_msg("s")])
    assert store_a.sessions.get(SessionId("cp-rp-1-c")) is None
    assert store_a.lineage.reopen_orphaned_compression(SessionId("cp-rp-1")) is True
    # Order 2: the reopen commits first: the parent is live and the publish rotates it.
    _orphaned_parent(backend, scope_a, "cp-rp-2")
    assert store_a.lineage.reopen_orphaned_compression(SessionId("cp-rp-2")) is True
    _publish(backend, scope_a, "cp-rp-2", _child("cp-rp-2-c"), [_msg("s")])
    # CONTRACT
    assert store_a.lineage.tip(SessionId("cp-rp-2")) == SessionId("cp-rp-2-c")
    assert (store_a.sessions.get(SessionId("cp-rp-2"))).end_reason == "compression"


def test_reopener_racing_a_publisher_lands_on_one_of_the_two_promised_states(backend, scope_a):
    # Shape: a parent ended by compression with no child; a stalled compressor (unfenced) finally publishes while recovery reopens.
    store_a = backend.open(scope_a)
    _orphaned_parent(backend, scope_a, "cp-rp2")
    outcomes = run_concurrently(lambda: store_a.lineage.reopen_orphaned_compression(SessionId("cp-rp2")), lambda: _publish(backend, scope_a, "cp-rp2", _child("cp-rp2-c"), [_msg("s")]), return_exceptions=True)
    reopened, published = outcomes
    parent = store_a.sessions.get(SessionId("cp-rp2"))
    child = store_a.sessions.get(SessionId("cp-rp2-c"))
    # CONTRACT: the reopen never fails because of the publisher; the publish either lost (committed first against the stamp: Conflict, no child, parent live after the reopen) or won (committed after the reopen: child present, parent re-stamped compression). Never a child beside a live parent.
    assert reopened is True
    if isinstance(published, tuple):
        assert child is not None and parent.end_reason == "compression"
        assert store_a.lineage.tip(SessionId("cp-rp2")) == SessionId("cp-rp2-c")
    else:
        assert isinstance(published, CompressionPublishRefused) and child is None and parent.ended_at is None


def test_stale_refresh_racing_a_reopen_never_both_succeed(backend, scope_a):
    store_a = backend.open(scope_a)
    _orphaned_parent(backend, scope_a, "cp-rr")
    token = _expired_lease(backend, scope_a, "cp-rr", "compressor")
    outcomes = run_concurrently(lambda: store_a.lineage.reopen_orphaned_compression(SessionId("cp-rr")), lambda: store_a.coordination.refresh_compression_lock(session_id=SessionId("cp-rr"), holder=token, ttl_seconds=60.0))
    reopened, refreshed = outcomes
    holder = store_a.coordination.get_compression_lock_holder(session_id=SessionId("cp-rr"))
    parent = store_a.sessions.get(SessionId("cp-rr"))
    # CONTRACT: exactly one of them wins the expired row: reopen True + refresh False + no holder + live parent, or refresh True + reopen False + holder back + parent still stamped.
    assert (reopened, refreshed) in {(True, False), (False, True)}
    if reopened:
        assert holder is None and parent.ended_at is None
    else:
        assert holder == "compressor" and parent.end_reason == "compression"


# ── finalize ───────────────────────────────────────────────────────────────────


def _undriven_child(backend, scope, parent: str, child: str, *, age: timedelta = _ORPHAN_AGE + timedelta(hours=1)) -> None:
    """A published continuation nobody ever drove: rows carried over, no API call, older than the window."""
    _parent_with_history(backend, scope, parent)
    _publish(backend, scope, parent, _child(child), [_msg("s")])
    _backdate_start(backend, scope, child, age)


def test_finalize_ends_undriven_continuations_once(backend, scope_a):
    store_a = backend.open(scope_a)
    _undriven_child(backend, scope_a, "cp-fin-p", "cp-fin-c")
    assert store_a.lineage.finalize_orphaned_compression_children() == 1
    row = store_a.sessions.get(SessionId("cp-fin-c"))
    # CONTRACT: ended with the orphan stamp, transcript kept, and idempotent.
    assert row.end_reason == END_REASON_ORPHANED_COMPRESSION and row.ended_at is not None
    assert [r.content for r in store_a.messages.read_transcript(SessionId("cp-fin-c"), TranscriptRead(view=TranscriptView.ACTIVE))] == ["s"]
    assert store_a.lineage.finalize_orphaned_compression_children() == 0
    assert (store_a.sessions.get(SessionId("cp-fin-p"))).end_reason == "compression"


def test_finalize_skips_every_non_orphan_shape(backend, scope_a, scope_b):
    store_a = backend.open(scope_a)
    store_b = backend.open(scope_b)
    assert store_a.lineage.finalize_orphaned_compression_children() == 0
    # Too young.
    _undriven_child(backend, scope_a, "cp-fy-p", "cp-fy-c", age=_ORPHAN_AGE - timedelta(hours=1))
    # Driven: an API call was accounted.
    _undriven_child(backend, scope_a, "cp-fd-p", "cp-fd-c")
    store_a.usage.queue_usage(SessionId("cp-fd-c"), UsageDelta(api_call_count=1))
    # Already ended.
    _undriven_child(backend, scope_a, "cp-fe-p", "cp-fe-c")
    store_a.sessions.end(SessionId("cp-fe-c"), "session_reset")
    # Parent not ended by compression (a plain child of a live parent).
    store_a.sessions.create(_new("cp-fl-p"))
    store_a.sessions.create(_new("cp-fl-c", parent_session_id=SessionId("cp-fl-p")))
    store_a.messages.append(SessionId("cp-fl-c"), _msg("x"))
    _backdate_start(backend, scope_a, "cp-fl-c", _ORPHAN_AGE * 2)
    # No transcript rows at all.
    store_a.sessions.create(_new("cp-fn-p"))
    store_a.sessions.end(SessionId("cp-fn-p"), "compression")
    store_a.sessions.create(_new("cp-fn-c", parent_session_id=SessionId("cp-fn-p")))
    _backdate_start(backend, scope_a, "cp-fn-c", _ORPHAN_AGE * 2)
    # Another scope's orphan.
    _undriven_child(backend, scope_b, "cp-fs-p", "cp-fs-c")
    assert store_a.lineage.finalize_orphaned_compression_children() == 0
    for sid in ("cp-fy-c", "cp-fd-c", "cp-fl-c", "cp-fn-c"):
        assert (store_a.sessions.get(SessionId(sid))).end_reason != END_REASON_ORPHANED_COMPRESSION
    assert (store_a.sessions.get(SessionId("cp-fe-c"))).end_reason == "session_reset"
    # CONTRACT: the other scope's orphan is that scope's to finalize.
    assert store_b.lineage.finalize_orphaned_compression_children() == 1


# ── legacy chain stamping ──────────────────────────────────────────────────────


def _legacy_edge(backend, scope, parent: str, child: str, **kwargs) -> None:
    """A compression edge as an older store left it: the child created while the parent was live, the compression end stamped after."""
    store = backend.open(scope)
    store.sessions.create(_new(child, parent_session_id=SessionId(parent), **kwargs))
    store.sessions.end(SessionId(parent), "compression")


def test_stamp_legacy_compression_chains_copies_a_stamped_surface_up_unset_ancestors(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("cp-st-root", source="local"))
    _legacy_edge(backend, scope_a, "cp-st-root", "cp-st-mid", source="local")
    _legacy_edge(backend, scope_a, "cp-st-mid", "cp-st-tip", source="telegram")
    store_a.sessions.create(_new("cp-st-desk", source="desktop"))
    _legacy_edge(backend, scope_a, "cp-st-desk", "cp-st-desk-kid", source="telegram")
    store_a.sessions.create(_new("cp-st-fork", source="local"))
    _legacy_edge(backend, scope_a, "cp-st-fork", "cp-st-branch", source="telegram", lineage_kind=SessionLineageKind.BRANCH)
    created = {sid: store_a.sessions.get(SessionId(sid)).created_source for sid in ("cp-st-root", "cp-st-mid", "cp-st-desk", "cp-st-fork")}
    assert created == {"cp-st-root": "local", "cp-st-mid": "local", "cp-st-desk": "desktop", "cp-st-fork": "local"}
    # CONTRACT: every unset ancestor of a stamped continuation takes its surface; the count is the rows stamped.
    assert store_a.lineage.stamp_legacy_compression_chains() == 2
    assert store_a.sessions.get(SessionId("cp-st-root")).created_source == "telegram"
    assert store_a.sessions.get(SessionId("cp-st-mid")).created_source == "telegram"
    # CONTRACT: a known surface is never replaced, and a fork child stamps nothing above it.
    assert store_a.sessions.get(SessionId("cp-st-desk")).created_source == "desktop"
    assert store_a.sessions.get(SessionId("cp-st-fork")).created_source == "local"
    # CONTRACT: idempotent.
    assert store_a.lineage.stamp_legacy_compression_chains() == 0


def test_publish_stamps_unset_ancestors_with_the_parents_surface_and_carries_its_state(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("cp-sp-root", source="local"))
    _continue(backend, scope_a, "cp-sp-root", "cp-sp-mid", source="local")
    store_a.sessions.create(_new("cp-sp-side", source="local"))
    store_a.sessions.end(SessionId("cp-sp-side"), "compression")
    assert store_a.sessions.set_pinned(SessionId("cp-sp-mid"), True) is True
    _publish(backend, scope_a, "cp-sp-mid", _child("cp-sp-tip", source="telegram"), [_msg("s")])
    child = store_a.sessions.get(SessionId("cp-sp-tip"))
    # CONTRACT: an unset parent surface leaves the child's own as its creation surface, stamped up the chain in the same publish.
    assert child.created_source == "telegram"
    assert [store_a.sessions.get(SessionId(sid)).created_source for sid in ("cp-sp-root", "cp-sp-mid", "cp-sp-side")] == ["telegram", "telegram", "local"]
    # CONTRACT: a continuation carries the parent's pin.
    assert child.pinned is True
    _publish(backend, scope_a, "cp-sp-tip", _child("cp-sp-next", source="cli"), [_msg("s")])
    # CONTRACT: a stamped parent surface is the child's creation surface whatever the child's own source.
    assert store_a.sessions.get(SessionId("cp-sp-next")).created_source == "telegram"


# ── durable compressor counters ────────────────────────────────────────────────


def test_counters_default_persist_and_clamp(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("cp-ctr"))
    sid = SessionId("cp-ctr")
    assert store_a.lineage.compression_fallback_streak(sid) == 0
    assert store_a.lineage.compression_ineffective_count(sid) == 0
    assert store_a.lineage.compression_recovery_deadline(sid) == 0.0
    store_a.lineage.set_compression_fallback_streak(sid, 3)
    store_a.lineage.set_compression_ineffective_count(sid, 2)
    deadline = datetime(2026, 9, 22, 12, 0, 0, tzinfo=UTC).timestamp()
    store_a.lineage.set_compression_recovery_deadline(sid, deadline)
    assert store_a.lineage.compression_fallback_streak(sid) == 3
    assert store_a.lineage.compression_ineffective_count(sid) == 2
    assert store_a.lineage.compression_recovery_deadline(sid) == deadline
    # CONTRACT: negatives clamp to 0; None disarms; last write wins.
    store_a.lineage.set_compression_fallback_streak(sid, -5)
    store_a.lineage.set_compression_ineffective_count(sid, -1)
    store_a.lineage.set_compression_recovery_deadline(sid, None)
    assert store_a.lineage.compression_fallback_streak(sid) == 0
    assert store_a.lineage.compression_ineffective_count(sid) == 0
    assert store_a.lineage.compression_recovery_deadline(sid) == 0.0
    # CONTRACT: 0, a negative deadline and an unparsable value each disarm the clock; a later deadline re-arms it.
    for disarm in (0, -5.0, "not a number"):
        store_a.lineage.set_compression_recovery_deadline(sid, deadline)
        store_a.lineage.set_compression_recovery_deadline(sid, disarm)
        assert store_a.lineage.compression_recovery_deadline(sid) == 0.0
    store_a.lineage.set_compression_recovery_deadline(sid, deadline + 60)
    assert store_a.lineage.compression_recovery_deadline(sid) == deadline + 60


def test_overload_streak_defaults_clamps_and_increments_atomically(backend, scope_a, scope_b):
    """The sustained-overload abort streak a rebuilt compressor inherits."""
    store_a = backend.open(scope_a)
    store_b = backend.open(scope_b)
    store_a.sessions.create(_new("cp-ovl"))
    store_b.sessions.create(_new("cp-ovl"))
    sid, ghost = SessionId("cp-ovl"), SessionId("cp-ovl-ghost")
    assert store_a.lineage.compression_overload_streak(sid) == 0
    # CONTRACT: increment returns the new value; concurrent increments each land (no lost strike).
    assert store_a.lineage.increment_compression_overload_streak(sid) == 1
    seen = run_concurrently(*[lambda: store_a.lineage.increment_compression_overload_streak(sid) for _ in range(8)])
    assert sorted(seen) == list(range(2, 10))
    assert store_a.lineage.compression_overload_streak(sid) == 9
    # CONTRACT: set is last-write-wins and clamps negatives to 0.
    store_a.lineage.set_compression_overload_streak(sid, 4)
    assert store_a.lineage.compression_overload_streak(sid) == 4
    store_a.lineage.set_compression_overload_streak(sid, -2)
    assert store_a.lineage.compression_overload_streak(sid) == 0
    # CONTRACT: profile isolation; the same id in another scope never moved.
    assert store_b.lineage.compression_overload_streak(sid) == 0
    # CONTRACT: an unknown session reads 0, increments to None, and nothing is created.
    assert store_a.lineage.increment_compression_overload_streak(ghost) is None
    store_a.lineage.set_compression_overload_streak(ghost, 3)
    assert store_a.lineage.compression_overload_streak(ghost) == 0
    assert store_a.sessions.get(ghost) is None
    # CONTRACT: an empty id reads 0 and increments to None.
    assert store_a.lineage.compression_overload_streak(SessionId("")) == 0
    assert store_a.lineage.increment_compression_overload_streak(SessionId("")) is None


def test_counters_unknown_session_reads_default_and_writes_nothing(backend, scope_a, scope_b):
    store_a = backend.open(scope_a)
    store_b = backend.open(scope_b)
    ghost = SessionId("cp-ctr-ghost")
    store_a.lineage.set_compression_fallback_streak(ghost, 4)
    store_a.lineage.set_compression_ineffective_count(ghost, 4)
    store_a.lineage.set_compression_recovery_deadline(ghost, time.time())
    assert store_a.sessions.get(ghost) is None
    assert store_a.lineage.compression_fallback_streak(ghost) == 0
    assert store_a.lineage.compression_recovery_deadline(ghost) == 0.0
    # CONTRACT: an empty id reads the defaults and every setter is a no-op on it.
    for setter in ("set_compression_fallback_streak", "set_compression_ineffective_count", "set_compression_overload_streak"):
        getattr(backend.open(scope_a).lineage, setter)(SessionId(""), 3)
    store_a.lineage.set_compression_recovery_deadline(SessionId(""), time.time())
    for verb in ("compression_fallback_streak", "compression_ineffective_count", "compression_overload_streak", "compression_recovery_deadline"):
        assert getattr(backend.open(scope_a).lineage, verb)(SessionId("")) == 0
    store_a.sessions.create(_new("cp-ctr-iso"))
    store_a.lineage.set_compression_fallback_streak(SessionId("cp-ctr-iso"), 7)
    # CONTRACT: scope isolation.
    assert store_b.lineage.compression_fallback_streak(SessionId("cp-ctr-iso")) == 0
