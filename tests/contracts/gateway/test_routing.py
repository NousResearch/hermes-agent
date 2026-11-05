"""Contract cases for the routing-peer, origin, orphan, never-active and topic-preparation verbs of ``persistence.gateway.routing.RoutingRepository``, plus the ``gateway_runtime`` helper.

Every case runs on every registered backend through the ``backend`` fixture except ``test_gateway_runtime_precedence``, which is backend-neutral. Cases seed through the public sessions, messages, usage, coordination and routing verbs.
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime, timedelta
from typing import Any, Callable

import pytest

from persistence import SessionArtifacts, SessionId
from persistence.conversation.messages import NewMessage
from persistence.conversation.sessions import RECOVERABLE_END_REASONS, NewSession, SessionLineageKind
from persistence.conversation.usage import UsageDelta
from persistence.gateway.routing import (
    BARE_BILLING_PROVIDERS,
    END_REASON_SUPERSEDED_BY_REPAIR,
    PEER_RESET_END_REASONS,
    TOPIC_SCHEMA_META_KEY,
    GatewayRuntime,
    NeverActivePrune,
    RoutingRepository,
    gateway_runtime,
)
from tests.contracts._concurrency import run_concurrently
from tests.contracts._waiting import wait_past
from tests.contracts._scopes import MEDIA_ROOT

T0 = datetime(2026, 9, 21, 12, 0, tzinfo=UTC)
FRESH = 0  # every already-created row is older than a zero-day window
SRC = "telegram"


def _new(session_id: str, source: str = SRC, **kwargs: Any) -> NewSession:
    return NewSession(session_id=SessionId(session_id), source=source, **kwargs)


def _peer(store: Any, session_id: str, key: str = "telegram:1", *, source: str = SRC, origin: Any = None, **kwargs: Any) -> None:
    origin_json = None if origin is None else json.dumps(origin)
    store.routing.record_session_peer(session_id, source=source, session_key=key, origin_json=origin_json, **kwargs)


def _latest(store: Any, key: str = "telegram:1", *, source: str = SRC, **kwargs: Any) -> Any:
    return store.routing.latest_session_for_peer(source=source, session_key=key, **kwargs)


def _msg(content: str = "x", *, at: datetime = T0) -> NewMessage:
    return NewMessage(role="user", content=content, timestamp=at)


def _entry(key: str, session_id: str, **extra: Any) -> str:
    return json.dumps({"session_key": key, "session_id": session_id, "created_at": T0.isoformat(), "updated_at": T0.isoformat(), "platform": SRC, "chat_type": "dm", **extra})


def _keyed(backend: Any, scope: Any, session_id: str, key: str, **kwargs: Any) -> None:
    """A keyed gateway session, created through the sessions capability with the peer's identity."""
    backend.open(scope).sessions.create(_new(session_id, session_key=key, **kwargs))


def _tick() -> None:
    # started_at is store-stamped; outrunning the last stamp keeps sequential creates strictly ordered.
    wait_past(time.time())


def _race(*calls: Callable[[], Any]) -> list:
    """Run each call on its own thread (real contention on one handle); results in call order."""
    return run_concurrently(*calls)


# ── protocol + empty identities ───────────────────────────────────────────────


def test_repository_satisfies_protocol(backend, scope_a):
    assert isinstance(backend.open(scope_a).routing, RoutingRepository)


def test_empty_identities_are_silent_no_ops(backend, scope_a):
    store_a = backend.open(scope_a)
    # record_session_peer returns on an empty id or key; adopt returns False; the origin and peer reads return None.
    store_a.routing.record_session_peer("", source=SRC, session_key="k")
    store_a.routing.record_session_peer("o", source=SRC, session_key="")
    store_a.routing.record_session_peer("o", source=SRC)
    assert store_a.sessions.get(SessionId("o")) is None
    assert store_a.routing.adopt_orphan("", "d") is False
    assert store_a.routing.adopt_orphan("o", "") is False
    assert store_a.routing.session_for_origin(platform="", chat_id="1") is None
    assert store_a.routing.session_for_origin(platform=SRC, chat_id="") is None
    _keyed(backend, scope_a, "gr-empty", "telegram:e")
    assert store_a.routing.latest_session_for_peer(source=SRC, session_key="") is None
    assert store_a.routing.latest_session_for_peer(source=SRC) is None


# ── record_session_peer ───────────────────────────────────────────────────────


def test_record_peer_self_heals_a_missing_row(backend, scope_a):
    store_a = backend.open(scope_a)
    _peer(store_a, "gr-heal", "telegram:7", user_id="u7", chat_id="7", chat_type="dm", display_name="Seven", origin={"scope_id": "T7"})
    # The row is inserted with the peer's full identity, the peer's source as created source, and the store owner's profile.
    row = store_a.sessions.get(SessionId("gr-heal"))
    assert row is not None
    assert (row.source, row.created_source, row.session_key, row.user_id, row.chat_id, row.chat_type, row.thread_id) == (
        SRC, SRC, "telegram:7", "u7", "7", "dm", None,
    )
    assert row.ended_at is None and store_a.sessions.owns_session(SessionId("gr-heal"))
    found = _latest(store_a, "telegram:7")
    assert found is not None and found.session.session_id == "gr-heal"
    assert (found.session.display_name, found.session.origin) == ("Seven", {"scope_id": "T7"})


def test_record_peer_replaces_identity_columns_and_keeps_facets_when_not_given(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("gr-rep", source="cli", user_id="old", chat_id="old", thread_id="t-old"))
    _peer(store_a, "gr-rep", "telegram:8", user_id="u8", chat_id="8", chat_type="dm", display_name="Eight", origin={"a": 1})
    # None on an identity column stores null (thread_id dropped); source becomes the peer's.
    row = store_a.sessions.get(SessionId("gr-rep"))
    assert (row.source, row.session_key, row.user_id, row.chat_id, row.chat_type, row.thread_id) == (
        SRC, "telegram:8", "u8", "8", "dm", None,
    )
    # A per-turn refresh with neither facet leaves both stored facets in place.
    _peer(store_a, "gr-rep", "telegram:8", user_id="u8", chat_id="8", chat_type="dm")
    found = _latest(store_a, "telegram:8")
    assert (found.session.display_name, found.session.origin) == ("Eight", {"a": 1})
    # A given facet replaces the stored one.
    _peer(store_a, "gr-rep", "telegram:8", display_name="Otto")
    found = _latest(store_a, "telegram:8")
    assert (found.session.display_name, found.session.origin) == ("Otto", {"a": 1})


def test_record_peer_with_ancestors_stamps_the_compression_chain_only(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("gr-a0"))
    store_a.sessions.end(SessionId("gr-a0"), "compression")
    store_a.sessions.create(_new("gr-a1", parent_session_id=SessionId("gr-a0")))
    store_a.sessions.end(SessionId("gr-a1"), "compression")
    store_a.sessions.create(_new("gr-a2", parent_session_id=SessionId("gr-a1")))
    # A fork child of the same parent is not on the chain, nor is a child whose parent did not compress.
    store_a.sessions.create(_new("gr-fork", parent_session_id=SessionId("gr-a1"), lineage_kind=SessionLineageKind.BRANCH))
    store_a.sessions.create(_new("gr-plain", parent_session_id=SessionId("gr-a2")))
    _peer(store_a, "gr-a2", "telegram:anc", chat_id="9", include_compression_ancestors=True)
    # The row and every compression ancestor carry the peer; siblings and non-compression children do not.
    for sid in ("gr-a0", "gr-a1", "gr-a2"):
        row = store_a.sessions.get(SessionId(sid))
        assert (row.session_key, row.chat_id) == ("telegram:anc", "9"), sid
    for sid in ("gr-fork", "gr-plain"):
        assert store_a.sessions.get(SessionId(sid)).session_key != "telegram:anc", sid
    # A branch row stops the walk: stamping from the fork reaches no ancestor.
    _peer(store_a, "gr-fork", "telegram:forked", include_compression_ancestors=True)
    assert store_a.sessions.get(SessionId("gr-a1")).session_key == "telegram:anc"
    # The ancestors form never inserts a missing row.
    _peer(store_a, "gr-absent", "telegram:anc", include_compression_ancestors=True)
    assert store_a.sessions.get(SessionId("gr-absent")) is None


def test_record_peer_is_scope_isolated(backend, scope_a, scope_b):
    store_a = backend.open(scope_a)
    store_b = backend.open(scope_b)
    _peer(store_a, "gr-iso", "telegram:iso")
    assert store_b.sessions.get(SessionId("gr-iso")) is None
    assert _latest(store_b, "telegram:iso") is None


def test_record_peer_reinserts_a_retired_id(backend, scope_a):
    store_a = backend.open(scope_a)
    _keyed(backend, scope_a, "gr-retired", "telegram:ret")
    assert store_a.sessions.delete(SessionId("gr-retired"), media_root=MEDIA_ROOT) == ("gr-retired",)
    # The self-heal insert runs no retirement check: the row comes back.
    _peer(store_a, "gr-retired", "telegram:ret")
    assert store_a.sessions.get(SessionId("gr-retired")).session_key == "telegram:ret"


# ── latest_session_for_peer ───────────────────────────────────────────────────


def test_latest_for_peer_reads_none_for_an_unknown_peer(backend, scope_a):
    assert _latest(backend.open(scope_a), "telegram:nobody") is None


def test_latest_for_peer_prefers_messages_then_recency(backend, scope_a):
    store_a = backend.open(scope_a)
    _keyed(backend, scope_a, "gr-p-old", "telegram:p")
    _tick()
    _keyed(backend, scope_a, "gr-p-mid", "telegram:p")
    _tick()
    _keyed(backend, scope_a, "gr-p-new", "telegram:p")
    # Without messages anywhere, the newest row wins.
    found = _latest(store_a, "telegram:p")
    assert found.session.session_id == "gr-p-new" and found.has_messages is False
    # A row with messages beats newer rows without them.
    store_a.messages.append(SessionId("gr-p-old"), _msg())
    found = _latest(store_a, "telegram:p")
    assert found.session.session_id == "gr-p-old" and found.has_messages is True
    assert found.session.session_key == "telegram:p"
    # Facets never recorded read None.
    assert (found.session.display_name, found.session.origin) == (None, None)
    # The activity stamp outranks started_at as recency.
    store_a.messages.append(SessionId("gr-p-mid"), _msg())
    store_a.sessions.touch_activity(SessionId("gr-p-old"), time.time() + 60)
    assert _latest(store_a, "telegram:p").session.session_id == "gr-p-old"


def test_latest_for_peer_admits_open_recoverable_and_handed_off_rows(backend, scope_a):
    store_a = backend.open(scope_a)
    _keyed(backend, scope_a, "gr-r-closed", "telegram:r")
    store_a.sessions.end(SessionId("gr-r-closed"), "compression")
    assert _latest(store_a, "telegram:r") is None
    _tick()
    _keyed(backend, scope_a, "gr-r-rec", "telegram:r")
    store_a.sessions.end(SessionId("gr-r-rec"), sorted(RECOVERABLE_END_REASONS)[0])
    assert _latest(store_a, "telegram:r").session.session_id == "gr-r-rec"
    # A row a completed handoff owns stays a candidate whatever end reason it carries.
    _tick()
    _keyed(backend, scope_a, "gr-r-handoff", "telegram:r")
    assert store_a.coordination.request_handoff(session_id=SessionId("gr-r-handoff"), platform="telegram")
    assert store_a.coordination.claim_handoff(session_id=SessionId("gr-r-handoff"))
    store_a.coordination.complete_handoff(session_id=SessionId("gr-r-handoff"))
    store_a.sessions.end(SessionId("gr-r-handoff"), "cli_close")
    assert _latest(store_a, "telegram:r").session.session_id == "gr-r-handoff"


def test_latest_for_peer_is_fenced_by_a_later_reset_boundary(backend, scope_a):
    store_a = backend.open(scope_a)
    _keyed(backend, scope_a, "gr-f-cand", "telegram:f")
    store_a.messages.append(SessionId("gr-f-cand"), _msg())
    _tick()
    _keyed(backend, scope_a, "gr-f-boundary", "telegram:f")
    store_a.sessions.end(SessionId("gr-f-boundary"), sorted(PEER_RESET_END_REASONS)[0])
    # A same-peer row that ended with a reset reason after the candidate's recency blocks it.
    assert _latest(store_a, "telegram:f") is None
    _tick()
    _keyed(backend, scope_a, "gr-f-after", "telegram:f")
    # A row started after the boundary is not fenced by it.
    assert _latest(store_a, "telegram:f").session.session_id == "gr-f-after"


def test_latest_for_peer_falls_back_to_the_peer_tuple_only_with_chat_id_and_type_and_messages(backend, scope_a):
    store_a = backend.open(scope_a)
    _keyed(backend, scope_a, "gr-t-1", "telegram:t-old-key", user_id="u", chat_id="42", chat_type="dm")
    # No chat_id and chat_type in the query: the tuple arm never runs.
    assert _latest(store_a, "telegram:t-new", user_id="u") is None
    # A tuple match without messages is not a candidate.
    assert _latest(store_a, "telegram:t-new", user_id="u", chat_id="42", chat_type="dm") is None
    store_a.messages.append(SessionId("gr-t-1"), _msg())
    found = _latest(store_a, "telegram:t-new", user_id="u", chat_id="42", chat_type="dm")
    assert found.session.session_id == "gr-t-1" and found.has_messages is True
    # Null columns equal an omitted query field; a differing thread does not match.
    assert _latest(store_a, "telegram:t-new", user_id="u", chat_id="42", chat_type="dm", thread_id="9") is None
    # Another source never matches the tuple.
    assert _latest(store_a, "x:t-new", source="discord", user_id="u", chat_id="42", chat_type="dm") is None


def test_latest_for_peer_tuple_arm_keeps_to_the_store_owners_rows(backend, scope_a):
    store_a = backend.open(scope_a)
    _keyed(backend, scope_a, "gr-t-foreign", "telegram:tf", chat_id="77", chat_type="dm", profile_name="someone-else")
    store_a.messages.append(SessionId("gr-t-foreign"), _msg())
    # A sibling profile's row with the same peer tuple is never adopted by the tuple arm.
    assert _latest(store_a, "telegram:tf-new", chat_id="77", chat_type="dm") is None
    # The exact-key arm carries no profile fence.
    assert _latest(store_a, "telegram:tf").session.session_id == "gr-t-foreign"
    _keyed(backend, scope_a, "gr-t-own", "telegram:to", chat_id="77", chat_type="dm")
    store_a.messages.append(SessionId("gr-t-own"), _msg())
    assert _latest(store_a, "telegram:tf-new", chat_id="77", chat_type="dm").session.session_id == "gr-t-own"
    # A sibling profile's reset boundary does not fence this profile's row.
    _tick()
    _keyed(backend, scope_a, "gr-t-foreign-reset", "telegram:tfr", chat_id="77", chat_type="dm", profile_name="someone-else")
    store_a.sessions.end(SessionId("gr-t-foreign-reset"), "session_reset")
    assert _latest(store_a, "telegram:tf-new", chat_id="77", chat_type="dm").session.session_id == "gr-t-own"


def test_latest_for_peer_still_finds_hidden_and_archived_rows(backend, scope_a):
    store_a = backend.open(scope_a)
    _keyed(backend, scope_a, "gr-h", "telegram:h")
    store_a.sessions.set_hidden(SessionId("gr-h"), True)
    store_a.sessions.set_archived(SessionId("gr-h"), True)
    found = _latest(store_a, "telegram:h")
    assert found is not None and found.session.session_id == "gr-h"


# ── session_for_origin ────────────────────────────────────────────────────────


def test_session_for_origin_matches_source_case_blind_and_open_keyed_rows_only(backend, scope_a):
    store_a = backend.open(scope_a)
    _keyed(backend, scope_a, "gr-o-1", "telegram:o", chat_id="100", user_id="alice")
    store_a.sessions.create(_new("gr-o-unkeyed", chat_id="100"))
    assert store_a.routing.session_for_origin(platform="Telegram", chat_id="100") == "gr-o-1"
    assert store_a.routing.session_for_origin(platform=SRC, chat_id="101") is None
    store_a.sessions.end(SessionId("gr-o-1"), "idle")
    # Ended rows never resolve; the unkeyed row is not a candidate.
    assert store_a.routing.session_for_origin(platform=SRC, chat_id="100") is None


def test_session_for_origin_filters_by_thread_and_prefers_the_newest(backend, scope_a):
    store_a = backend.open(scope_a)
    _keyed(backend, scope_a, "gr-o-t1", "telegram:o1", chat_id="200", thread_id="t1")
    _tick()
    _keyed(backend, scope_a, "gr-o-t2", "telegram:o2", chat_id="200", thread_id="t2")
    _keyed(backend, scope_a, "gr-o-none", "telegram:o3", chat_id="201")
    assert store_a.routing.session_for_origin(platform=SRC, chat_id="200", thread_id="t1") == "gr-o-t1"
    assert store_a.routing.session_for_origin(platform=SRC, chat_id="200", thread_id="t3") is None
    # An empty thread matches a row without one.
    assert store_a.routing.session_for_origin(platform=SRC, chat_id="201", thread_id="") == "gr-o-none"
    # No thread filter, no user ambiguity: the newest row wins.
    assert store_a.routing.session_for_origin(platform=SRC, chat_id="200") == "gr-o-t2"


def test_session_for_origin_disambiguates_by_sender(backend, scope_a):
    store_a = backend.open(scope_a)
    _keyed(backend, scope_a, "gr-o-a", "telegram:oa", chat_id="300", user_id="alice")
    _tick()
    _keyed(backend, scope_a, "gr-o-b", "telegram:ob", chat_id="300", user_id="bob")
    # Several distinct users and no sender given: None rather than a guess.
    assert store_a.routing.session_for_origin(platform=SRC, chat_id="300") is None
    # The exact sender wins regardless of recency.
    assert store_a.routing.session_for_origin(platform=SRC, chat_id="300", user_id="alice") == "gr-o-a"
    # A sender nobody matches, several candidates: None.
    assert store_a.routing.session_for_origin(platform=SRC, chat_id="300", user_id="carol") is None
    store_a.sessions.end(SessionId("gr-o-b"), "idle")
    # A single candidate resolves even for a non-matching sender.
    assert store_a.routing.session_for_origin(platform=SRC, chat_id="300", user_id="carol") == "gr-o-a"


def test_session_for_origin_folds_ascii_only_and_strips_users_for_ambiguity(backend, scope_a):
    store_a = backend.open(scope_a)
    _keyed(backend, scope_a, "gr-o-umlaut", "telegram:um", source="Ärger", chat_id="500")
    # Only ASCII letters fold (SQLite LOWER); a non-ASCII case variant does not match.
    assert store_a.routing.session_for_origin(platform="äRGER", chat_id="500") is None
    assert store_a.routing.session_for_origin(platform="ÄRGER", chat_id="500") == "gr-o-umlaut"
    _keyed(backend, scope_a, "gr-o-ws1", "telegram:ws1", chat_id="600", user_id="alice")
    _tick()
    _keyed(backend, scope_a, "gr-o-ws2", "telegram:ws2", chat_id="600", user_id=" alice ")
    # Users are whitespace-stripped for the ambiguity check, so these two are one user and the newest wins.
    assert store_a.routing.session_for_origin(platform=SRC, chat_id="600") == "gr-o-ws2"


def test_session_for_origin_is_scope_isolated(backend, scope_a, scope_b):
    _keyed(backend, scope_a, "gr-o-iso", "telegram:iso2", chat_id="400")
    assert backend.open(scope_b).routing.session_for_origin(platform=SRC, chat_id="400") is None


# ── orphaned_sessions + adopt_orphan ──────────────────────────────────────────


def _orphan_with_lineage_donor(backend, scope, donor: str = "gr-d", orphan: str = "gr-o") -> None:
    store = backend.open(scope)
    _keyed(backend, scope, donor, "telegram:d", chat_id="1", chat_type="dm", user_id="u1")
    store.messages.append(SessionId(donor), _msg(at=T0))
    store.sessions.end(SessionId(donor), "idle")
    store.sessions.create(_new(orphan, parent_session_id=SessionId(donor)))
    store.messages.append(SessionId(orphan), _msg(at=T0 + timedelta(seconds=5)))


def test_orphan_report_by_lineage(backend, scope_a):
    store_a = backend.open(scope_a)
    _orphan_with_lineage_donor(backend, scope_a)
    # Message-less unkeyed rows, tool sources and fork children are not orphans.
    store_a.sessions.create(_new("gr-silent"))
    store_a.sessions.create(_new("gr-tool", source="tool"))
    store_a.messages.append(SessionId("gr-tool"), _msg())
    store_a.sessions.create(_new("gr-branch", parent_session_id=SessionId("gr-d"), lineage_kind=SessionLineageKind.BRANCH))
    store_a.messages.append(SessionId("gr-branch"), _msg())
    reports = store_a.routing.orphaned_sessions()
    assert [r.orphan_id for r in reports] == ["gr-o"]
    report = reports[0]
    assert (report.donor_id, report.session_key, report.evidence, report.adoptable) == ("gr-d", "telegram:d", "lineage", True)
    assert report.source == SRC and report.message_count == 1 and report.reason == ""
    # Recency is the newest stored row when any exists, even one stamped before the row started.
    assert report.last_active == (T0 + timedelta(seconds=5)).timestamp()
    assert isinstance(report.started_at, float)
    # Report only; the orphan is still unkeyed.
    assert store_a.sessions.get(SessionId("gr-o")).session_key is None


def test_orphan_report_counts_imported_rows(backend, scope_a):
    store_a = backend.open(scope_a)
    payload = {
        "id": "gr-imported", "source": SRC, "started_at": T0.timestamp(),
        "messages": [
            {"role": "user", "content": "a", "timestamp": T0.timestamp()},
            {"role": "assistant", "content": "b", "timestamp": (T0 + timedelta(seconds=1)).timestamp()},
        ],
    }
    result = store_a.maintenance.import_sessions([payload])
    assert result.ok and result.imported == 1
    (report,) = store_a.routing.orphaned_sessions()
    # Imported rows are stored rows: every row counted and the newest row as recency.
    assert (report.orphan_id, report.message_count, report.last_active) == ("gr-imported", 2, (T0 + timedelta(seconds=1)).timestamp())


def test_orphan_with_an_unkeyed_or_missing_parent_is_not_adoptable(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.create(_new("gr-up"))
    store_a.sessions.create(_new("gr-uo", parent_session_id=SessionId("gr-up")))
    store_a.messages.append(SessionId("gr-uo"), _msg(at=datetime.now(UTC) + timedelta(seconds=5)))
    (report,) = store_a.routing.orphaned_sessions()
    assert (report.orphan_id, report.donor_id, report.evidence, report.adoptable) == ("gr-uo", None, "", False)
    assert report.reason == "parent session carries no gateway identity of this source"
    # A row with a parent never falls back to the contiguity window, even with a keyed row right before it.
    _keyed(backend, scope_a, "gr-near", "telegram:near")
    store_a.messages.append(SessionId("gr-near"), _msg(at=datetime.now(UTC) - timedelta(seconds=1)))
    report = next(r for r in store_a.routing.orphaned_sessions() if r.orphan_id == "gr-uo")
    assert (report.donor_id, report.adoptable) == (None, False)


def test_orphan_report_by_contiguity_window(backend, scope_a):
    store_a = backend.open(scope_a)
    now = datetime.now(UTC)
    _keyed(backend, scope_a, "gr-cd", "telegram:cd", user_id="u1")
    store_a.messages.append(SessionId("gr-cd"), _msg(at=now + timedelta(seconds=1)))
    _keyed(backend, scope_a, "gr-cd-other-user", "telegram:cx", user_id="u2")
    store_a.messages.append(SessionId("gr-cd-other-user"), _msg(at=now + timedelta(seconds=1)))
    _keyed(backend, scope_a, "gr-cd-foreign", "telegram:cf", user_id="u1", profile_name="someone-else")
    store_a.messages.append(SessionId("gr-cd-foreign"), _msg(at=now + timedelta(seconds=1)))
    _tick()
    store_a.sessions.create(_new("gr-co", user_id="u1"))
    store_a.messages.append(SessionId("gr-co"), _msg(at=now + timedelta(seconds=30)))
    (report,) = store_a.routing.orphaned_sessions()
    # Exactly one user-compatible keyed row of the orphan's profile fell quiet inside the window: contiguity donor.
    assert (report.donor_id, report.session_key, report.evidence, report.adoptable) == ("gr-cd", "telegram:cd", "contiguity", True)
    # A zero window admits nobody, with its reason text.
    (report,) = store_a.routing.orphaned_sessions(max_gap_s=0)
    assert (report.donor_id, report.evidence, report.adoptable) == (None, "", False)
    assert report.reason == "no keyed predecessor fell quiet within 0s of this session's start"
    # Two matching predecessors are ambiguous, never guessed.
    _keyed(backend, scope_a, "gr-cd2", "telegram:cd2")
    store_a.messages.append(SessionId("gr-cd2"), _msg(at=now + timedelta(seconds=2)))
    report = next(r for r in store_a.routing.orphaned_sessions() if r.orphan_id == "gr-co")
    assert (report.donor_id, report.adoptable) == (None, False)
    assert report.reason == "ambiguous: more than one keyed predecessor matches this window"


def test_orphans_claiming_one_donor_are_all_demoted_and_reports_are_oldest_first(backend, scope_a):
    store_a = backend.open(scope_a)
    _orphan_with_lineage_donor(backend, scope_a, orphan="gr-o1")
    _tick()
    store_a.sessions.create(_new("gr-o2", parent_session_id=SessionId("gr-d")))
    store_a.messages.append(SessionId("gr-o2"), _msg())
    reports = store_a.routing.orphaned_sessions()
    assert [r.orphan_id for r in reports] == ["gr-o1", "gr-o2"]
    # The donor is named but contested; neither orphan is adoptable.
    for report in reports:
        assert (report.donor_id, report.evidence, report.adoptable) == ("gr-d", "lineage", False)
        assert report.reason == "ambiguous: more than one unkeyed session claims this predecessor"


def test_adopt_orphan_stamps_fills_nulls_and_retires_the_donor(backend, scope_a):
    store_a = backend.open(scope_a)
    _orphan_with_lineage_donor(backend, scope_a)
    _peer(store_a, "gr-d", "telegram:d", chat_id="1", chat_type="dm", user_id="u1", display_name="Donor", origin={"o": 1})
    donor_before = store_a.sessions.get(SessionId("gr-d"))
    assert store_a.routing.adopt_orphan("gr-o", "gr-d") is True
    orphan = store_a.sessions.get(SessionId("gr-o"))
    # The orphan carries the donor's key and its null addressing and facets filled from the donor; the parent stays.
    assert (orphan.session_key, orphan.chat_id, orphan.chat_type, orphan.user_id, orphan.parent_session_id) == ("telegram:d", "1", "dm", "u1", "gr-d")
    assert (orphan.display_name, orphan.origin) == ("Donor", {"o": 1})
    donor = store_a.sessions.get(SessionId("gr-d"))
    # An already-ended donor keeps its stamp; the reason becomes superseded_by_repair.
    assert donor.end_reason == END_REASON_SUPERSEDED_BY_REPAIR and donor.ended_at == donor_before.ended_at
    assert END_REASON_SUPERSEDED_BY_REPAIR not in RECOVERABLE_END_REASONS
    # Recovery for the peer now lands on the orphan (the donor is not recoverable).
    assert _latest(store_a, "telegram:d").session.session_id == "gr-o"
    assert store_a.routing.orphaned_sessions() == ()


def test_adopt_orphan_keeps_the_orphans_own_values_and_ends_a_live_donor(backend, scope_a):
    store_a = backend.open(scope_a)
    _keyed(backend, scope_a, "gr-ld", "telegram:ld", chat_id="1", chat_type="dm", user_id="u1")
    store_a.sessions.create(_new("gr-lo", chat_id="own-chat", user_id="own-user"))
    store_a.messages.append(SessionId("gr-lo"), _msg())
    assert store_a.routing.adopt_orphan("gr-lo", "gr-ld") is True
    orphan = store_a.sessions.get(SessionId("gr-lo"))
    assert (orphan.session_key, orphan.chat_id, orphan.chat_type, orphan.user_id, orphan.parent_session_id) == ("telegram:ld", "own-chat", "dm", "own-user", "gr-ld")
    donor = store_a.sessions.get(SessionId("gr-ld"))
    assert donor.ended_at is not None and donor.end_reason == END_REASON_SUPERSEDED_BY_REPAIR


def test_adopt_orphan_refusals_write_nothing(backend, scope_a):
    store_a = backend.open(scope_a)
    _orphan_with_lineage_donor(backend, scope_a)
    store_a.sessions.create(_new("gr-other-src", source="discord"))
    _keyed(backend, scope_a, "gr-keyed", "telegram:k")
    _keyed(backend, scope_a, "gr-foreign", "telegram:fk", profile_name="someone-else")
    # Same id, missing rows, unkeyed donor, keyed orphan, differing sources, another profile's donor: False.
    assert store_a.routing.adopt_orphan("gr-o", "gr-o") is False
    assert store_a.routing.adopt_orphan("gr-o", "gr-ghost") is False
    assert store_a.routing.adopt_orphan("gr-ghost", "gr-d") is False
    assert store_a.routing.adopt_orphan("gr-o", "gr-other-src") is False
    assert store_a.routing.adopt_orphan("gr-keyed", "gr-d") is False
    assert store_a.routing.adopt_orphan("gr-other-src", "gr-d") is False
    assert store_a.routing.adopt_orphan("gr-o", "gr-foreign") is False
    orphan = store_a.sessions.get(SessionId("gr-o"))
    donor = store_a.sessions.get(SessionId("gr-d"))
    assert orphan.session_key is None and donor.end_reason == "idle"
    assert store_a.sessions.get(SessionId("gr-foreign")).ended_at is None
    # A donor deleted between report and adopt: False, orphan untouched.
    assert store_a.sessions.delete(SessionId("gr-d"), media_root=MEDIA_ROOT) == ("gr-d",)
    assert store_a.routing.adopt_orphan("gr-o", "gr-d") is False
    assert store_a.sessions.get(SessionId("gr-o")).session_key is None


def test_two_adopters_for_one_orphan_exactly_one_wins(backend, scope_a):
    store_a = backend.open(scope_a)
    _keyed(backend, scope_a, "gr-race-d1", "telegram:rd1", chat_id="1", chat_type="dm")
    _keyed(backend, scope_a, "gr-race-d2", "telegram:rd2", chat_id="2", chat_type="dm")
    store_a.sessions.create(_new("gr-race-o"))
    store_a.messages.append(SessionId("gr-race-o"), _msg())
    outcomes = _race(
        lambda: store_a.routing.adopt_orphan("gr-race-o", "gr-race-d1"),
        lambda: store_a.routing.adopt_orphan("gr-race-o", "gr-race-d2"),
    )
    # Exactly one True; the orphan carries the winner's key and chat; only the winner's donor is retired.
    assert sorted(outcomes) == [False, True]
    winner, loser = ("gr-race-d1", "gr-race-d2") if outcomes[0] else ("gr-race-d2", "gr-race-d1")
    orphan = store_a.sessions.get(SessionId("gr-race-o"))
    winner_row = store_a.sessions.get(SessionId(winner))
    loser_row = store_a.sessions.get(SessionId(loser))
    assert (orphan.session_key, orphan.chat_id) == (winner_row.session_key, winner_row.chat_id)
    assert winner_row.end_reason == END_REASON_SUPERSEDED_BY_REPAIR
    assert loser_row.ended_at is None and loser_row.session_key is not None


# ── never_active_keyed_sessions + prune_never_active ─────────────────────────


def test_never_active_lists_only_keyed_open_silent_unpinned_rows(backend, scope_a):
    store_a = backend.open(scope_a)
    _keyed(backend, scope_a, "gr-na-1", "telegram:na1", chat_id="1", chat_type="dm", user_id="u")
    _tick()
    _keyed(backend, scope_a, "gr-na-2", "telegram:na2")
    store_a.sessions.create(_new("gr-na-unkeyed"))
    _keyed(backend, scope_a, "gr-na-msg", "telegram:na3")
    store_a.messages.append(SessionId("gr-na-msg"), _msg())
    _keyed(backend, scope_a, "gr-na-usage", "telegram:na4")
    store_a.usage.queue_usage(SessionId("gr-na-usage"), UsageDelta(api_call_count=1))
    assert store_a.usage.flush_usage()
    _keyed(backend, scope_a, "gr-na-title", "telegram:na5")
    store_a.sessions.set_title(SessionId("gr-na-title"), "kept")
    _keyed(backend, scope_a, "gr-na-pinned", "telegram:na6")
    store_a.sessions.set_pinned(SessionId("gr-na-pinned"), True)
    _keyed(backend, scope_a, "gr-na-archived", "telegram:na7")
    store_a.sessions.set_archived(SessionId("gr-na-archived"), True)
    _keyed(backend, scope_a, "gr-na-ended", "telegram:na8")
    store_a.sessions.end(SessionId("gr-na-ended"), "idle")
    _keyed(backend, scope_a, "gr-na-active", "telegram:na9")
    store_a.sessions.touch_activity(SessionId("gr-na-active"), time.time())
    _tick()
    rows = store_a.routing.never_active_keyed_sessions(older_than_days=FRESH)
    # The two silent keyed open rows only, oldest first, with their identity columns.
    assert [r.session_id for r in rows] == ["gr-na-1", "gr-na-2"]
    assert (rows[0].session_key, rows[0].source, rows[0].chat_id, rows[0].chat_type, rows[0].user_id) == ("telegram:na1", SRC, "1", "dm", "u")
    assert isinstance(rows[0].started_at, float) and rows[0].started_at < rows[1].started_at
    # A window the rows have not aged past lists nothing.
    assert store_a.routing.never_active_keyed_sessions(older_than_days=1) == ()


@pytest.mark.parametrize("operation", ["never_active_keyed_sessions", "prune_never_active"])
def test_never_active_refuses_a_negative_retention(backend, scope_a, operation):
    call = getattr(backend.open(scope_a).routing, operation)
    media = {"media_root": MEDIA_ROOT} if operation == "prune_never_active" else {}
    with pytest.raises(ValueError, match=r"older_than_days must be >= 0, got -0\.5: a negative retention builds a future cutoff"):
        call(older_than_days=-0.5, **media)


def test_prune_never_active_removes_routing_entries_in_every_routing_scope_then_the_rows(backend, scope_a, scope_b):
    store_a = backend.open(scope_a)
    store_b = backend.open(scope_b)
    _keyed(backend, scope_a, "gr-pr-1", "telegram:pr1")
    _keyed(backend, scope_a, "gr-pr-2", "telegram:pr2")
    _keyed(backend, scope_a, "gr-pr-live", "telegram:pr3")
    store_a.messages.append(SessionId("gr-pr-live"), _msg())
    _keyed(backend, scope_b, "gr-pr-other-scope", "telegram:pr4")
    store_a.routing.upsert_routing_entry("telegram:pr1", _entry("telegram:pr1", "gr-pr-1"), routing_scope="ns-a")
    store_a.routing.upsert_routing_entry("telegram:pr2", _entry("telegram:pr2", "gr-pr-2"), routing_scope="ns-b")
    store_a.routing.upsert_routing_entry("telegram:pr3", _entry("telegram:pr3", "gr-pr-live"), routing_scope="ns-a")
    result = store_a.routing.prune_never_active(older_than_days=FRESH, media_root=MEDIA_ROOT)
    assert result == NeverActivePrune(
        sessions_deleted=2, routing_entries_deleted=2, sessions_skipped=0, deleted_ids=("gr-pr-1", "gr-pr-2"), removed_ids=("gr-pr-1", "gr-pr-2")
    )
    assert store_a.sessions.get(SessionId("gr-pr-1")) is None
    assert store_a.sessions.get(SessionId("gr-pr-2")) is None
    assert store_a.sessions.get(SessionId("gr-pr-live")).session_key == "telegram:pr3"
    assert set(store_a.routing.load_routing_entries(routing_scope="ns-a")) == {"telegram:pr3"}
    assert store_a.routing.load_routing_entries(routing_scope="ns-b") == {}
    # Scope isolation; a second prune finds nothing.
    assert store_b.sessions.get(SessionId("gr-pr-other-scope")) is not None
    assert store_a.routing.prune_never_active(older_than_days=FRESH, media_root=MEDIA_ROOT) == NeverActivePrune(0, 0, 0, (), ())


def test_prune_never_active_skips_a_row_under_a_live_turn_lease(backend, scope_a):
    store_a = backend.open(scope_a)
    _keyed(backend, scope_a, "gr-lease", "telegram:lease")
    _keyed(backend, scope_a, "gr-free", "telegram:free")
    store_a.routing.upsert_routing_entry("telegram:lease", _entry("telegram:lease", "gr-lease"))
    assert store_a.coordination.try_acquire_turn_lease(session_id=SessionId("gr-lease"), holder="live", ttl_seconds=300.0)
    result = store_a.routing.prune_never_active(older_than_days=FRESH, media_root=MEDIA_ROOT)
    # The leased row and its routing entry stay; it is counted skipped and left out of both id lists.
    assert result == NeverActivePrune(sessions_deleted=1, routing_entries_deleted=0, sessions_skipped=1, deleted_ids=("gr-free",), removed_ids=("gr-free",))
    assert store_a.sessions.get(SessionId("gr-lease")) is not None
    assert set(store_a.routing.load_routing_entries()) == {"telegram:lease"}


def _plant_artifacts(transcript_dir, *session_ids: str) -> None:
    transcript_dir.mkdir(exist_ok=True)
    for sid in session_ids:
        for name in (f"{sid}.json", f"{sid}.jsonl", f"request_dump_{sid}_1.json"):
            (transcript_dir / name).write_text("{}")


def test_prune_never_active_removes_the_artifacts_of_every_id_it_deleted(backend, scope_a, tmp_path):
    store_a = backend.open(scope_a)
    files = tmp_path / "sessions"
    _keyed(backend, scope_a, "gr-art-1", "telegram:art1")
    store_a.sessions.create(
        _new("gr-art-child", parent_session_id=SessionId("gr-art-1"), lineage_kind=SessionLineageKind.DELEGATE)
    )
    store_a.messages.append(SessionId("gr-art-child"), _msg())
    _keyed(backend, scope_a, "gr-art-live", "telegram:art2")
    store_a.messages.append(SessionId("gr-art-live"), _msg())
    _plant_artifacts(files, "gr-art-1", "gr-art-child", "gr-art-live", "gr-art-stranger")
    result = store_a.routing.prune_never_active(older_than_days=FRESH, artifacts=SessionArtifacts(files), media_root=MEDIA_ROOT)
    # The candidate and its cascaded delegate child are gone with their transcript pair and request dumps; the active row and a file no store row names keep theirs.
    assert (result.sessions_deleted, result.deleted_ids, result.removed_ids) == (1, ("gr-art-1",), ("gr-art-1", "gr-art-child"))
    assert store_a.sessions.get(SessionId("gr-art-child")) is None
    assert sorted(p.name for p in files.iterdir()) == [
        "gr-art-live.json", "gr-art-live.jsonl", "gr-art-stranger.json", "gr-art-stranger.jsonl",
        "request_dump_gr-art-live_1.json", "request_dump_gr-art-stranger_1.json",
    ]


def test_prune_never_active_reports_the_delegate_children_its_delete_cascades_to(backend, scope_a):
    store_a = backend.open(scope_a)
    _keyed(backend, scope_a, "gr-parent", "telegram:parent")
    store_a.sessions.create(
        _new("gr-delegate", parent_session_id=SessionId("gr-parent"), lineage_kind=SessionLineageKind.DELEGATE)
    )
    store_a.messages.append(SessionId("gr-delegate"), _msg())
    result = store_a.routing.prune_never_active(older_than_days=FRESH, media_root=MEDIA_ROOT)
    # The never-active parent is the one candidate and the only deleted id; its delete cascades to the delegate child, which the removed ids carry with it.
    assert result.sessions_deleted == 1
    assert result.deleted_ids == ("gr-parent",)
    assert result.removed_ids == ("gr-delegate", "gr-parent")
    assert store_a.sessions.get(SessionId("gr-parent")) is None
    assert store_a.sessions.get(SessionId("gr-delegate")) is None


def test_prune_never_active_without_artifacts_touches_no_file(backend, scope_a, tmp_path):
    store_a = backend.open(scope_a)
    files = tmp_path / "sessions"
    _keyed(backend, scope_a, "gr-noart", "telegram:noart")
    _plant_artifacts(files, "gr-noart")
    assert store_a.routing.prune_never_active(older_than_days=FRESH, media_root=MEDIA_ROOT).sessions_deleted == 1
    assert len(list(files.iterdir())) == 3


def _arm_activity_before_the_delete(store, monkeypatch) -> None:
    """Interpose on the ``sessions.delete_many`` the prune performs so a message lands on each target between the candidate read and the delete."""
    real = store.sessions.delete_many

    def _delete_many(session_ids, **kwargs):
        for session_id in session_ids:
            store.messages.append(session_id, _msg("late"))
        return real(session_ids, **kwargs)

    monkeypatch.setattr(store.sessions, "delete_many", _delete_many)


def test_prune_deletes_a_candidate_that_became_active_meanwhile(backend, scope_a, monkeypatch):
    store_a = backend.open(scope_a)
    _keyed(backend, scope_a, "gr-snap", "telegram:snap")
    _keyed(backend, scope_a, "gr-snap-live", "telegram:snap2")
    store_a.messages.append(SessionId("gr-snap-live"), _msg())
    _arm_activity_before_the_delete(store_a, monkeypatch)
    result = store_a.routing.prune_never_active(older_than_days=FRESH, media_root=MEDIA_ROOT)
    # The delete does not re-check activity (it deletes the candidate list it read); a row active before the call is never a candidate.
    assert result.sessions_deleted == 1
    assert store_a.sessions.get(SessionId("gr-snap")) is None
    assert store_a.sessions.get(SessionId("gr-snap-live")).session_key == "telegram:snap2"


def test_prune_leaves_the_route_of_a_candidate_gone_meanwhile_to_the_sweep(backend, scope_a, monkeypatch):
    store_a = backend.open(scope_a)
    _keyed(backend, scope_a, "gr-gone", "telegram:gone")
    _keyed(backend, scope_a, "gr-stale", "telegram:stale")
    store_a.routing.upsert_routing_entry("telegram:gone", _entry("telegram:gone", "gr-gone"), routing_scope="ns")
    real_list = store_a.routing.never_active_keyed_sessions

    def _list_then_discard(**kwargs):
        rows = real_list(**kwargs)
        # Removed without retirement between the candidate read and the delete.
        store_a.sessions.discard_unadmitted(SessionId("gr-gone"), media_root=MEDIA_ROOT)
        return rows

    monkeypatch.setattr(store_a.routing, "never_active_keyed_sessions", _list_then_discard)
    result = store_a.routing.prune_never_active(older_than_days=FRESH, media_root=MEDIA_ROOT)
    # CONTRACT: the prune ends with its delete, so the entry of a candidate that went another way stays until the caller sweeps the deleted ids.
    assert result.deleted_ids == ("gr-gone", "gr-stale")
    assert set(store_a.routing.load_routing_entries(routing_scope="ns")) == {"telegram:gone"}
    assert store_a.routing.sweep_routing_entries_for_sessions(result.deleted_ids) == 1
    assert store_a.routing.load_routing_entries(routing_scope="ns") == {}


def test_sweep_routing_entries_drops_every_entry_naming_an_id_in_every_routing_scope(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.routing.upsert_routing_entry("telegram:sw1", _entry("telegram:sw1", "gr-sw"), routing_scope="ns-a")
    store_a.routing.upsert_routing_entry("telegram:sw2", _entry("telegram:sw2", "gr-sw"), routing_scope="ns-b")
    store_a.routing.upsert_routing_entry("telegram:sw3", _entry("telegram:sw3", "gr-sw-keep"), routing_scope="ns-a")
    # CONTRACT: every entry naming one of the ids goes, in any routing scope, and the count says how many; others stay.
    assert store_a.routing.sweep_routing_entries_for_sessions(["gr-sw", "gr-sw-ghost"]) == 2
    assert set(store_a.routing.load_routing_entries(routing_scope="ns-a")) == {"telegram:sw3"}
    assert store_a.routing.load_routing_entries(routing_scope="ns-b") == {}
    assert store_a.routing.sweep_routing_entries_for_sessions([]) == 0


def test_two_prunes_over_one_store_remove_every_candidate_once(backend, scope_a):
    store_a = backend.open(scope_a)
    ids = [f"gr-pp-{i}" for i in range(6)]
    for i, sid in enumerate(ids):
        _keyed(backend, scope_a, sid, f"telegram:pp{i}")
        store_a.routing.upsert_routing_entry(f"telegram:pp{i}", _entry(f"telegram:pp{i}", sid), routing_scope="ns")
    results = _race(
        lambda: store_a.routing.prune_never_active(older_than_days=FRESH, media_root=MEDIA_ROOT),
        lambda: store_a.routing.prune_never_active(older_than_days=FRESH, media_root=MEDIA_ROOT),
    )
    # Every candidate is gone and each session deletion is counted once across the two contenders.
    for sid in ids:
        assert store_a.sessions.get(SessionId(sid)) is None
    assert sum(r.sessions_deleted for r in results) == len(ids)
    # The routing count is the entries a contender read before its delete; the one that saw the full candidate list counted them all.
    assert max(r.routing_entries_deleted for r in results) == len(ids)
    assert all(0 <= r.routing_entries_deleted <= len(ids) for r in results)
    assert store_a.routing.load_routing_entries(routing_scope="ns") == {}


# ── prepare_topic_storage ─────────────────────────────────────────────────────


def test_prepare_topic_storage_is_idempotent_and_sets_the_schema_version(backend, scope_a, scope_b):
    store_a = backend.open(scope_a)
    store_b = backend.open(scope_b)
    assert store_a.state_meta.get(TOPIC_SCHEMA_META_KEY) is None
    store_a.routing.prepare_topic_storage()
    assert store_a.state_meta.get(TOPIC_SCHEMA_META_KEY) == "3"
    store_a.routing.prepare_topic_storage()
    assert store_a.state_meta.get(TOPIC_SCHEMA_META_KEY) == "3"
    assert store_b.state_meta.get(TOPIC_SCHEMA_META_KEY) is None
    # The topic surface works after preparation; an implicit preparation by bind_topic sets the same version.
    store_a.sessions.create(_new("gr-topic"))
    store_a.routing.bind_topic(chat_id="c1", thread_id="t1", user_id="u", session_key="tg:c1:t1", session_id="gr-topic")
    assert store_a.routing.get_topic_binding(chat_id="c1", thread_id="t1").session_id == "gr-topic"
    store_b.sessions.create(_new("gr-topic"))
    store_b.routing.bind_topic(chat_id="c1", thread_id="t1", user_id="u", session_key="tg:c1:t1", session_id="gr-topic")
    assert store_b.state_meta.get(TOPIC_SCHEMA_META_KEY) == "3"


# ── gateway_runtime (pure projection) ────────────────────────────────────────


def test_gateway_runtime_precedence():
    nested = {"gateway_runtime": {"provider": "openrouter", "base_url": "https://x", "api_mode": None}}
    assert gateway_runtime(nested) == GatewayRuntime(provider="openrouter", base_url="https://x")
    # Nested with a provider wins over top-level keys and the billing fallback.
    assert gateway_runtime({**nested, "provider": "top"}, billing_provider="bill") == GatewayRuntime(provider="openrouter", base_url="https://x")
    # Top-level truthy keys next.
    assert gateway_runtime({"provider": "top", "base_url": "", "api_mode": "chat"}) == GatewayRuntime(provider="top", api_mode="chat")
    # The billing provider fills a missing top-level provider, also beside other top-level keys.
    assert gateway_runtime({}, billing_provider=" nous ") == GatewayRuntime(provider="nous")
    assert gateway_runtime({"base_url": "https://z"}, billing_provider="nous") == GatewayRuntime(provider="nous", base_url="https://z")
    assert gateway_runtime({"provider": "top"}, billing_provider="nous") == GatewayRuntime(provider="top")
    # Bare buckets are not identities.
    for bare in BARE_BILLING_PROVIDERS:
        assert gateway_runtime(None, billing_provider=bare.upper()) == GatewayRuntime()
    # A bare custom bucket with its served endpoint surfaces as provider custom with that endpoint.
    assert gateway_runtime({}, billing_provider="custom", billing_base_url=" https://c ") == GatewayRuntime(provider="custom", base_url="https://c")
    assert gateway_runtime({"base_url": "https://top"}, billing_provider="custom", billing_base_url="https://c") == GatewayRuntime(provider="custom", base_url="https://top")
    assert gateway_runtime({"provider": "top"}, billing_provider="custom", billing_base_url="https://c") == GatewayRuntime(provider="top")
    assert gateway_runtime({}, billing_provider="auto", billing_base_url="https://c") == GatewayRuntime()
    # A provider-less nested object contributes its remaining members last.
    assert gateway_runtime({"gateway_runtime": {"base_url": "https://y"}}) == GatewayRuntime(base_url="https://y")
    assert gateway_runtime({"gateway_runtime": {"base_url": "https://y"}}, billing_provider="nous") == GatewayRuntime(provider="nous")
    # The nested arm keeps empty strings (only None members are dropped).
    assert gateway_runtime({"gateway_runtime": {"provider": "p", "base_url": "", "api_mode": None}}) == GatewayRuntime(provider="p", base_url="")
    # Non-string members read None; the empty route is empty.
    assert gateway_runtime({"provider": 3, "gateway_runtime": "nope"}) == GatewayRuntime()
    assert gateway_runtime(None).is_empty
