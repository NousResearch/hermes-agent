"""Contract cases for the text of ``RewindRefused``: at every refusal site of ``rewind_user_turn`` the error's message is the store's own refusal text and its ``reason`` the contract reason.

The warm-history refusals are reached through the public verb. The refusals no public call can provoke (a read that races a write, a durable turn without row identity, a carrier rewind whose handoff did not come back) are reached by interposing on the plan's durable reads or on the pinned write of each backend.
"""

from __future__ import annotations

from dataclasses import is_dataclass, replace
from datetime import UTC, datetime
from typing import Any, Callable

import pytest

from persistence import RewindRefused
from persistence.conversation.messages import ConversationRequest, NewMessage
from persistence.conversation.sessions import NewSession
from persistence.core.identity import SessionId

S = SessionId("rewind-text-s1")
HISTORY_CHANGED = "session history changed before the rewind could be persisted"
ACTIVE_CHANGED = "active transcript changed before the rewind could be persisted"


def ts(i: int) -> float:
    return datetime(2026, 9, 11, 10, 0, i, tzinfo=UTC).timestamp()


def _seed(backend, scope, first_ask: Any = "u1"):
    """u1 a1 u2 a2 on a fresh session; returns the repository and the row ids."""
    store = backend.open(scope)
    store.sessions.create(NewSession(session_id=S, source="tui"))
    repo = store.messages
    rows = [("user", first_ask), ("assistant", "a1"), ("user", "u2"), ("assistant", "a2")]
    return repo, [repo.append(S, NewMessage(role=role, content=content, timestamp=ts(i))) for i, (role, content) in enumerate(rows)]


def _interpose_plan_reads(backend_name: str, repo, monkeypatch, edit: Callable[[list, bool], list]) -> None:
    """Route the rewind plan's two durable reads through ``edit(rows, repaired)``."""
    if backend_name == "sqlite":
        from persistence.sqlite.conversation._store import rewind as module

        real = module.get_messages_as_conversation

        def read(holder, sid, **kwargs):
            return edit(real(holder, sid, **kwargs), bool(kwargs.get("repair_alternation")))

        monkeypatch.setattr(module, "get_messages_as_conversation", read)
    else:
        real_conversation = repo.read_conversation

        def read_conversation(sid, request=ConversationRequest()):
            return edit(list(real_conversation(sid, request)), request.repair_alternation)

        monkeypatch.setattr(repo, "read_conversation", read_conversation)


def _without_row_ids(rows: list) -> list:
    return [replace(m, row_id=None) if is_dataclass(m) else {k: v for k, v in m.items() if k != "_row_id"} for m in rows]


def _refusal(repo, **kwargs) -> RewindRefused:
    with pytest.raises(RewindRefused) as refused:
        repo.rewind_user_turn(S, -1, **kwargs)
    return refused.value


def test_warm_history_refusals_carry_the_history_text(backend, scope_a):
    """CONTRACT: a warm history with another user-turn count, or another addressed turn, is history_changed with the store's text."""
    repo, _ids = _seed(backend, scope_a)
    warm = [{"role": "user", "content": "u1"}, {"role": "assistant", "content": "a1"}, {"role": "user", "content": "u2"}]
    for disagreeing in (warm[:2], [*warm[:2], {"role": "user", "content": "u2 edited"}]):
        refused = _refusal(repo, warm_history=disagreeing)
        assert (refused.reason, str(refused)) == ("history_changed", HISTORY_CHANGED)


def test_a_target_missing_from_the_stored_read_carries_the_history_text(backend_name, backend, scope_a, monkeypatch):
    """CONTRACT: a target the stored (unrepaired) projection no longer holds is history_changed with the store's text."""
    repo, _ids = _seed(backend, scope_a)
    _interpose_plan_reads(backend_name, repo, monkeypatch, lambda rows, repaired: rows if repaired else [])
    refused = _refusal(repo)
    assert (refused.reason, str(refused)) == ("history_changed", HISTORY_CHANGED)


def test_a_target_without_row_identity_carries_the_identity_text(backend_name, backend, scope_a, monkeypatch):
    """CONTRACT: a durable target with no row identity is target_unidentified with the store's text."""
    repo, _ids = _seed(backend, scope_a)
    _interpose_plan_reads(backend_name, repo, monkeypatch, lambda rows, repaired: _without_row_ids(rows) if repaired else rows)
    refused = _refusal(repo)
    assert (refused.reason, str(refused)) == ("target_unidentified", "rewind target has no durable row identity")


def test_an_active_set_changed_before_the_write_carries_the_active_transcript_text(backend_name, backend, scope_a, monkeypatch):
    """CONTRACT: the write's active-id pin refusing a row appended after the plan is history_changed with the write's own text, which differs from the plan's."""
    repo, _ids = _seed(backend, scope_a)

    def append_after_plan(rows, repaired):
        if repaired:
            repo.append(S, NewMessage(role="user", content="u3", timestamp=ts(9)))
        return rows

    _interpose_plan_reads(backend_name, repo, monkeypatch, append_after_plan)
    refused = _refusal(repo)
    assert (refused.reason, str(refused)) == ("history_changed", ACTIVE_CHANGED)


def test_a_target_edited_before_the_write_carries_the_target_text(backend_name, backend, scope_a, monkeypatch):
    """CONTRACT: a target whose live payload changed between the plan and the write is target_changed with the store's text."""
    repo, ids = _seed(backend, scope_a)

    def edit_after_plan(rows, repaired):
        if repaired:
            assert repo.set_user_message_content(S, ids[2], "u2 edited") is True
        return rows

    _interpose_plan_reads(backend_name, repo, monkeypatch, edit_after_plan)
    refused = _refusal(repo)
    assert (refused.reason, str(refused)) == ("target_changed", "rewind target changed before it could be persisted")


def test_a_carrier_rewind_without_its_handoff_carries_the_handoff_text(backend_name, backend, scope_a, monkeypatch):
    """CONTRACT: a composite-carrier rewind whose handoff scaffold did not come back is handoff_lost with the store's text."""
    from tests.contracts._reference.conversation._turn_policy import composite_carrier

    store = backend.open(scope_a)
    store.sessions.create(NewSession(session_id=S, source="tui"))
    repo = store.messages
    repo.append(S, NewMessage(role="user", content="older ask", timestamp=ts(0)))
    repo.append(S, NewMessage(role="assistant", content="older reply", timestamp=ts(1)))
    repo.append(S, NewMessage(role="user", content=composite_carrier("old task", "REAL ASK"), timestamp=ts(2)))
    repo.append(S, NewMessage(role="assistant", content="failed", timestamp=ts(3)))
    if backend_name == "sqlite":
        from persistence.sqlite.conversation._store import rewind as module

        real = module.rewind_to_message
        monkeypatch.setattr(module, "rewind_to_message", lambda *a, **kw: {**real(*a, **kw), "replacement_message_id": None})
    else:
        real_rewind_to = repo._rewind_to
        monkeypatch.setattr(repo, "_rewind_to", lambda *a, **kw: (real_rewind_to(*a, **kw)[0], None))
    refused = _refusal(repo)
    assert (refused.reason, str(refused)) == ("handoff_lost", "rewind did not retain its compaction handoff")
