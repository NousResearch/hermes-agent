"""The drain that settles a chat client's rows for a finished background delegation."""

from __future__ import annotations

import queue

import pytest

from acp_adapter import delegation_notices as notices


def batch_event(**overrides):
    event = {
        "type": "async_delegation",
        "delegation_id": "deleg_abc123",
        "session_key": "8e440013-bd85-4c93-ace0-070a12b1d2fe",
        "parent_session_id": "8e440013-bd85-4c93-ace0-070a12b1d2fe",
        "status": "completed",
        "goals": ["say one", "say two"],
        "results": [
            {"task_index": 0, "status": "completed", "summary": "hi from one"},
            {"task_index": 1, "status": "failed", "summary": "boom"},
        ],
    }
    event.update(overrides)
    return event


class TestOwnership:
    def test_any_session_field_can_prove_the_owner(self):
        assert notices.owns_event(batch_event(), "8e440013-bd85-4c93-ace0-070a12b1d2fe")
        assert notices.owns_event(
            batch_event(session_key="", origin_session_id="sess-9"), "sess-9"
        )

    def test_a_foreign_session_never_claims_it(self):
        assert not notices.owns_event(batch_event(), "another-session")
        assert not notices.owns_event(batch_event(), "")


class TestRows:
    def test_a_batch_becomes_one_row_per_task_keyed_the_way_the_dispatch_keyed_them(self):
        rows = notices.child_rows(batch_event())

        assert [row["id"] for row in rows] == ["deleg_abc123:0", "deleg_abc123:1"]
        assert [row["state"] for row in rows] == ["completed", "failed"]
        assert rows[0]["summary"] == "hi from one"

    def test_a_single_delegation_is_its_own_row(self):
        rows = notices.child_rows(
            batch_event(delegation_id="deleg_solo", results=None, summary="done it", status="completed")
        )

        assert rows == [{"id": "deleg_solo", "state": "completed", "summary": "done it"}]

    def test_an_outcome_hermes_does_not_name_is_unverifiable(self):
        rows = notices.child_rows(
            batch_event(results=[{"task_index": 0, "status": "unknown outcome"}])
        )

        assert rows[0]["state"] == "unverifiable"

    def test_a_cancelled_child_reads_as_stopped(self):
        rows = notices.child_rows(
            batch_event(results=[{"task_index": 0, "status": "cancelled"}])
        )

        assert rows[0]["state"] == "stopped"


class TestUpdate:
    def test_the_update_names_the_answers_and_carries_the_roster(self):
        update = notices.completion_update(batch_event())

        assert update["sessionUpdate"] == "agent_message_chunk"
        assert "hi from one" in update["content"]["text"]
        assert "1 of 2 subagents finished" in update["content"]["text"]
        assert [row["id"] for row in update["_meta"]["hermes"]["subagents"]] == [
            "deleg_abc123:0",
            "deleg_abc123:1",
        ]

    def test_a_lone_child_is_reported_by_its_own_answer(self):
        update = notices.completion_update(
            batch_event(delegation_id="deleg_solo", results=None, summary="hi from solo")
        )

        assert update["content"]["text"] == "Subagent completed: hi from solo"


class TestClaim:
    @pytest.fixture
    def registry(self, monkeypatch):
        pending: queue.Queue = queue.Queue()
        monkeypatch.setattr(
            "tools.process_registry.process_registry", type("R", (), {"completion_queue": pending})
        )
        return pending

    def test_it_takes_its_own_event_and_leaves_every_other_queued(self, registry, monkeypatch):
        monkeypatch.setattr(
            "tools.async_delegation.claim_event_delivery", lambda event, consumer: "claim-1"
        )
        foreign = batch_event(
            session_key="someone-else", parent_session_id="someone-else", delegation_id="deleg_other"
        )
        registry.put(batch_event())
        registry.put(foreign)

        claimed = notices.claim_completions(["8e440013-bd85-4c93-ace0-070a12b1d2fe"])

        assert [event["delegation_id"] for _session, event, _claim in claimed] == ["deleg_abc123"]
        assert [event["delegation_id"] for event in list(registry.queue)] == ["deleg_other"]

    def test_an_event_another_drain_owns_stays_queued(self, registry, monkeypatch):
        monkeypatch.setattr(
            "tools.async_delegation.claim_event_delivery", lambda event, consumer: None
        )
        registry.put(batch_event())

        assert notices.claim_completions(["8e440013-bd85-4c93-ace0-070a12b1d2fe"]) == []
        assert registry.qsize() == 1
