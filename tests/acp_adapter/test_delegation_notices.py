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


def live_record(**overrides):
    record = {
        "delegation_id": "deleg_abc123",
        "session_key": "8e440013-bd85-4c93-ace0-070a12b1d2fe",
        "status": "running",
        "goals": ["say one", "say two"],
        "dispatched_at": 1000.0,
        "children_activity": [
            {"api_calls": 4, "current_tool": "search_files", "seconds_since_activity": 1.2},
            {"api_calls": 2, "current_tool": "read_file", "seconds_since_activity": 0.4},
        ],
    }
    record.update(overrides)
    return record


class TestLiveProgress:
    def test_only_the_callers_running_delegations_are_reported(self, monkeypatch):
        monkeypatch.setattr(
            "tools.async_delegation.list_async_delegations",
            lambda: [
                live_record(),
                live_record(delegation_id="deleg_other", session_key="someone-else"),
                live_record(delegation_id="deleg_done", status="completed"),
            ],
        )

        live = notices.live_delegations("8e440013-bd85-4c93-ace0-070a12b1d2fe")

        assert [record["delegation_id"] for record in live] == ["deleg_abc123"]

    def test_a_batch_reports_the_rows_the_dispatch_drew(self):
        rows = notices.progress_rows(live_record())

        assert [row["id"] for row in rows] == ["deleg_abc123:0", "deleg_abc123:1"]
        assert [row["state"] for row in rows] == ["working", "working"]
        assert [row["label"] for row in rows] == ["say one", "say two"]

    def test_each_row_names_the_tool_that_child_is_in(self):
        rows = notices.progress_rows(live_record())

        assert [row["activity"] for row in rows] == ["search_files", "read_file"]
        assert [row["api_calls"] for row in rows] == [4, 2]
        assert [row["seconds_since_activity"] for row in rows] == [1.2, 0.4]

    def test_elapsed_counts_from_the_dispatch_not_the_tick(self):
        rows = notices.progress_rows(live_record(dispatched_at=1_000_000.0))

        assert all(row["elapsed_seconds"] > 1_000_000 for row in rows)

    def test_a_single_delegation_keeps_the_bare_id(self):
        rows = notices.progress_rows(live_record(goals=None, goal="say one"))

        assert [row["id"] for row in rows] == ["deleg_abc123"]
        assert "label" not in rows[0]

    def test_a_split_unit_reports_the_calls_own_indexes(self):
        rows = notices.progress_rows(
            live_record(
                goals=["say one", "say two", "say three"],
                task_indexes=[2],
                children_activity=[{"api_calls": 1, "current_tool": "patch"}],
            )
        )

        assert [row["id"] for row in rows] == ["deleg_abc123:2"]
        assert rows[0]["label"] == "say three"
        assert rows[0]["activity"] == "patch"

    def test_a_child_with_no_sampled_activity_still_gets_a_row(self):
        rows = notices.progress_rows(live_record(children_activity=None))

        assert [row["id"] for row in rows] == ["deleg_abc123:0", "deleg_abc123:1"]
        assert all("activity" not in row for row in rows)

    def test_the_update_carries_the_roster_without_a_transcript_line(self):
        update = notices.progress_update(notices.progress_rows(live_record()))

        assert update["sessionUpdate"] == "session_info_update"
        assert "content" not in update
        assert update["_meta"]["hermes"]["progress"] is True
        assert len(update["_meta"]["hermes"]["subagents"]) == 2

    def test_rows_of_every_live_delegation_are_merged(self, monkeypatch):
        monkeypatch.setattr(
            "tools.async_delegation.list_async_delegations",
            lambda: [live_record(), live_record(delegation_id="deleg_second")],
        )

        rows = notices.live_progress_rows("8e440013-bd85-4c93-ace0-070a12b1d2fe")

        assert [row["id"] for row in rows] == [
            "deleg_abc123:0", "deleg_abc123:1", "deleg_second:0", "deleg_second:1"
        ]

    def test_nothing_running_means_nothing_is_sent(self, monkeypatch):
        monkeypatch.setattr("tools.async_delegation.list_async_delegations", list)

        assert notices.live_progress_rows("8e440013-bd85-4c93-ace0-070a12b1d2fe") == []
