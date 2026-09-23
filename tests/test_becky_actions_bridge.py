from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest

from gateway.action_journal import ActionJournal, MutationEvent
from gateway.becky_actions import OneShotResult
from gateway.becky_loops import BeckyLoopsBridgeServer, BeckyLoopsConfig
from gateway.becky_loop_summarizer import LoopSummary


KEY = UUID("8c9c8217-cc0f-463d-a430-173f1802edb2")
WHEN = datetime(2026, 9, 22, 18, 0, tzinfo=UTC)


class Store:
    def list_topics(self, chat_id: str) -> list[dict[str, Any]]:
        del chat_id
        return []


class Summarizer:
    async def summarize(self, **kwargs: Any) -> LoopSummary:
        del kwargs
        return LoopSummary(
            summary="summary",
            decisions=[],
            unresolved_items=[],
            next_action=None,
            waiting_on=None,
            key_events=[],
            final_outcome=None,
        )


def config() -> BeckyLoopsConfig:
    return BeckyLoopsConfig(
        enabled=True,
        chat_id="123456789",
        token="t" * 64,
        port=0,
    )


def event(*, status: str = "succeeded", key: UUID = KEY) -> MutationEvent:
    return MutationEvent(
        source_event_key=key,
        status=status,
        action_type="calendar",
        title="Dentist appointment",
        description="Created a Calendar event.",
        provider="Google Calendar",
        operation="create_event",
        destination="primary",
        occurred_at=WHEN,
        context="Calendar event creation completed.",
    )


def request(method: str, params: dict[str, Any], request_id: int = 1) -> str:
    return json.dumps({
        "jsonrpc": "2.0",
        "id": request_id,
        "method": method,
        "params": params,
    })


@pytest.mark.asyncio
async def test_action_capabilities_are_exact() -> None:
    server = BeckyLoopsBridgeServer(
        config=config(), store=Store(), summarizer=Summarizer()
    )
    response = await server._dispatch(
        request("becky.actions.capabilities", {})
    )
    assert response == {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "schema_version": "1",
            "policy_version": "1",
            "methods": ["execute_one_shot", "list_mutations", "start_loop"],
        },
    }


@pytest.mark.asyncio
async def test_list_mutations_is_cursor_paginated_and_profile_scoped(tmp_path: Path) -> None:
    journal = ActionJournal(tmp_path / "actions.sqlite3", profile_key="becky")
    journal.append(event())
    server = BeckyLoopsBridgeServer(
        config=config(),
        store=Store(),
        summarizer=Summarizer(),
        action_journal=journal,
    )

    first = await server._dispatch(
        request("becky.actions.list_mutations", {"after_cursor": None, "limit": 1})
    )
    assert first["result"]["schema_version"] == "1"
    assert len(first["result"]["events"]) == 1
    assert first["result"]["events"][0]["source_event_key"] == str(KEY)
    assert first["result"]["next_cursor"] is None

    invalid = await server._dispatch(
        request("becky.actions.list_mutations", {"after_cursor": "bad", "limit": 1})
    )
    assert invalid["error"]["message"] == "mutation_cursor_invalid"

    malformed = await server._dispatch(
        request("becky.actions.list_mutations", {"after_cursor": None, "limit": 1, "extra": True})
    )
    assert malformed["error"]["message"] == "protocol"


@pytest.mark.asyncio
async def test_one_shot_journals_result_and_replays_without_repeating(tmp_path: Path) -> None:
    journal = ActionJournal(tmp_path / "actions.sqlite3", profile_key="becky")
    calls: list[dict[str, Any]] = []

    async def executor(**kwargs: Any) -> OneShotResult:
        calls.append(kwargs)
        return OneShotResult(schema_version="1", disposition="succeeded", event=event())

    server = BeckyLoopsBridgeServer(
        config=config(),
        store=Store(),
        summarizer=Summarizer(),
        action_journal=journal,
        one_shot_executor=executor,
    )
    params = {
        "title": "Dentist appointment",
        "text": "Add a dentist appointment tomorrow at 2 PM",
        "idempotency_key": str(KEY),
        "note_default": "obsidian",
        "policy_version": "1",
    }
    first = await server._dispatch(request("becky.actions.execute_one_shot", params))
    replay = await server._dispatch(request("becky.actions.execute_one_shot", params, 2))
    assert first["result"]["disposition"] == "succeeded"
    assert replay["result"] == first["result"]
    assert len(calls) == 1
    assert len(journal.list(after_cursor=None).events) == 1


@pytest.mark.asyncio
async def test_failed_one_shot_is_journaled_as_failed(tmp_path: Path) -> None:
    journal = ActionJournal(tmp_path / "actions.sqlite3", profile_key="becky")

    async def executor(**kwargs: Any) -> OneShotResult:
        del kwargs
        return OneShotResult(
            schema_version="1",
            disposition="failed",
            event=event(status="failed"),
        )

    server = BeckyLoopsBridgeServer(
        config=config(),
        store=Store(),
        summarizer=Summarizer(),
        action_journal=journal,
        one_shot_executor=executor,
    )
    response = await server._dispatch(
        request(
            "becky.actions.execute_one_shot",
            {
                "title": "Dentist appointment",
                "text": "Add a dentist appointment tomorrow at 2 PM",
                "idempotency_key": str(KEY),
                "note_default": "obsidian",
                "policy_version": "1",
            },
        )
    )
    assert response["result"]["disposition"] == "failed"
    assert journal.list(after_cursor=None).events[0].status.value == "failed"


@pytest.mark.asyncio
async def test_one_shot_rejects_executor_event_with_private_payload() -> None:
    journal = ActionJournal(":memory:", profile_key="becky")

    async def executor(**kwargs: Any) -> dict[str, Any]:
        del kwargs
        return {
            "schema_version": "1",
            "disposition": "succeeded",
            "event": {
                **event().model_dump(mode="json"),
                "raw_output": "credential must never cross the bridge",
            },
        }

    server = BeckyLoopsBridgeServer(
        config=config(),
        store=Store(),
        summarizer=Summarizer(),
        action_journal=journal,
        one_shot_executor=executor,
    )
    response = await server._dispatch(
        request(
            "becky.actions.execute_one_shot",
            {
                "title": "Dentist appointment",
                "text": "Add a dentist appointment tomorrow at 2 PM",
                "idempotency_key": str(KEY),
                "note_default": "obsidian",
                "policy_version": "1",
            },
        )
    )
    assert response["error"]["message"] == "protocol"
    assert journal.list(after_cursor=None).events == []


@pytest.mark.asyncio
async def test_one_shot_without_executor_is_safe_remote_failure() -> None:
    server = BeckyLoopsBridgeServer(
        config=config(), store=Store(), summarizer=Summarizer()
    )
    response = await server._dispatch(
        request(
            "becky.actions.execute_one_shot",
            {
                "title": "Dentist appointment",
                "text": "Add a dentist appointment tomorrow at 2 PM",
                "idempotency_key": str(KEY),
                "note_default": "obsidian",
                "policy_version": "1",
            },
        )
    )
    assert response["error"]["message"] == "one_shot_not_configured"


@pytest.mark.asyncio
async def test_start_loop_calls_starter_once_and_never_mutates(tmp_path: Path) -> None:
    journal = ActionJournal(tmp_path / "actions.sqlite3", profile_key="becky")
    calls: list[dict[str, Any]] = []

    async def starter(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {
            "schema_version": "1",
            "state": "completed",
            "title": kwargs["title"],
            "telegram_url": "https://t.me/c/123456789/42",
        }

    server = BeckyLoopsBridgeServer(
        config=config(),
        store=Store(),
        summarizer=Summarizer(),
        action_journal=journal,
        action_loop_starter=starter,
    )
    params = {
        "title": "Dentist appointment",
        "context": "The Calendar event creation failed; help me follow up.",
        "prior_status": "failed",
        "idempotency_key": str(KEY),
    }
    first = await server._dispatch(request("becky.actions.start_loop", params))
    replay = await server._dispatch(request("becky.actions.start_loop", params, 2))
    assert first["result"]["state"] == "completed"
    assert replay["result"] == first["result"]
    assert len(calls) == 1
    assert journal.list(after_cursor=None).events == []
