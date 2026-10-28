"""Native task cards follow the adapter's opt-in, not the platform it serves."""

import pytest

from gateway.config import Platform
from gateway.platforms.base import BasePlatformAdapter, SendResult
from tests.gateway.test_run_progress_topics import (
    DuplicateNativeToolsAgent,
    FailingNativeTaskCardAdapter,
    NativeTaskCardAdapter,
    _run_with_agent,
)


class NoMetadataEditCardAdapter(FailingNativeTaskCardAdapter):
    async def edit_message(self, chat_id, message_id, content, *, finalize=False) -> SendResult:
        self.edits.append({"message_id": message_id, "content": content})
        return SendResult(success=True, message_id=message_id)


class NonEditableCardAdapter(FailingNativeTaskCardAdapter):
    edit_message = BasePlatformAdapter.edit_message


async def run_turn(monkeypatch, tmp_path, adapter_cls):
    adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        DuplicateNativeToolsAgent,
        session_id="sess-capability-cards",
        platform=Platform.TELEGRAM,
        adapter_cls=adapter_cls,
    )
    assert result["final_response"] == "done"
    return adapter


@pytest.mark.asyncio
async def test_non_slack_adapter_that_opts_in_gets_task_cards(monkeypatch, tmp_path):
    adapter = await run_turn(monkeypatch, tmp_path, NativeTaskCardAdapter)

    tasks = {}
    for update in adapter.native_updates:
        assert "recipient_team_id" not in update["metadata"]
        tasks.update({task["id"]: task for task in update["tasks"]})
    assert {call_id: task["status"] for call_id, task in tasks.items()} == {
        "call-a": "complete",
        "call-b": "error",
    }
    assert (tasks["call-a"]["tool_name"], tasks["call-a"]["preview"]) == ("web_search", "alpha")
    assert adapter.sent == []
    assert adapter.native_stops == 1


@pytest.mark.asyncio
async def test_card_fallback_edits_without_passing_unaccepted_metadata(monkeypatch, tmp_path):
    adapter = await run_turn(monkeypatch, tmp_path, NoMetadataEditCardAdapter)

    assert len(adapter.sent) == 1
    assert {edit["message_id"] for edit in adapter.edits} == {"progress-1"}
    assert adapter.edits[-1]["content"].endswith("web_search - beta - error")


@pytest.mark.asyncio
async def test_card_fallback_stays_silent_when_the_adapter_cannot_edit(monkeypatch, tmp_path):
    adapter = await run_turn(monkeypatch, tmp_path, NonEditableCardAdapter)

    assert len(adapter.native_updates) == 1
    assert adapter.sent == []
