from types import SimpleNamespace

import pytest

from gateway.platforms.base import SendResult
from gateway.progress_delivery import ProgressDeliveryState


class _SmallProgressAdapter:
    MAX_MESSAGE_LENGTH = 10


class _CaptureProgressAdapter(_SmallProgressAdapter):
    def __init__(self, results: list[SendResult]) -> None:
        self.results = list(results)
        self.sent: list[str] = []

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        self.sent.append(content)
        return self.results.pop(0)


class _AmbiguousEditCaptureAdapter(_CaptureProgressAdapter):
    def __init__(self, send_results: list[SendResult]) -> None:
        super().__init__(send_results)
        self.edited: list[str] = []

    async def edit_message(self, chat_id, message_id, content) -> SendResult:
        self.edited.append(content)
        return SendResult(
            success=False,
            error="ack lost",
            retryable=True,
            ambiguous=True,
        )


def _state(adapter=None) -> ProgressDeliveryState:
    context = SimpleNamespace(
        progress_grouping="grouped",
        source=SimpleNamespace(chat_id="chat-1"),
        _progress_metadata=None,
        _progress_reply_to=None,
        _cleanup_progress=False,
        _cleanup_msg_ids=[],
        last_progress_msg=[None],
        repeat_count=[0],
    )
    return ProgressDeliveryState(context, adapter or _SmallProgressAdapter())


def test_progress_delivery_state_splits_groups_at_platform_boundary() -> None:
    state = _state()

    assert state.split_groups(["abcdef", "ghij"]) == [["abcdef"], ["ghij"]]

    chunks = state.split_line("klmnopqrstuv")
    assert chunks == ["klmnopqrst", "uv"]
    assert "".join(chunks) == "klmnopqrstuv"
    assert all(len(chunk) <= state.text_limit for chunk in chunks)


def test_progress_delivery_state_tracks_only_undelivered_suffix() -> None:
    state = _state()
    state.progress_lines = ["first", "second", "third"]
    state.delivered_progress_lines = ["first", "second"]

    assert state.undelivered_lines() == ["third"]


def test_progress_delivery_state_drops_only_attempted_separate_send() -> None:
    state = _state()
    state.can_edit = False
    state.progress_lines = ["first", "second"]
    state.delivered_progress_lines = ["first"]

    state.drop_attempted_send(sent_all=False, attempted_line="second")

    assert state.progress_lines == ["first"]
    assert state.delivered_progress_lines == ["first"]
    assert state.progress_msg_id is None


@pytest.mark.asyncio
async def test_ambiguous_fresh_send_retains_only_never_attempted_groups() -> None:
    adapter = _CaptureProgressAdapter([
        SendResult(
            success=False,
            error="ack lost",
            retryable=True,
            ambiguous=True,
        )
    ])
    state = _state(adapter)

    assert not await state.start_fresh_bubbles(["abcdef", "ghij"])

    assert adapter.sent == ["abcdef"]
    assert state.progress_lines == ["ghij"]
    assert state.delivered_progress_lines == []
    assert state.progress_msg_id is None


@pytest.mark.asyncio
async def test_too_long_continuation_sends_only_undelivered_suffix() -> None:
    adapter = _CaptureProgressAdapter([
        SendResult(success=True, message_id="new-bubble")
    ])
    state = _state(adapter)
    state.progress_msg_id = "full-bubble"
    state.progress_lines = ["first", "second"]
    state.delivered_progress_lines = ["first"]

    assert await state.continue_after_too_long()

    assert adapter.sent == ["second"]
    assert state.progress_lines == ["second"]
    assert state.delivered_progress_lines == ["second"]
    assert state.progress_msg_id == "new-bubble"


@pytest.mark.asyncio
async def test_drain_preserves_suffix_after_ambiguous_edit_retry() -> None:
    adapter = _AmbiguousEditCaptureAdapter([
        SendResult(success=True, message_id="suffix-bubble")
    ])
    state = _state(adapter)
    state.progress_msg_id = "existing-bubble"
    state.progress_lines = ["first", "second", "third"]
    state.delivered_progress_lines = ["first", "second"]
    state.pending_ambiguous_edit = (
        "existing-bubble",
        "first\nsecond",
        ["first", "second"],
    )

    assert await state.settle_edit_during_drain()

    assert adapter.edited == ["first\nsecond"]
    assert adapter.sent == ["third"]
    assert state.progress_lines == ["third"]
    assert state.delivered_progress_lines == ["third"]
    assert state.progress_msg_id == "suffix-bubble"
    assert state.pending_ambiguous_edit is None
