"""Gateway native clarify-batch bridge regressions."""

import concurrent.futures

from types import SimpleNamespace
from unittest.mock import MagicMock

from gateway.platforms.base import SendResult
from gateway.run_turn_runner import TurnRunner


def test_native_batch_callback_registers_once_sends_all_and_returns_partial_answers(monkeypatch):
    from tools import clarify_gateway as cm

    with cm._lock:
        cm._entries.clear()
        cm._session_index.clear()

    adapter = MagicMock()
    adapter.send_clarify_batch = MagicMock()
    ctx = SimpleNamespace(
        _status_adapter=adapter, _status_chat_id="42", _status_thread_metadata={"thread_id": "7"},
        session_key="telegram:42:7",
    )
    runner = object.__new__(TurnRunner)
    runner._ctx = ctx
    runner._close_native_stream_boundary = MagicMock()
    runner._stream_consumer = MagicMock(return_value=None)
    future = MagicMock()
    future.result.return_value = SendResult(success=True)
    runner._schedule = MagicMock(return_value=future)
    monkeypatch.setattr(cm, "wait_for_batch_responses", lambda ids, timeout: ({ids[0]: "850", ids[1]: ""}, False))

    result = runner._clarify_callback_sync("", None, questions=[
        {"qid": "q0", "question": "Budget?", "choices": ["850", "500"]},
        {"qid": "q1", "question": "Screenshot?", "choices": None},
    ])

    assert result == {"answers": {"q0": "850", "q1": ""}, "timed_out": False}
    kwargs = adapter.send_clarify_batch.call_args.kwargs
    assert [q["question"] for q in kwargs["questions"]] == ["Budget?", "Screenshot?"]
    assert len(kwargs["clarify_ids"]) == 2
    adapter.clear_clarify_batch.assert_called_once_with(kwargs["batch_id"])


def test_native_batch_send_timeout_keeps_cards_armed_until_batch_wait_and_restores_continuation(monkeypatch):
    """A late Telegram send acknowledgement must not invalidate rendered cards."""
    from tools import clarify_gateway as cm

    with cm._lock:
        cm._entries.clear()
        cm._session_index.clear()

    adapter = MagicMock()
    adapter.send_clarify_batch = MagicMock()
    ctx = SimpleNamespace(
        _status_adapter=adapter, _status_chat_id="42", _status_thread_metadata={}, session_key="telegram:42",
    )
    runner = object.__new__(TurnRunner)
    runner._ctx = ctx
    runner._close_native_stream_boundary = MagicMock()
    stream = MagicMock()
    runner._stream_consumer = MagicMock(return_value=stream)
    future = MagicMock()
    future.result.side_effect = concurrent.futures.TimeoutError()
    runner._schedule = MagicMock(return_value=future)
    observed = {}

    real_wait = cm.wait_for_batch_responses

    def wait_for_batch(ids, timeout):
        observed["ids"] = ids
        observed["timeout"] = timeout
        assert all(not cm._entries[cid].event.is_set() for cid in ids)
        for cid in ids:
            cm.resolve_gateway_clarify(cid, "answer")
        return real_wait(ids, timeout=0)

    monkeypatch.setattr(cm, "wait_for_batch_responses", wait_for_batch)

    result = runner._clarify_callback_sync("", None, questions=[
        {"qid": "q0", "question": "First?", "choices": ["A", "B"]},
        {"qid": "q1", "question": "Second?", "choices": ["C", "D"]},
    ])

    assert result == {"answers": {"q0": "answer", "q1": "answer"}, "timed_out": False}
    assert observed["timeout"] == cm.get_clarify_timeout()
    assert cm._entries == {}
    assert cm._session_index == {}
    adapter.clear_clarify_batch.assert_called_once()
    adapter.resume_typing_for_chat.assert_called_once_with("42")
    stream.request_reopen_seed.assert_called_once()
