"""Gateway native clarify-batch bridge regressions."""

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
