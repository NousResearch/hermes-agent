"""Regression tests for CLI /retry history replacement semantics."""

from tests.cli.test_cli_init import _make_cli


AUTO_CONTINUE_MARKER = "[Continuing after max-iteration exhaustion]"


def test_retry_last_truncates_history_before_requeueing_message():
    cli = _make_cli()
    cli.conversation_history = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "one"},
        {"role": "user", "content": "retry me"},
        {"role": "assistant", "content": "old answer"},
    ]

    retry_msg = cli.retry_last()

    assert retry_msg == "retry me"
    assert cli.conversation_history == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "one"},
    ]

    cli.conversation_history.append({"role": "user", "content": retry_msg})
    cli.conversation_history.append({"role": "assistant", "content": "new answer"})

    assert [m["content"] for m in cli.conversation_history if m["role"] == "user"] == [
        "first",
        "retry me",
    ]


def test_process_command_retry_requeues_original_message_not_retry_command():
    cli = _make_cli()
    queued = []

    class _Queue:
        def put(self, value):
            queued.append(value)

    cli._pending_input = _Queue()
    cli.conversation_history = [
        {"role": "user", "content": "retry me"},
        {"role": "assistant", "content": "old answer"},
    ]

    cli.process_command("/retry")

    assert queued == ["retry me"]
    assert cli.conversation_history == []


def test_retry_ignores_synthetic_continuation_user_marker():
    cli = _make_cli()
    cli.conversation_history = [
        {"role": "user", "content": "original task"},
        {"role": "assistant", "content": "calling a tool"},
        {"role": "tool", "tool_call_id": "call-1", "content": "result"},
        {"role": "assistant", "content": "Starting a fresh continuation turn."},
        {"role": "user", "content": f"{AUTO_CONTINUE_MARKER}\nkeep going"},
        {"role": "assistant", "content": "finished"},
    ]

    retry_msg = cli.retry_last()

    assert retry_msg == "original task"
    assert cli.conversation_history == []


def test_undo_counts_a_synthetic_continuation_as_part_of_the_original_turn():
    cli = _make_cli()
    cli._session_db = None
    cli.session_id = None
    cli.agent = None
    prefills = []
    cli._prefill_input_buffer = prefills.append
    cli.conversation_history = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "original task"},
        {"role": "assistant", "content": "Starting a fresh continuation turn."},
        {"role": "user", "content": f"{AUTO_CONTINUE_MARKER}\nkeep going"},
        {"role": "assistant", "content": "finished"},
    ]

    cli.undo_last()

    assert cli.conversation_history == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "first answer"},
    ]
    assert prefills == ["original task"]
