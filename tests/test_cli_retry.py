"""Regression tests for CLI /retry history replacement semantics."""

from tests.test_cli_init import _make_cli


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
