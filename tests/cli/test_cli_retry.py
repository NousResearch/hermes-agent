"""Regression tests for CLI /retry and /undo history replacement semantics."""

from tests.cli.test_cli_init import _make_cli


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


def test_undo_last_reports_one_turn_and_internal_transcript_count(capsys):
    cli = _make_cli()
    cli.conversation_history = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "one"},
        {"role": "user", "content": "fix /undo"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "call_1", "function": {"name": "search_files", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": '{"success": true}'},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "[Internal: inspect the attached image and continue.]"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            ],
            "_native_image_attachment": True,
        },
        {"role": "assistant", "content": "intermediate note"},
        {"role": "assistant", "content": "final answer"},
    ]

    cli.undo_last()

    out = capsys.readouterr().out
    assert cli.conversation_history == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "one"},
    ]
    assert "Undid 1 turn" in out
    assert "6 internal transcript messages" in out
    assert 'Removed your last message: "fix /undo"' in out
    assert "Undid 5 message(s)" not in out
