"""Regression tests for CLI /retry history replacement semantics."""

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


def test_retry_last_prints_redacted_structured_preview(capsys):
    cli = _make_cli()
    content = [
        {"type": "text", "text": "look"},
        {"type": "image_url", "image_url": {"url": "DATA:image/png;base64,raw-image"}},
    ]
    cli.conversation_history = [
        {"role": "user", "content": content},
        {"role": "assistant", "content": "old answer"},
    ]

    retry_msg = cli.retry_last()
    output = capsys.readouterr().out

    assert retry_msg == content
    assert "[image]" in output
    assert "raw-image" not in output


def test_undo_last_prints_redacted_structured_preview(capsys):
    cli = _make_cli()
    cli.conversation_history = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "read this"},
                {
                    "type": "file",
                    "file": {"filename": "notes.pdf", "file_data": "raw-file"},
                },
            ],
        },
        {"role": "assistant", "content": "old answer"},
    ]

    cli.undo_last()
    output = capsys.readouterr().out

    assert "[file]" in output
    assert "raw-file" not in output
    assert cli.conversation_history == []


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
