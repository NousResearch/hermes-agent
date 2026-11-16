import json
import sys

from hermes_cli.session_export import export_record_count, render_sessions_export
from hermes_cli.session_export_html import (
    _generate_messages_html,
    generate_multi_session_html_export,
)


def _sample_session():
    return {
        "id": "sess-123",
        "source": "cli",
        "model": "test/model",
        "title": "Debug auth flow",
        "started_at": 1700000000,
        "message_count": 5,
        "messages": [
            {
                "id": 1,
                "role": "system",
                "content": "hidden system context",
                "timestamp": 1700000000,
            },
            {
                "id": 2,
                "role": "user",
                "content": "Why is login broken?",
                "timestamp": 1700000001,
                "platform_message_id": "evt-2",
            },
            {
                "id": 3,
                "role": "assistant",
                "content": "I will inspect the auth middleware.",
                "timestamp": 1700000002,
            },
            {
                "id": 4,
                "role": "tool",
                "tool_name": "read_file",
                "content": "def redirect_after_login(): pass",
                "timestamp": 1700000003,
            },
            {
                "id": 5,
                "role": "user",
                "content": [{"type": "text", "text": "Only show me the prompts."}],
                "timestamp": 1700000004,
            },
        ],
    }










def test_html_export_escapes_tool_call_names():
    payload = '<img src=x onerror="alert(document.domain)">'

    rendered = _generate_messages_html(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": payload, "arguments": "<b>x</b>"},
                    }
                ],
            }
        ]
    )

    assert payload not in rendered
    assert '&lt;img src=x onerror=&quot;alert(document.domain)&quot;&gt;' in rendered
    assert "&lt;b&gt;x&lt;/b&gt;" in rendered




def test_export_record_count_switches_unit_for_prompt_only_exports():
    assert export_record_count([_sample_session()]) == (1, "session")
    assert export_record_count([_sample_session()], only="user-prompts") == (
        2,
        "prompt",
    )


def test_sessions_export_cli_prompt_only_stdout(monkeypatch, capsys, tmp_path):
    import hermes_cli.main as main_mod
    import hermes_state

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    path = tmp_path / "state.db"
    db = hermes_state.SessionDB(db_path=path)
    db.create_session("sess-123", "cli")
    for message in _sample_session()["messages"]:
        db.append_message("sess-123", message["role"], message["content"])
    db.close()
    before = path.read_bytes()
    monkeypatch.setattr(
        sys,
        "argv",
        ["hermes", "sessions", "export", "-", "--session-id", "sess", "--only", "user-prompts"],
    )

    main_mod.main()

    output = capsys.readouterr().out
    records = [json.loads(line) for line in output.splitlines()]
    assert [record["text"] for record in records] == [
        "Why is login broken?",
        "Only show me the prompts.",
    ]
    assert path.read_bytes() == before


