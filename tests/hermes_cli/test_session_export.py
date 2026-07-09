import json
import sys

from hermes_cli.session_export import export_record_count, render_sessions_export


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


def test_default_jsonl_preserves_full_session_shape():
    session = _sample_session()

    rendered = render_sessions_export([session])

    assert [json.loads(line) for line in rendered.splitlines()] == [session]


def test_prompt_only_jsonl_emits_one_record_per_user_prompt():
    rendered = render_sessions_export(
        [_sample_session()],
        only="user-prompts",
    )

    records = [json.loads(line) for line in rendered.splitlines()]

    assert records == [
        {
            "session_id": "sess-123",
            "index": 1,
            "created_at": "2023-11-14T22:13:21Z",
            "role": "user",
            "text": "Why is login broken?",
            "message_id": 2,
            "event_id": "evt-2",
        },
        {
            "session_id": "sess-123",
            "index": 2,
            "created_at": "2023-11-14T22:13:24Z",
            "role": "user",
            "text": "Only show me the prompts.",
            "message_id": 5,
        },
    ]


def test_prompt_only_markdown_excludes_assistant_tool_and_system_content():
    rendered = render_sessions_export(
        [_sample_session()],
        fmt="markdown",
        only="user-prompts",
    )

    assert "# User prompts for session sess-123" in rendered
    assert "## 1. 2023-11-14T22:13:21Z" in rendered
    assert "Why is login broken?" in rendered
    assert "Only show me the prompts." in rendered
    assert "I will inspect the auth middleware." not in rendered
    assert "def redirect_after_login" not in rendered
    assert "hidden system context" not in rendered


def test_full_markdown_renderer_collapses_tool_output_and_filters_system():
    rendered = render_sessions_export([_sample_session()], fmt="markdown")

    assert "# Session: Debug auth flow" in rendered
    assert "## User - 2023-11-14T22:13:21Z" in rendered
    assert "## Assistant - 2023-11-14T22:13:22Z" in rendered
    assert "<details><summary>read_file</summary>" in rendered
    assert "```text\ndef redirect_after_login(): pass\n```" in rendered
    assert "hidden system context" not in rendered


def test_export_record_count_switches_unit_for_prompt_only_exports():
    assert export_record_count([_sample_session()]) == (1, "session")
    assert export_record_count([_sample_session()], only="user-prompts") == (
        2,
        "prompt",
    )


def test_sessions_export_cli_prompt_only_stdout(monkeypatch, capsys):
    import hermes_cli.main as main_mod
    import hermes_state

    captured = {}

    class FakeDB:
        def resolve_session_id(self, session_id):
            captured["resolved_from"] = session_id
            return "sess-123"

        def export_session(self, session_id):
            captured["exported"] = session_id
            return _sample_session()

        def close(self):
            captured["closed"] = True

    monkeypatch.setattr(hermes_state, "SessionDB", lambda: FakeDB())
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
    assert captured == {
        "resolved_from": "sess",
        "exported": "sess-123",
        "closed": True,
    }


def test_sessions_export_cli_prompt_only_markdown_file(monkeypatch, capsys, tmp_path):
    import hermes_cli.main as main_mod
    import hermes_state

    class FakeDB:
        def resolve_session_id(self, _session_id):
            return "sess-123"

        def export_session(self, _session_id):
            return _sample_session()

        def close(self):
            pass

    output_path = tmp_path / "prompts.md"
    monkeypatch.setattr(hermes_state, "SessionDB", lambda: FakeDB())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "hermes",
            "sessions",
            "export",
            str(output_path),
            "--session-id",
            "sess",
            "--format",
            "md",
            "--only",
            "user-prompts",
        ],
    )

    main_mod.main()

    assert f"Exported 2 prompts to {output_path}" in capsys.readouterr().out
    content = output_path.read_text(encoding="utf-8")
    assert "# User prompts for session sess-123" in content
    assert "Why is login broken?" in content
    assert "I will inspect the auth middleware." not in content


def test_sessions_export_only_rejects_unsupported_format(monkeypatch, capsys):
    import hermes_cli.main as main_mod
    import hermes_state

    class FakeDB:
        def export_all(self, source=None):
            raise AssertionError("should refuse before touching the DB")

        def close(self):
            pass

    monkeypatch.setattr(hermes_state, "SessionDB", lambda: FakeDB())
    monkeypatch.setattr(
        sys,
        "argv",
        ["hermes", "sessions", "export", "-", "--format", "html", "--only", "user-prompts"],
    )

    main_mod.main()

    assert "--only user-prompts supports --format jsonl or md." in capsys.readouterr().out
