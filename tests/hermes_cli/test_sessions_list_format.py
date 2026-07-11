import json

from hermes_cli.main import _format_sessions_list


SESSIONS = [
    {
        "id": "20260525_213536_16412f",
        "source": "cli",
        "title": "Test Message Confirmation",
        "preview": "test preview",
        "last_active": 1760000000.0,
        "started_at": 1759999900.0,
        "message_count": 4,
        "system_prompt": "private prompt content should not appear in list JSON",
    },
    {
        "id": "20260525_204817_9bb28c",
        "source": "telegram",
        "title": None,
        "preview": "hello\tworld\nagain",
        "last_active": 1759990000.0,
        "started_at": 1759989900.0,
        "message_count": 2,
    },
]


def test_sessions_list_json_format_is_machine_readable():
    output = _format_sessions_list(SESSIONS, output_format="json")

    data = json.loads(output)

    assert data == [
        {
            "id": "20260525_213536_16412f",
            "title": "Test Message Confirmation",
            "preview": "test preview",
            "last_active": 1760000000.0,
            "source": "cli",
        },
        {
            "id": "20260525_204817_9bb28c",
            "title": None,
            "preview": "hello\tworld\nagain",
            "last_active": 1759990000.0,
            "source": "telegram",
        },
    ]
    assert "system_prompt" not in output
    assert output.endswith("\n")


def test_sessions_list_tsv_format_has_stable_columns_and_escapes_newlines_tabs():
    output = _format_sessions_list(SESSIONS, output_format="tsv")

    lines = output.splitlines()

    assert lines[0] == "id\ttitle\tpreview\tlast_active\tsource"
    assert lines[1] == "20260525_213536_16412f\tTest Message Confirmation\ttest preview\t1760000000.0\tcli"
    assert lines[2] == "20260525_204817_9bb28c\t\thello world again\t1759990000.0\ttelegram"


def test_sessions_list_table_format_preserves_existing_human_columns():
    output = _format_sessions_list(SESSIONS, output_format="table")

    assert "Title" in output
    assert "Preview" in output
    assert "Last Active" in output
    assert "ID" in output
    assert "Test Message Confirmation" in output
    assert "20260525_213536_16412f" in output
