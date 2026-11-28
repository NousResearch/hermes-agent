import json

from gateway import channel_directory
from hermes_cli import mcp_config


def test_session_entries_decode_legacy_blob_values():
    origins = [
        (
            {
                b"chat_id": b"room-7",
                b"thread_id": b"thread-2",
                b"chat_name": b"Support",
                b"chat_topic": b"Billing",
            },
            b"group",
        )
    ]

    entries = channel_directory._entries_from_origins(
        "discord", "test", lambda: origins
    )

    assert entries == [
        {
            "id": "room-7:thread-2",
            "name": "Support / Billing",
            "type": "group",
            "thread_id": "thread-2",
        }
    ]
    json.dumps(entries)


def test_mcp_list_reports_non_mapping_server_entries(monkeypatch, capsys):
    monkeypatch.setattr(
        mcp_config,
        "_get_mcp_servers",
        lambda: {
            "broken": "legacy-value",
            "working": {"url": "https://example.test/mcp"},
        },
    )

    mcp_config.cmd_mcp_list()
    output = capsys.readouterr().out

    assert "broken" in output
    assert "invalid" in output
    assert "working" in output
