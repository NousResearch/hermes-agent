from __future__ import annotations

import json

from gateway.validation import discord_context_smoke as smoke


def test_main_without_credentials_writes_failure_report(tmp_path) -> None:
    rc = smoke.main([
        "--channel-id",
        "1501008202630696981",
        "--output-dir",
        str(tmp_path),
    ])

    report = json.loads((tmp_path / "discord_context_smoke_report.json").read_text(encoding="utf-8"))
    assert rc == 2
    assert report["ok"] is False
    assert "Missing --bot-user-id or HERMES_BOT_USER_ID." in report["errors"]
    assert "Missing --token or HERMES_CONTEXT_VALIDATION_DISCORD_TOKEN." in report["errors"]


def test_first_context_dump_attachment_picks_message_txt() -> None:
    attachment = smoke._first_context_dump_attachment(
        {
            "attachments": [
                {"filename": "debug.json", "url": "https://example.invalid/debug.json"},
                {"filename": "session.message.txt", "url": "https://example.invalid/session.message.txt"},
            ]
        }
    )

    assert attachment == {
        "filename": "session.message.txt",
        "url": "https://example.invalid/session.message.txt",
    }
