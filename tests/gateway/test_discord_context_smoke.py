from __future__ import annotations

import json
from pathlib import Path

from gateway.validation import discord_context_smoke as smoke


class FakeDiscordRest:
    sent_messages: list[str] = []

    def __init__(self, token: str, *, token_kind: str = "bot") -> None:
        self.token = token
        self.token_kind = token_kind

    def send_message(self, channel_id: str, content: str) -> dict[str, object]:
        self.sent_messages.append(content)
        return {"id": str(len(self.sent_messages))}

    def channel_messages_after(
        self, channel_id: str, after_message_id: str
    ) -> list[dict[str, object]]:
        if after_message_id == "1":
            return [
                {
                    "id": "2",
                    "author": {"id": "bot-1"},
                    "content": (
                        "Hi - using openai/gpt-oss-20b via OpenRouter. "
                        "Context: gateway-context-v1; skills: none; 100 tokens."
                    ),
                }
            ]
        if after_message_id == "2":
            return [
                {
                    "id": "3",
                    "author": {"id": "bot-1"},
                    "content": "context dump",
                    "attachments": [
                        {
                            "filename": "session.message.txt",
                            "url": "https://example.invalid/session.message.txt",
                        }
                    ],
                }
            ]
        return []

    def download(self, url: str, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "\n".join(
                [
                    "Estimated total tokens: 100",
                    "## Context Layers",
                    "SOUL",
                    "## Raw API Messages",
                    "[]",
                    "## Tool Schemas",
                    "[]",
                    "## Debug Metadata",
                    '{"base_context_version": "gateway-context-v1", "loaded_skill_names": [], "source": {"platform": "discord"}}',
                ]
            ),
            encoding="utf-8",
        )


def test_main_without_credentials_writes_failure_report(tmp_path) -> None:
    rc = smoke.main(
        [
            "--channel-id",
            "1501008202630696981",
            "--output-dir",
            str(tmp_path),
        ]
    )

    report = json.loads(
        (tmp_path / "discord_context_smoke_report.json").read_text(encoding="utf-8")
    )
    assert rc == 2
    assert report["ok"] is False
    assert "Missing --bot-user-id or HERMES_BOT_USER_ID." in report["errors"]
    assert (
        "Missing --token or HERMES_CONTEXT_VALIDATION_DISCORD_TOKEN."
        in report["errors"]
    )


def test_first_context_dump_attachment_picks_message_txt() -> None:
    attachment = smoke._first_context_dump_attachment(
        {
            "attachments": [
                {"filename": "debug.json", "url": "https://example.invalid/debug.json"},
                {
                    "filename": "session.message.txt",
                    "url": "https://example.invalid/session.message.txt",
                },
            ]
        }
    )

    assert attachment == {
        "filename": "session.message.txt",
        "url": "https://example.invalid/session.message.txt",
    }


def test_run_smoke_uses_slash_first_context_dump_and_checks_dump(
    monkeypatch, tmp_path
) -> None:
    FakeDiscordRest.sent_messages = []
    monkeypatch.setattr(smoke, "DiscordRest", FakeDiscordRest)

    report = smoke.run_smoke(
        channel_id="1501008202630696981",
        bot_user_id="bot-1",
        token="token",
        token_kind="bot",
        timeout_seconds=1,
        output_dir=tmp_path,
    )

    assert report.ok is True
    assert FakeDiscordRest.sent_messages[1] == "/context-dump <@bot-1>"
    assert report.checks["dump_has_base_context_version"] is True
    assert report.checks["dump_has_loaded_skill_names"] is True
    assert report.checks["dump_has_source_metadata"] is True
