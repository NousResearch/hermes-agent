from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

from gateway.route_banner import format_context_window_token_line
from gateway.startup_context import (
    discord_api_context_row,
    discord_api_timeout,
    discord_context_channel_id,
    render_reply_chain_payload,
)


@dataclass
class _Source:
    chat_id: str
    parent_chat_id: str = ""
    thread_id: str | None = None


class _Module:
    def external_context_row(self, **kwargs):
        return kwargs

    def is_noise(self, row):
        return row.get("content") == "noise"


def test_discord_context_channel_id_uses_parent_for_fresh_auto_thread() -> None:
    event = SimpleNamespace(raw_message=SimpleNamespace(channel=SimpleNamespace(id=123)))
    source = _Source(chat_id="456", parent_chat_id="123", thread_id="456")

    assert discord_context_channel_id(event, source) == "123"


def test_discord_context_channel_id_uses_thread_for_existing_thread() -> None:
    event = SimpleNamespace(raw_message=SimpleNamespace(channel=SimpleNamespace(id=456)))
    source = _Source(chat_id="456", parent_chat_id="123", thread_id="456")

    assert discord_context_channel_id(event, source) == "456"


def test_discord_api_timeout_is_clamped(monkeypatch) -> None:
    monkeypatch.setenv("HERMES_DISCORD_CONTEXT_API_TIMEOUT", "99")

    assert discord_api_timeout() == 8.0


def test_discord_api_context_row_uses_attachment_fallback_and_filters_noise() -> None:
    module = _Module()
    message = SimpleNamespace(
        id=1,
        content="",
        channel=SimpleNamespace(id=2),
        author=SimpleNamespace(id=3, display_name="Nick", name="nick", bot=False),
        attachments=[SimpleNamespace(filename="diagram.png")],
        reference=SimpleNamespace(message_id=4),
        created_at=SimpleNamespace(isoformat=lambda: "2026-06-21T00:00:00"),
    )

    row = discord_api_context_row(module, message)

    assert row is not None
    assert row["content"] == "(attachment: diagram.png)"
    assert row["reply_to_id"] == 4
    assert row["author"] == "Nick"

    message.content = "noise"
    assert discord_api_context_row(module, message) is None


def test_render_reply_chain_payload_root_to_leaf_with_truncation() -> None:
    rendered = render_reply_chain_payload(
        {
            "reply_chain": [
                {"timestamp": "t1", "author": "A", "id": "1", "content": "root"},
                {"timestamp": "t2", "author": "B", "id": "2", "content": "leaf"},
            ],
            "reply_chain_truncated": True,
        }
    )

    assert rendered.startswith("## Discord Reply Chain")
    assert rendered.index("root") < rendered.index("leaf")
    assert "[older reply-chain messages omitted]" in rendered


def test_route_banner_context_token_line_includes_context_version_and_skills() -> None:
    line = format_context_window_token_line(
        approx_tokens=250,
        context_length=1000,
        threshold_tokens=800,
        base_context_version="gateway-context-v1",
        loaded_skill_names=["coding"],
    )

    assert "gateway-context-v1" in line
    assert "skills: coding" in line
    assert "~250 / 1,000" in line
    assert "compress at ~800" in line
