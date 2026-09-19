"""Telegram-only provider visibility for the gateway model picker."""

from types import SimpleNamespace

import pytest

from gateway.config import Platform
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from plugins.platforms.telegram.adapter import _apply_yaml_config


_ALL_ROWS = [
    {
        "slug": "google-ai-studio",
        "name": "Gemini",
        "is_current": True,
        "models": ["m1"],
        "total_models": 1,
    },
    {"slug": "explicit", "name": "Explicit", "is_current": False, "models": ["m2"], "total_models": 1},
    {"slug": "ambient", "name": "Ambient", "is_current": False, "models": ["m3"], "total_models": 1},
]


class _CapturingPicker:
    def __init__(self, extra):
        self.config = SimpleNamespace(extra=extra)
        self.providers = None

    async def send_model_picker(self, **kwargs):
        self.providers = kwargs["providers"]
        return SimpleNamespace(success=True)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("platform", "extra", "current_provider", "rows", "expected_slugs"),
    [
        (
            Platform.TELEGRAM,
            {},
            "google",
            _ALL_ROWS,
            ["google-ai-studio", "explicit", "ambient"],
        ),
        (
            Platform.TELEGRAM,
            {"show_all_providers": False},
            "google",
            _ALL_ROWS,
            ["google-ai-studio", "explicit"],
        ),
        (
            Platform.TELEGRAM,
            {"show_all_providers": False},
            "google",
            _ALL_ROWS[1:],
            ["explicit", "gemini"],
        ),
        (
            Platform.TELEGRAM,
            {"show_all_providers": False},
            2070,
            [
                {
                    "slug": "2070",
                    "name": "2070",
                    "is_current": True,
                    "models": ["m1"],
                    "total_models": 1,
                },
                *_ALL_ROWS[1:],
            ],
            ["2070", "explicit"],
        ),
        (
            Platform.TELEGRAM,
            {"show_all_providers": "false"},
            "google",
            _ALL_ROWS,
            ["google-ai-studio", "explicit", "ambient"],
        ),
        (
            Platform.DISCORD,
            {"show_all_providers": False},
            "google",
            _ALL_ROWS,
            ["google-ai-studio", "explicit", "ambient"],
        ),
    ],
)
async def test_picker_provider_visibility_is_exact_and_telegram_only(
    monkeypatch, platform, extra, current_provider, rows, expected_slugs
):
    source = SessionSource(platform=platform, chat_id="chat", chat_type="dm")
    event = MessageEvent(text="/model", message_type=MessageType.TEXT, source=source)
    adapter = _CapturingPicker(extra)
    runner = object.__new__(GatewayRunner)
    runner._thread_metadata_for_source = lambda *_args: None
    runner._reply_anchor_for_event = lambda *_args: None

    monkeypatch.setattr(
        "hermes_cli.model_switch_providers.list_picker_providers",
        lambda **_kwargs: list(rows),
    )
    monkeypatch.setattr(
        "hermes_cli.auth.is_provider_explicitly_configured",
        lambda slug: slug == "explicit",
    )

    sent = await runner._send_model_picker(
        event,
        source,
        adapter,
        "session",
        {
            "current_provider": current_provider,
            "current_base_url": "",
            "current_model": "m1",
            "user_providers": {},
            "custom_providers": [],
            "excluded_providers": [],
        },
        lambda *_args: None,
    )

    assert sent is True
    assert [row["slug"] for row in adapter.providers] == expected_slugs
    assert any(row["is_current"] for row in adapter.providers)


def test_telegram_yaml_bridge_preserves_show_all_providers_value():
    assert _apply_yaml_config({}, {"show_all_providers": False}) == {
        "show_all_providers": False
    }
