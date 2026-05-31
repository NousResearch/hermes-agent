import importlib

import pytest

from gateway.platforms.base import BasePlatformAdapter, MediaKind, classify_media_kind

I, V, A, D = MediaKind.IMAGE, MediaKind.VIDEO, MediaKind.VOICE, MediaKind.DOCUMENT
FULL = frozenset({I, V, A, D})

# The single source of truth for what each adapter natively delivers. A kind
# listed here MUST be backed by a real send_* override; the descriptor exists
# so dispatch sites can skip-and-warn rather than leak a path as chat text.
PINNED_MEDIA_KINDS = {
    "gateway.platforms.telegram:TelegramAdapter": FULL,
    "gateway.platforms.slack:SlackAdapter": FULL,
    "gateway.platforms.signal:SignalAdapter": FULL,
    "gateway.platforms.matrix:MatrixAdapter": FULL,
    "gateway.platforms.whatsapp:WhatsAppAdapter": FULL,
    "gateway.platforms.wecom:WeComAdapter": FULL,
    "gateway.platforms.bluebubbles:BlueBubblesAdapter": FULL,
    "gateway.platforms.feishu:FeishuAdapter": FULL,
    "gateway.platforms.qqbot.adapter:QQAdapter": FULL,
    "gateway.platforms.weixin:WeixinAdapter": FULL,
    "gateway.platforms.email:EmailAdapter": frozenset({I, D}),
    "gateway.platforms.yuanbao:YuanbaoAdapter": frozenset({I, D}),
    "gateway.platforms.dingtalk:DingTalkAdapter": frozenset(),
    "gateway.platforms.sms:SmsAdapter": frozenset(),
    "gateway.platforms.homeassistant:HomeAssistantAdapter": frozenset(),
    "gateway.platforms.webhook:WebhookAdapter": frozenset(),
    "gateway.platforms.msgraph_webhook:MSGraphWebhookAdapter": frozenset(),
    "plugins.platforms.discord.adapter:DiscordAdapter": FULL,
    "plugins.platforms.google_chat.adapter:GoogleChatAdapter": FULL,
    "plugins.platforms.mattermost.adapter:MattermostAdapter": FULL,
    "plugins.platforms.line.adapter:LineAdapter": frozenset({I, V, A}),
    "plugins.platforms.teams.adapter:TeamsAdapter": frozenset({I}),
    "plugins.platforms.simplex.adapter:SimplexAdapter": frozenset(),
    "plugins.platforms.irc.adapter:IRCAdapter": frozenset(),
    "plugins.platforms.ntfy.adapter:NtfyAdapter": frozenset(),
}


@pytest.mark.parametrize("ref,expected", PINNED_MEDIA_KINDS.items())
def test_media_kinds_pinned(ref, expected):
    mod, cls = ref.split(":")
    adapter_cls = getattr(importlib.import_module(mod), cls)
    assert adapter_cls.MEDIA_KINDS == expected


def test_media_kind_has_four_members():
    assert {k.name for k in MediaKind} == {"IMAGE", "VIDEO", "VOICE", "DOCUMENT"}


def test_base_default_is_fail_closed_empty():
    assert BasePlatformAdapter.MEDIA_KINDS == frozenset()


def test_classify_image_video_document():
    assert classify_media_kind("/x/a.png", platform="qqbot") is MediaKind.IMAGE
    assert classify_media_kind("/x/a.mp4", platform="qqbot") is MediaKind.VIDEO
    assert classify_media_kind("/x/a.pdf", platform="qqbot") is MediaKind.DOCUMENT


def test_classify_audio_routes_to_voice_on_non_telegram():
    assert classify_media_kind("/x/a.mp3", platform="slack") is MediaKind.VOICE
    assert classify_media_kind("/x/a.ogg", is_voice=True, platform="slack") is MediaKind.VOICE


def test_classify_force_document_overrides_image():
    assert classify_media_kind("/x/a.png", platform="qqbot", force_document=True) is MediaKind.DOCUMENT
