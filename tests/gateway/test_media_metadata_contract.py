"""Contract: media-send overrides must accept the ``metadata`` kwarg.

``BasePlatformAdapter.send_multiple_images`` passes ``metadata=metadata``
to ``send_image`` / ``send_image_file`` / ``send_animation`` on every send.
An override whose signature stops at ``reply_to`` raises ``TypeError:
send_image() got an unexpected keyword argument 'metadata'`` at runtime —
which is exactly how image delivery broke on WhatsApp and email.

This mirrors ``test_discord_media_metadata.py`` but covers the two
adapters that previously slipped, plus a best-effort sweep over every
adapter that imports cleanly so the next slip is caught at test time.
"""

from __future__ import annotations

import importlib
import inspect

import pytest


def _accepts_metadata(method) -> bool:
    params = inspect.signature(method).parameters
    if "metadata" in params:
        return True
    # A ``**kwargs`` catch-all also absorbs metadata (the convention used by
    # WhatsApp's send_video / send_voice / send_document overrides).
    return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


# (module, class) for the two adapters this fix targeted. These must import
# in CI, so assert directly rather than skipping.
@pytest.mark.parametrize(
    "module_name, class_name",
    [
        ("plugins.platforms.whatsapp.adapter", "WhatsAppAdapter"),
        ("plugins.platforms.email.adapter", "EmailAdapter"),
    ],
)
def test_send_image_accepts_metadata(module_name, class_name):
    cls = getattr(importlib.import_module(module_name), class_name)
    assert _accepts_metadata(cls.send_image), (
        f"{class_name}.send_image must accept 'metadata' (or **kwargs) — "
        f"send_multiple_images passes it on every send"
    )


# Best-effort sweep across all shipped adapters. Modules whose optional
# platform SDK isn't installed are skipped; an adapter that imports but
# whose override drops metadata is a hard failure.
_ALL_ADAPTERS = [
    ("gateway.platforms.bluebubbles", "BlueBubblesAdapter"),
    ("plugins.platforms.dingtalk.adapter", "DingTalkAdapter"),
    ("gateway.platforms.discord", "DiscordAdapter"),
    ("plugins.platforms.email.adapter", "EmailAdapter"),
    ("plugins.platforms.feishu.adapter", "FeishuAdapter"),
    ("plugins.platforms.matrix.adapter", "MatrixAdapter"),
    ("gateway.platforms.mattermost", "MattermostAdapter"),
    ("gateway.platforms.signal", "SignalAdapter"),
    ("plugins.platforms.slack.adapter", "SlackAdapter"),
    ("plugins.platforms.telegram.adapter", "TelegramAdapter"),
    ("plugins.platforms.wecom.adapter", "WeComAdapter"),
    ("gateway.platforms.weixin", "WeixinAdapter"),
    ("plugins.platforms.whatsapp.adapter", "WhatsAppAdapter"),
    ("gateway.platforms.yuanbao", "YuanbaoAdapter"),
]


@pytest.mark.parametrize("module_name, class_name", _ALL_ADAPTERS)
def test_all_adapters_send_image_metadata_sweep(module_name, class_name):
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # optional platform dep not installed
        pytest.skip(f"{module_name} not importable: {exc}")
    cls = getattr(module, class_name, None)
    if cls is None or "send_image" not in cls.__dict__:
        pytest.skip(f"{class_name} has no send_image override")
    assert _accepts_metadata(cls.send_image), (
        f"{class_name}.send_image drops the 'metadata' kwarg"
    )
