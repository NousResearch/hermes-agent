"""Text-batch delays: Telegram cadence by default, an explicit value honoured.

WhatsApp (5s/10s) and Weixin (3s/5s) used to hold every reply for seconds
before dispatching; the unset default is Telegram's 0.3s (1.0s near a split
chunk) (#44883, #25056). An explicitly configured delay is operator policy and
is used as configured, bounded only by a safety cap that keeps a typo out of
``asyncio.sleep()`` — not clamped to the defaults' neighbourhood (#117104).

WhatsApp and Weixin read ``config.extra``; Telegram reads its env vars, and its
short-message fast tiers apply only while the delay is left at the default.
"""

import logging

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import SessionSource
from gateway.platforms.event import MessageEvent, MessageType
from gateway.platforms.weixin import WeixinAdapter
from plugins.platforms.telegram.adapter import TelegramAdapter
from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

_TELEGRAM_ENV = {
    "text_batch_delay_seconds": "HERMES_TELEGRAM_TEXT_BATCH_DELAY_SECONDS",
    "text_batch_split_delay_seconds": "HERMES_TELEGRAM_TEXT_BATCH_SPLIT_DELAY_SECONDS",
}


@pytest.fixture(params=["whatsapp", "weixin", "telegram"])
def build(request, tmp_path, monkeypatch):
    """Construct the real adapter with the given text-batch settings through its own config surface."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    for env in _TELEGRAM_ENV.values():
        monkeypatch.delenv(env, raising=False)

    def _build(**settings):
        if request.param == "telegram":
            for env in _TELEGRAM_ENV.values():
                monkeypatch.delenv(env, raising=False)
            for key, value in settings.items():
                monkeypatch.setenv(_TELEGRAM_ENV[key], str(value))
            return TelegramAdapter(PlatformConfig(enabled=True, token="t", extra={}))
        cls = WhatsAppAdapter if request.param == "whatsapp" else WeixinAdapter
        return cls(PlatformConfig(enabled=True, extra=dict(settings)))

    _build.platform = request.param
    return _build


_HUGE = 1e9  # far above any sane safety cap


@pytest.mark.parametrize(
    "settings, expected",
    [
        ({"text_batch_delay_seconds": 40}, (40, 40)),  # split is floored at the delay
        ({"text_batch_delay_seconds": 40, "text_batch_split_delay_seconds": 10}, (40, 40)),
        ({"text_batch_delay_seconds": 40, "text_batch_split_delay_seconds": 90}, (40, 90)),
        ({"text_batch_delay_seconds": _HUGE, "text_batch_split_delay_seconds": _HUGE}, ("cap", "cap")),
    ],
    ids=["delay-only", "split-below-delay", "split-above-delay", "above-safety-cap"],
)
def test_explicit_delay_is_honoured_up_to_the_safety_cap(build, settings, expected, caplog):
    with caplog.at_level(logging.WARNING):
        adapter = build(**settings)

    actual = (adapter._text_batch_delay_seconds, adapter._text_batch_split_delay_seconds)
    clamped = expected[0] == "cap"
    if clamped:
        cap = adapter._TEXT_BATCH_SAFETY_CAP_S
        assert cap >= 60  # the cap only guards asyncio.sleep(); it is not a latency policy
        expected = (cap, cap)
    assert actual == expected
    warned = " ".join(r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING).lower()
    assert ("text_batch_delay_seconds" in warned) is clamped


def _short_event(text="hello"):
    event = MessageEvent(text=text, message_type=MessageType.TEXT,
                         source=SessionSource(platform=Platform.WHATSAPP, chat_id="1", chat_type="dm"))
    event._last_chunk_len = len(text)
    return event


@pytest.mark.parametrize("value", [None, "nan", "inf", "-1", "soon"],
                         ids=["unset", "nan", "inf", "negative", "unparseable"])
def test_unset_or_unusable_delay_uses_shared_defaults(build, value):
    """An unusable value behaves exactly like an unset one, including which quiet period a short burst gets."""
    settings = {} if value is None else {"text_batch_delay_seconds": value, "text_batch_split_delay_seconds": value}
    adapter = build(**settings)

    assert adapter._text_batch_delay_seconds == adapter._TEXT_BATCH_DEFAULT_DELAY_S
    assert adapter._text_batch_split_delay_seconds == adapter._TEXT_BATCH_DEFAULT_SPLIT_DELAY_S
    assert adapter._text_batch_delay_for(_short_event()) == build()._text_batch_delay_for(_short_event())


@pytest.mark.parametrize("explicit", [False, True], ids=["unset", "explicit-40"])
@pytest.mark.parametrize("chunk", ["short", "medium", "near-split"])
def test_telegram_fast_tiers_apply_only_to_the_default_delay(monkeypatch, tmp_path, explicit, chunk):
    """Telegram's 0.18s / 0.24s short-message tiers keep the default snappy; an explicit delay is policy."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_TELEGRAM_TEXT_BATCH_SPLIT_DELAY_SECONDS", raising=False)
    if explicit:
        monkeypatch.setenv("HERMES_TELEGRAM_TEXT_BATCH_DELAY_SECONDS", "40")
    else:
        monkeypatch.delenv("HERMES_TELEGRAM_TEXT_BATCH_DELAY_SECONDS", raising=False)
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="t", extra={}))
    text = {"short": "hello", "medium": "x" * 600, "near-split": "x" * adapter._SPLIT_THRESHOLD}[chunk]
    pending = MessageEvent(text=text, message_type=MessageType.TEXT,
                           source=SessionSource(platform=Platform.TELEGRAM, chat_id="1", chat_type="dm"))
    pending._last_chunk_len = len(text)

    waited = adapter._text_batch_delay_for(pending)

    if chunk == "near-split":
        assert waited == adapter._text_batch_split_delay_seconds
    elif explicit:
        assert waited == adapter._text_batch_delay_seconds == 40
    else:
        tier = adapter._TEXT_BATCH_FAST_DELAY_S if chunk == "short" else adapter._TEXT_BATCH_SHORT_DELAY_S
        assert waited == min(adapter._text_batch_delay_seconds, tier) < adapter._text_batch_delay_seconds
