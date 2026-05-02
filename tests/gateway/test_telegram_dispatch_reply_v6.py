"""Tests for v6 dispatch reply enforcement.

Covers ``GatewayRunner._v6_maybe_inject_dispatcher`` — injects
``@<dispatcher>`` into the agent's FINAL response when completing a
bot-to-bot ``[派单]`` dispatch. Runs entirely at the runner level
(response-return seam); tool-echo messages emitted during the agent
run are intentionally outside this hook's reach.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gateway.config import Platform
from gateway.run import GatewayRunner


BOT_USERNAME = "hm_xiaoxiong_content_bot"        # this bot (染墨)
DISPATCHER_USERNAME = "hm_xiaoxiong_main_bot"    # dispatcher (知微)
STRANGER_BOT_USERNAME = "hm_xiaoxiong_dev_bot"   # a third teammate


def _make_runner():
    """Create a GatewayRunner stub with a mocked Telegram adapter."""
    runner = GatewayRunner.__new__(GatewayRunner)
    adapter = MagicMock()
    adapter._bot = MagicMock()
    adapter._bot.username = BOT_USERNAME
    runner.adapters = {Platform.TELEGRAM: adapter}
    return runner


def _make_event(text: str = "", is_bot: bool = True, username: str = DISPATCHER_USERNAME):
    """Build a minimal MessageEvent stand-in with a telegram.Message-ish raw."""
    from_user = SimpleNamespace(is_bot=is_bot, username=username)
    raw = SimpleNamespace(from_user=from_user, text=text, caption=None)
    return SimpleNamespace(raw_message=raw)


def _make_source(platform: Platform = Platform.TELEGRAM):
    return SimpleNamespace(platform=platform)


def test_inject_when_response_missing_at():
    """The real-world bug: subagent forgot to @-back — v6 injects."""
    runner = _make_runner()
    event = _make_event(f"[派单] @{BOT_USERNAME} 写 200 字 SEO 文案")
    out = runner._v6_maybe_inject_dispatcher(
        "找新词是内容创作的核心技能...", event, _make_source()
    )
    assert out.startswith(f"@{DISPATCHER_USERNAME} ")
    assert "找新词是内容创作的核心技能..." in out


def test_no_inject_when_response_has_at_at_start():
    """Model already @-ed dispatcher at the start — leave alone."""
    runner = _make_runner()
    event = _make_event(f"[派单] @{BOT_USERNAME} 写 X")
    original = f"@{DISPATCHER_USERNAME} [完成] 交付：..."
    assert runner._v6_maybe_inject_dispatcher(original, event, _make_source()) == original


def test_no_inject_when_response_has_at_in_middle():
    """Model @-ed dispatcher mid-text — still skip (whole-text scan)."""
    runner = _make_runner()
    event = _make_event(f"[派单] @{BOT_USERNAME} 写 X")
    original = f"[完成] @{DISPATCHER_USERNAME} 交付：..."
    assert runner._v6_maybe_inject_dispatcher(original, event, _make_source()) == original


def test_no_inject_when_response_has_at_case_insensitive():
    """Case-insensitive mention match."""
    runner = _make_runner()
    event = _make_event(f"[派单] @{BOT_USERNAME} 写 X")
    original = f"@{DISPATCHER_USERNAME.upper()} [完成] hi"
    assert runner._v6_maybe_inject_dispatcher(original, event, _make_source()) == original


def test_inject_when_existing_at_is_prefix_match_only():
    """``@dispatcher_assistant`` must NOT match whole-token ``@dispatcher``."""
    runner = _make_runner()
    event = _make_event(f"[派单] @{BOT_USERNAME} do X")
    original = f"@{DISPATCHER_USERNAME}_assistant did something"
    out = runner._v6_maybe_inject_dispatcher(original, event, _make_source())
    assert out.startswith(f"@{DISPATCHER_USERNAME} ")


def test_no_inject_when_trigger_is_user():
    """v6 must only fire on bot-to-bot dispatches."""
    runner = _make_runner()
    event = _make_event(f"[派单] @{BOT_USERNAME} 写 X", is_bot=False, username="Kit668")
    out = runner._v6_maybe_inject_dispatcher("写好了，在群里看看?", event, _make_source())
    assert not out.startswith("@Kit668")
    assert out == "写好了，在群里看看?"


def test_no_inject_when_trigger_lacks_tag():
    """Without [派单] tag, v6 is dormant — breaks the loop path."""
    runner = _make_runner()
    event = _make_event(f"@{BOT_USERNAME} 刚才那个 fix 还行")
    out = runner._v6_maybe_inject_dispatcher("嗯，老板要的话可以推", event, _make_source())
    assert "@" + DISPATCHER_USERNAME not in out


def test_no_inject_on_self_dispatch():
    """Never inject @self — defensive against pathological configurations."""
    runner = _make_runner()
    event = _make_event(f"[派单] @{BOT_USERNAME} 自己派自己", username=BOT_USERNAME)
    out = runner._v6_maybe_inject_dispatcher("跑一下", event, _make_source())
    assert not out.startswith(f"@{BOT_USERNAME}")


def test_inject_tolerates_leading_whitespace():
    """Dispatcher accidentally wrote ``  [派单] @bot ...`` with leading spaces."""
    runner = _make_runner()
    event = _make_event(f"   [派单] @{BOT_USERNAME} 做一下")
    out = runner._v6_maybe_inject_dispatcher("好了，这是交付内容", event, _make_source())
    assert out.startswith(f"@{DISPATCHER_USERNAME} ")


def test_tag_must_be_at_start_of_trigger_text():
    """Only leading [派单] activates — [派单] in the middle is NOT a dispatch."""
    runner = _make_runner()
    event = _make_event("我们昨天讨论了[派单]的规则")
    out = runner._v6_maybe_inject_dispatcher(
        "收到，[派单] 规则我记住了", event, _make_source()
    )
    assert not out.startswith(f"@{DISPATCHER_USERNAME}")


def test_no_inject_when_response_empty():
    """Empty response passes through unchanged."""
    runner = _make_runner()
    event = _make_event(f"[派单] @{BOT_USERNAME} do X")
    assert runner._v6_maybe_inject_dispatcher("", event, _make_source()) == ""


def test_no_inject_when_event_is_none():
    """Missing event (synthetic triggers / CLI) — pass through."""
    runner = _make_runner()
    out = runner._v6_maybe_inject_dispatcher("hello", None, _make_source())
    assert out == "hello"


def test_no_inject_when_raw_message_is_none():
    """Event without raw_message (e.g. internal synthetic events) — pass through."""
    runner = _make_runner()
    event = SimpleNamespace(raw_message=None)
    out = runner._v6_maybe_inject_dispatcher("hello", event, _make_source())
    assert out == "hello"


def test_no_inject_on_non_telegram_platform():
    """Discord/Slack etc. not handled yet; guard is Telegram-only."""
    runner = _make_runner()
    event = _make_event(f"[派单] @{BOT_USERNAME} do X")
    source = SimpleNamespace(platform=Platform.DISCORD)
    out = runner._v6_maybe_inject_dispatcher("hello world", event, source)
    assert out == "hello world"


def test_inject_uses_caption_when_text_missing():
    """For media messages, trigger text lives in .caption — should still work."""
    runner = _make_runner()
    from_user = SimpleNamespace(is_bot=True, username=DISPATCHER_USERNAME)
    raw = SimpleNamespace(from_user=from_user, text=None, caption=f"[派单] @{BOT_USERNAME} 做一下")
    event = SimpleNamespace(raw_message=raw)
    out = runner._v6_maybe_inject_dispatcher("好了", event, _make_source())
    assert out.startswith(f"@{DISPATCHER_USERNAME} ")
