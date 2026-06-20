"""Tests for resolve_topic_prompt: per-topic prompt lookup in telegram group_topics config."""

import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


# ── Fake telegram module tree so importing the adapter module works ──
# Mirrors the pattern in test_telegram_thread_fallback.py. The full
# import surface of gateway/platforms/telegram.py must be satisfied —
# `from telegram.constants import ParseMode, ChatType` will explode at
# import time if either name is missing from the fake.
_fake_telegram = types.ModuleType("telegram")
_fake_telegram.Update = object
_fake_telegram.Bot = object
_fake_telegram.Message = object
_fake_telegram.InlineKeyboardButton = object
_fake_telegram.InlineKeyboardMarkup = object


class _FakeNetworkError(Exception):
    pass


class _FakeBadRequest(_FakeNetworkError):
    pass


class _FakeTimedOut(_FakeNetworkError):
    pass


class _FakeRetryAfter(Exception):
    def __init__(self, seconds):
        super().__init__(f"Retry after {seconds}")
        self.retry_after = seconds


_fake_telegram_error = types.ModuleType("telegram.error")
_fake_telegram_error.NetworkError = _FakeNetworkError
_fake_telegram_error.BadRequest = _FakeBadRequest
_fake_telegram_error.TimedOut = _FakeTimedOut
_fake_telegram_error.RetryAfter = _FakeRetryAfter
_fake_telegram_error.TelegramError = Exception
_fake_telegram.error = _fake_telegram_error

_fake_telegram_constants = types.ModuleType("telegram.constants")
_fake_telegram_constants.ParseMode = SimpleNamespace(MARKDOWN_V2="MarkdownV2", HTML="HTML")
_fake_telegram_constants.ChatType = SimpleNamespace(
    GROUP="group",
    SUPERGROUP="supergroup",
    CHANNEL="channel",
    PRIVATE="private",
)
_fake_telegram_constants.ReactionEmoji = type("ReactionEmoji", (), {})
_fake_telegram.constants = _fake_telegram_constants

_fake_telegram_ext = types.ModuleType("telegram.ext")
_fake_telegram_ext.Application = object
_fake_telegram_ext.ApplicationBuilder = object
_fake_telegram_ext.CommandHandler = object
_fake_telegram_ext.CallbackQueryHandler = object
_fake_telegram_ext.MessageHandler = object
_fake_telegram_ext.ContextTypes = SimpleNamespace(DEFAULT_TYPE=object)
_fake_telegram_ext.filters = object

_fake_telegram_request = types.ModuleType("telegram.request")
_fake_telegram_request.HTTPXRequest = object

sys.modules.setdefault("telegram", _fake_telegram)
sys.modules.setdefault("telegram.constants", _fake_telegram_constants)
sys.modules.setdefault("telegram.error", _fake_telegram_error)
sys.modules.setdefault("telegram.ext", _fake_telegram_ext)
sys.modules.setdefault("telegram.request", _fake_telegram_request)


from gateway.platforms.telegram import resolve_topic_prompt  # noqa: E402

# Note: if `from gateway.platforms.telegram import resolve_topic_prompt`
# raises an ImportError mentioning a name other than `resolve_topic_prompt`
# (e.g. `cannot import name 'X' from 'telegram.constants'`), telegram.py's
# import surface has grown — extend the fake module tree above to match.


class TestResolveTopicPrompt:
    def test_returns_prompt_for_matching_chat_and_thread(self):
        extra = {
            "group_topics": [
                {
                    "chat_id": "-1003742888118",
                    "topics": [
                        {"name": "ops", "thread_id": "5", "prompt": "Ops prompt"},
                    ],
                },
            ]
        }
        assert resolve_topic_prompt(extra, "-1003742888118", "5") == "Ops prompt"

    def test_returns_none_when_no_thread_id(self):
        # DM or non-forum group: no thread_id → no per-topic lookup applies.
        extra = {
            "group_topics": [
                {
                    "chat_id": "-1003742888118",
                    "topics": [{"name": "ops", "thread_id": "5", "prompt": "Ops"}],
                },
            ]
        }
        assert resolve_topic_prompt(extra, "-1003742888118", None) is None

    def test_returns_none_when_chat_id_misses(self):
        extra = {
            "group_topics": [
                {
                    "chat_id": "-1003742888118",
                    "topics": [{"name": "ops", "thread_id": "5", "prompt": "Ops"}],
                },
            ]
        }
        assert resolve_topic_prompt(extra, "-9999999999999", "5") is None

    def test_returns_none_when_thread_id_misses_in_matching_group(self):
        extra = {
            "group_topics": [
                {
                    "chat_id": "-1003742888118",
                    "topics": [{"name": "ops", "thread_id": "5", "prompt": "Ops"}],
                },
            ]
        }
        assert resolve_topic_prompt(extra, "-1003742888118", "999") is None

    def test_handles_int_typed_yaml_keys(self):
        # YAML loader may produce int chat_id and int thread_id inside
        # nested list-of-dicts (config.py:589-594 only normalises flat
        # channel_prompts keys, not group_topics).
        extra = {
            "group_topics": [
                {
                    "chat_id": -1003742888118,
                    "topics": [{"name": "ops", "thread_id": 5, "prompt": "Ops"}],
                },
            ]
        }
        assert resolve_topic_prompt(extra, "-1003742888118", "5") == "Ops"

    def test_returns_none_for_missing_group_topics(self):
        assert resolve_topic_prompt({}, "-1001", "5") is None
        assert resolve_topic_prompt({"group_topics": None}, "-1001", "5") is None
        assert resolve_topic_prompt({"group_topics": "not-a-list"}, "-1001", "5") is None

    def test_returns_none_for_malformed_group_entry(self):
        # Skip-and-continue: malformed entries do not crash the resolver.
        extra = {
            "group_topics": [
                None,
                "not-a-dict",
                {"chat_id": "-1001"},  # missing topics
                {"chat_id": "-1002", "topics": "not-a-list"},
                {
                    "chat_id": "-1003",
                    "topics": [
                        None,
                        "not-a-dict",
                        {"thread_id": "5", "prompt": "Match"},
                    ],
                },
            ]
        }
        assert resolve_topic_prompt(extra, "-1003", "5") == "Match"

    def test_returns_none_for_blank_prompt(self):
        # Whitespace-only prompts fall through, matching resolve_channel_prompt's convention.
        extra = {
            "group_topics": [
                {
                    "chat_id": "-1001",
                    "topics": [{"thread_id": "5", "prompt": "   "}],
                },
            ]
        }
        assert resolve_topic_prompt(extra, "-1001", "5") is None

    def test_disambiguates_colliding_thread_ids(self):
        # Proves #13256's collision shape cannot occur via this path:
        # same thread_id in two different supergroups returns distinct prompts.
        extra = {
            "group_topics": [
                {
                    "chat_id": "-1003742888118",
                    "topics": [{"thread_id": "5", "prompt": "Group A invoices"}],
                },
                {
                    "chat_id": "-1003953149701",
                    "topics": [{"thread_id": "5", "prompt": "Group B design"}],
                },
            ]
        }
        assert resolve_topic_prompt(extra, "-1003742888118", "5") == "Group A invoices"
        assert resolve_topic_prompt(extra, "-1003953149701", "5") == "Group B design"

    def test_multiline_prompt_preserved(self):
        # YAML literal-block prompts arrive with embedded newlines.
        multiline = "Line one.\nLine two.\nLine three."
        extra = {
            "group_topics": [
                {
                    "chat_id": "-1001",
                    "topics": [{"thread_id": "5", "prompt": multiline}],
                },
            ]
        }
        assert resolve_topic_prompt(extra, "-1001", "5") == multiline


class TestCallSitePrecedence:
    """Caller-contract tests: verify the exact two-step resolution order
    that _event_from_message implements (resolve_topic_prompt first,
    resolve_channel_prompt as fallback). These are *not* end-to-end
    integration tests — they replicate the helper-call sequence rather
    than driving a fake telegram.Message through _event_from_message,
    which would require a much heavier mock surface for marginal gain.
    The full integration is exercised by the existing telegram suite
    (test_telegram_*.py) running with this branch's call-site change."""

    def _resolve_via_call_site(self, extra: dict, chat_id: str, thread_id: str | None) -> str | None:
        # Mirror exactly the lines telegram._event_from_message executes.
        from gateway.platforms.telegram import resolve_topic_prompt
        from gateway.platforms.base import resolve_channel_prompt

        prompt = resolve_topic_prompt(extra, chat_id, thread_id)
        if not prompt:
            prompt = resolve_channel_prompt(
                extra,
                thread_id or chat_id,
                chat_id if thread_id else None,
            )
        return prompt

    def test_per_topic_prompt_used_when_set(self):
        extra = {
            "group_topics": [
                {
                    "chat_id": "-1003742888118",
                    "topics": [{"thread_id": "5", "prompt": "Per-topic"}],
                },
            ],
            "channel_prompts": {"5": "Flat fallback"},
        }
        assert self._resolve_via_call_site(extra, "-1003742888118", "5") == "Per-topic"

    def test_falls_back_to_channel_prompts_when_no_per_topic(self):
        extra = {
            "group_topics": [
                {
                    "chat_id": "-1003742888118",
                    "topics": [{"thread_id": "6", "prompt": "Different topic"}],
                },
            ],
            "channel_prompts": {"5": "Flat for thread 5"},
        }
        # No per-topic prompt for thread 5 → falls back to channel_prompts["5"].
        assert self._resolve_via_call_site(extra, "-1003742888118", "5") == "Flat for thread 5"

    def test_per_topic_wins_over_channel_prompts_for_same_topic(self):
        extra = {
            "group_topics": [
                {
                    "chat_id": "-1003742888118",
                    "topics": [{"thread_id": "5", "prompt": "Per-topic wins"}],
                },
            ],
            "channel_prompts": {"5": "Flat would win without group_topics", "-1003742888118:5": "Composite would also lose"},
        }
        assert self._resolve_via_call_site(extra, "-1003742888118", "5") == "Per-topic wins"
