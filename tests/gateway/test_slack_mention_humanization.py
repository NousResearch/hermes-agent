"""
Tests for Slack inbound mention humanization + bot identity grounding.

Slack delivers user mentions as opaque IDs (``<@U123>``). Passing those to the
agent raw leaves it unable to tell one participant from another — or from
itself — so it can misread a mention of a human as a self-mention and reply to
messages addressed to that person (the reported "bot thinks it's @someone-else"
bug). Two cooperating fixes:

  * ``_humanize_user_mentions`` rewrites ``<@UID>`` → ``@DisplayName`` (the
    Slack equivalent of Discord's ``clean_content``).
  * ``_build_identity_prompt`` returns an ephemeral system-prompt line naming
    the bot's own Slack handle so the agent has a positive "that's me" anchor.
"""

import sys
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Mock slack-bolt if not installed (same pattern as test_slack_mention.py)
# ---------------------------------------------------------------------------

def _ensure_slack_mock():
    if "slack_bolt" in sys.modules and hasattr(sys.modules["slack_bolt"], "__file__"):
        return

    slack_bolt = MagicMock()
    slack_bolt.async_app.AsyncApp = MagicMock
    slack_bolt.adapter.socket_mode.async_handler.AsyncSocketModeHandler = MagicMock

    slack_sdk = MagicMock()
    slack_sdk.web.async_client.AsyncWebClient = MagicMock

    for name, mod in [
        ("slack_bolt", slack_bolt),
        ("slack_bolt.async_app", slack_bolt.async_app),
        ("slack_bolt.adapter", slack_bolt.adapter),
        ("slack_bolt.adapter.socket_mode", slack_bolt.adapter.socket_mode),
        ("slack_bolt.adapter.socket_mode.async_handler",
         slack_bolt.adapter.socket_mode.async_handler),
        ("slack_sdk", slack_sdk),
        ("slack_sdk.web", slack_sdk.web),
        ("slack_sdk.web.async_client", slack_sdk.web.async_client),
    ]:
        sys.modules.setdefault(name, mod)
    sys.modules.setdefault("aiohttp", MagicMock())


_ensure_slack_mock()

import plugins.platforms.slack.adapter as _slack_mod  # noqa: E402
_slack_mod.SLACK_AVAILABLE = True

from plugins.platforms.slack.adapter import SlackAdapter  # noqa: E402


def _make_adapter():
    # object.__new__ skips __init__ (heavy setup) — established slack-test pattern.
    return object.__new__(SlackAdapter)


def _adapter_with_names(names):
    """Adapter whose _resolve_user_name returns from a fixed UID→name map."""
    adapter = _make_adapter()

    async def _resolve(user_id, chat_id="", team_id=""):
        return names.get(user_id, user_id)

    adapter._resolve_user_name = _resolve  # type: ignore[assignment]
    return adapter


# ----- _humanize_user_mentions -------------------------------------------------


@pytest.mark.asyncio
async def test_humanizes_multiple_distinct_mentions():
    adapter = _adapter_with_names(
        {"U07ALICE": "Alice Example", "U07BOB": "Bob Example"}
    )
    out = await adapter._humanize_user_mentions(
        "hey <@U07ALICE> and <@U07BOB>", chat_id="C1"
    )
    assert out == "hey @Alice Example and @Bob Example"


@pytest.mark.asyncio
async def test_handles_labelled_mention_form():
    # Slack sometimes sends <@UID|handle>; only the ID drives resolution.
    adapter = _adapter_with_names({"U07ALICE": "Alice Example"})
    out = await adapter._humanize_user_mentions("<@U07ALICE|alice> hi", chat_id="C1")
    assert out == "@Alice Example hi"


# ----- _build_identity_prompt --------------------------------------------------

def test_identity_prompt_names_the_bot():
    adapter = _make_adapter()
    adapter._bot_display_name = "TestBot"
    adapter._team_bot_names = {}
    prompt = adapter._build_identity_prompt(team_id="T1")
    assert "@TestBot" in prompt
    # Must instruct that another participant's mention is not a self-mention.
    assert "not a mention of you" in prompt


