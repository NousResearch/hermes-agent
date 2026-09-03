"""Slack-specific Collective Wisdom cards and delivery behavior."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import PlatformConfig
from gateway.wisdom_command import (
    WisdomAction,
    WisdomCommandContext,
    WisdomItem,
    WisdomView,
)
from plugins.platforms.slack.adapter import SlackAdapter
from plugins.platforms.slack.wisdom_blocks import (
    render_wisdom_blocks,
    wisdom_fallback_text,
)


def _adapter() -> SlackAdapter:
    adapter = SlackAdapter(PlatformConfig(enabled=True, token="xoxb-test"))
    adapter._app = MagicMock()
    adapter._team_clients = {"T1": AsyncMock()}
    adapter._channel_team = {"D1": "T1", "C1": "T1"}
    return adapter


def test_wisdom_blocks_render_details_and_inline_actions():
    view = WisdomView(
        "Collective Wisdom",
        "Browse team skills",
        items=[
            WisdomItem(
                "release-helper · v3",
                "Coordinates a safe release and reports compatibility.",
                actions=[
                    WisdomAction(
                        "Install", callback_data="wi:cmd:opaque", primary=True
                    ),
                    WisdomAction("View in Portal ↗", url="https://portal.test/s/1"),
                ],
            )
        ],
    )

    blocks = render_wisdom_blocks(view)

    assert [block["type"] for block in blocks] == [
        "header",
        "section",
        "section",
        "actions",
    ]
    assert "Coordinates a safe release" in blocks[2]["text"]["text"]
    buttons = blocks[3]["elements"]
    assert buttons[0]["value"] == "wi:cmd:opaque"
    assert buttons[0]["style"] == "primary"
    assert buttons[1]["url"] == "https://portal.test/s/1"


def test_candidate_card_uses_returning_copy_and_requested_action_order():
    adapter = _adapter()
    view = adapter._wisdom_candidate_view(
        skill_name="Incident Handoff",
        skill_description="Transfer incident context between responders.",
        qualification="high_usage",
        status=(
            "Hermes detected another skill that could be useful to your team.\n\n"
            "Nothing is shared without your approval.\n\n"
            "Would you like to share?"
        ),
        actions=[
            WisdomAction("Not Now", callback_data="wi:defer:event-2"),
            WisdomAction("Review first", callback_data="wi:draft:event-2"),
            WisdomAction(
                "Yes",
                callback_data="wi:publish:event-2",
                primary=True,
            ),
        ],
    )

    blocks = render_wisdom_blocks(view)
    action_block = next(block for block in blocks if block["type"] == "actions")
    assert [button["text"]["text"] for button in action_block["elements"]] == [
        "Not Now",
        "Review first",
        "Yes",
    ]
    assert "Hermes detected *another* skill" in wisdom_fallback_text(view)
    assert "Transfer incident context between responders." in wisdom_fallback_text(view)
    assert "Would you like to share?" in wisdom_fallback_text(view)


@pytest.mark.asyncio
async def test_candidate_not_now_defers_the_slack_prompt_without_declining():
    adapter = _adapter()
    adapter._is_interactive_user_authorized = MagicMock(return_value=True)
    adapter._wisdom_callback_profile = MagicMock(return_value="demo")
    adapter._update_wisdom_interaction = AsyncMock()
    service = MagicMock()
    service.defer_candidate_prompt.return_value = {
        "skill_name": "incident-handoff",
        "qualification": "high_usage",
        "state": "deferred",
    }
    body = {
        "team": {"id": "T1"},
        "channel": {"id": "D1"},
        "user": {"id": "U1", "name": "Shannon"},
        "message": {"ts": "1.2"},
    }

    with patch("hermes_wisdom.service.WisdomService", return_value=service):
        await adapter._handle_wisdom_action(
            AsyncMock(), body, {"value": "wi:defer:event-1"}
        )

    service.defer_candidate_prompt.assert_called_once_with(
        "event-1", surface="slack"
    )
    service.decline_candidate.assert_not_called()
    view = adapter._update_wisdom_interaction.await_args.args[1]
    assert "Not sharing right now" in wisdom_fallback_text(view)


def test_wisdom_blocks_escape_untrusted_skill_text_and_limit_items():
    view = WisdomView(
        "Collective <Wisdom>",
        items=[WisdomItem(f"skill-{index} <here>") for index in range(8)],
    )

    blocks = render_wisdom_blocks(view)

    assert len([block for block in blocks if block["type"] == "section"]) == 5
    assert "&lt;here&gt;" in str(blocks)


def test_wisdom_back_control_is_in_a_separate_first_action_row():
    view = WisdomView(
        "Skill details",
        "A shared skill",
        actions=[WisdomAction("Install", callback_data="wi:cmd:install")],
        navigation_actions=[WisdomAction("← Back", callback_data="wi:cmd:back")],
    )

    blocks = render_wisdom_blocks(view)

    assert [block["type"] for block in blocks] == [
        "header",
        "actions",
        "section",
        "actions",
    ]
    assert [button["text"]["text"] for button in blocks[1]["elements"]] == ["← Back"]
    assert blocks[1]["elements"][0].get("style") is None
    assert blocks[3]["elements"][0]["text"]["text"] == "Install"


@pytest.mark.asyncio
async def test_group_private_action_becomes_user_bound_dm_continuation():
    adapter = _adapter()
    view = WisdomView(
        "Collective Wisdom",
        actions=[
            WisdomAction(
                "Continue in DM",
                "continue_dm",
                {"raw_args": "installed"},
                primary=True,
            )
        ],
    )
    context = WisdomCommandContext(
        user_id="U1",
        chat_id="C1",
        profile="demo",
        organization_id="org-1",
        is_group=True,
    )

    await adapter._prepare_wisdom_view(view, context, team_id="T1", channel_id="C1")

    action = view.actions[0]
    assert action.operation is None
    assert isinstance(action.callback_data, str)
    assert action.callback_data.startswith("wi:continue:")
    assert (
        adapter._wisdom_callback_profile(
            team_id="T1", channel_id="C1", value=action.callback_data
        )
        == "demo"
    )


@pytest.mark.asyncio
async def test_completed_feed_action_preserves_card_and_portal_link():
    adapter = _adapter()
    client = adapter._team_clients["T1"]
    client.chat_update = AsyncMock()
    body = {
        "team": {"id": "T1"},
        "channel": {"id": "D1"},
        "message": {
            "ts": "1.2",
            "text": "Collective Wisdom update",
            "blocks": [
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": "*Update available*\nskill v2"},
                },
                {
                    "type": "actions",
                    "elements": [
                        {
                            "type": "button",
                            "action_id": "hermes_wisdom_feed_0_0",
                            "value": "wi:plan:update:skill-1",
                            "text": {"type": "plain_text", "text": "Update"},
                        },
                        {
                            "type": "button",
                            "action_id": "hermes_wisdom_feed_0_1",
                            "value": "wisdom:portal",
                            "url": "https://portal.test/skill-1",
                            "text": {"type": "plain_text", "text": "View"},
                        },
                    ],
                },
            ],
        },
    }

    changed = await adapter._mark_wisdom_interaction_complete(
        body,
        callback_value="wi:plan:update:skill-1",
        completed_label="Updated v2",
    )

    assert changed is True
    blocks = client.chat_update.call_args.kwargs["blocks"]
    assert "Update available" in str(blocks)
    assert "https://portal.test/skill-1" in str(blocks)
    assert "wi:plan:update:skill-1" not in str(blocks)
    assert "Updated v2" in str(blocks)
