"""Regression: alert body nested in ``attachments[].blocks`` must reach the agent.

Reproduction (in-house alert webhook, 2026-09-03, macOS launchd gateway):

    text                              mention list only
    attachments[0].fallback           mention list only   <-- not a substitute
    attachments[0].blocks[0] context  ":alert: error"
    attachments[0].blocks[2] context  the alert body      <-- never delivered

Alert apps (Alertmanager, Grafana, PagerDuty, CI bots, in-house webhooks) post
with an empty/mention-only top-level ``text`` and put the real content in Block
Kit blocks nested inside a legacy attachment.

``_handle_slack_message_impl`` already reads the message's *top-level* ``blocks``
with ``_extract_text_from_slack_blocks`` + ``_serialize_slack_blocks_for_agent``,
but its attachment loop read only the six unfurl-shaped keys
(``title``/``title_link``/``from_url``/``text``/``footer``/``fallback``) and
never looked at ``att["blocks"]``. The body was therefore dropped on the live
inbound path, and the agent received the ``fallback`` string in its place.

That last part is what makes it a silent failure rather than an empty message:
when ``fallback`` carries something plausible (here, the mention list), the
agent has no way to tell it apart from a real body.

These tests pin the live inbound path specifically. The thread-history path
(``_extract_text_from_slack_attachments``) is a separate call chain covered by
its own tests.
"""

import asyncio
import importlib
import sys
from importlib.machinery import PathFinder
from types import ModuleType

from gateway.config import PlatformConfig


def _load_installed_package(name):
    if PathFinder.find_spec(name) is None:
        return None
    prefix = f"{name}."
    displaced = {
        m: sys.modules.pop(m)
        for m in tuple(sys.modules)
        if (m == name or m.startswith(prefix)) and not isinstance(sys.modules[m], ModuleType)
    }
    try:
        return importlib.import_module(name)
    except ImportError:
        sys.modules.update(displaced)
        return None


_load_installed_package("slack_bolt")
_load_installed_package("slack_sdk")

_slack_mod = importlib.import_module("plugins.platforms.slack.adapter")
SlackAdapter = _slack_mod.SlackAdapter

CHANNEL = "C06K5S8UV55"
TEAM = "T016HGM8GBY"
BOT = "U0BCLP7DB7B"
USER = "U0374GH838U"
TS = "1788415808.217759"

ALERT_BODY = "[login failed] SCRAP_ALL_SESSION  failed 1 / total 61"
MENTION_ONLY = f"<@{BOT}> <@U03LMEJ59U0>"


def _make_adapter(delivered):
    adapter = SlackAdapter(
        PlatformConfig(
            enabled=True,
            token="xoxb-fake",
            extra={"allow_bots": "all", "require_mention": False},
        )
    )
    adapter._bot_user_id = BOT

    async def _capture(event):
        delivered.append(event)

    adapter.handle_message = _capture

    async def _name(*a, **k):
        return "alertbot"

    adapter._resolve_user_name = _name
    return adapter


def _alert_event(*, fallback=MENTION_ONLY, blocks=None):
    """A webhook alert: body only in the attachment's nested Block Kit blocks."""
    if blocks is None:
        blocks = [
            {"type": "context", "elements": [{"type": "mrkdwn", "text": ":alert: error"}]},
            {"type": "divider"},
            {"type": "context", "elements": [{"type": "mrkdwn", "text": ALERT_BODY}]},
            {"type": "divider"},
        ]
    return {
        "type": "message",
        "subtype": "bot_message",
        "bot_id": "B01ALERTBOT",
        "user": USER,
        "text": MENTION_ONLY,
        "channel": CHANNEL,
        "channel_type": "channel",
        "ts": TS,
        "team": TEAM,
        "blocks": [
            {
                "type": "rich_text",
                "elements": [
                    {
                        "type": "rich_text_section",
                        "elements": [{"type": "user", "user_id": BOT}],
                    }
                ],
            }
        ],
        "attachments": [
            {"id": 1, "color": "d93f0b", "fallback": fallback, "blocks": blocks}
        ],
    }


def _body():
    return {"team_id": TEAM, "event_id": "Ev0BRUTU4GP7"}


def _deliver(event):
    delivered = []
    adapter = _make_adapter(delivered)
    asyncio.run(adapter._handle_slack_message(event, _body()))
    assert delivered, "handler delivered nothing"
    return delivered[0]


class TestAttachmentNestedBlocksOnLiveInbound:
    def test_alert_body_in_attachment_blocks_reaches_the_agent(self):
        """THE regression: the body lives in attachments[0].blocks and must survive."""
        msg = _deliver(_alert_event())
        assert ALERT_BODY in msg.text, (
            "alert body nested in attachments[].blocks never reached the agent; "
            f"got: {msg.text!r}"
        )

    def test_plausible_fallback_does_not_stand_in_for_the_body(self):
        """The silent-failure half.

        A ``fallback`` that looks like content (here, the mention list) must not
        be delivered *instead of* the structured body it summarizes. Asserting
        only on the body's presence would still pass if both were emitted, so
        this pins that the body is what represents the attachment.
        """
        msg = _deliver(_alert_event())
        body_at = msg.text.find(ALERT_BODY)
        assert body_at != -1, "body missing entirely"
        # The attachment section must be represented by its blocks, not by the
        # fallback summary of them.
        assert msg.text.count(MENTION_ONLY) <= 1, (
            "the attachment's fallback was emitted alongside the structured "
            f"body it summarizes; got: {msg.text!r}"
        )

    def test_fallback_still_used_when_attachment_has_no_blocks(self):
        """Do not regress the unfurl case #79051's sibling path relies on."""
        event = _alert_event()
        event["attachments"] = [
            {"id": 1, "fallback": "Grafana: disk usage 91%", "title": "Grafana"}
        ]
        msg = _deliver(event)
        assert "Grafana: disk usage 91%" in msg.text

    def test_link_unfurl_preview_still_rendered(self):
        """Ordinary link unfurls (title/from_url/text) are unaffected."""
        event = _alert_event()
        event["text"] = f"<@{BOT}> see this"
        event["attachments"] = [
            {
                "id": 1,
                "title": "Notion — Q3 plan",
                "from_url": "https://example.com/q3",
                "text": "Preview body of the linked page",
                "fallback": "[no preview available]",
            }
        ]
        msg = _deliver(event)
        assert "Notion — Q3 plan" in msg.text
        assert "Preview body of the linked page" in msg.text

    def test_message_type_attachment_is_still_skipped(self):
        """``is_msg_unfurl`` attachments must not be read, blocks or not."""
        event = _alert_event()
        event["attachments"] = [
            {
                "id": 1,
                "is_msg_unfurl": True,
                "fallback": "echo of our own message",
                "blocks": [
                    {
                        "type": "context",
                        "elements": [{"type": "mrkdwn", "text": "ECHOED_CONTENT"}],
                    }
                ],
            }
        ]
        msg = _deliver(event)
        assert "ECHOED_CONTENT" not in msg.text
        assert "echo of our own message" not in msg.text

    def test_section_and_header_blocks_are_read_too(self):
        """``context`` is one carrier; ``section``/``header`` are the others."""
        event = _alert_event(
            blocks=[
                {"type": "header", "text": {"type": "plain_text", "text": "HEADER_BODY"}},
                {"type": "section", "text": {"type": "mrkdwn", "text": "SECTION_BODY"}},
            ]
        )
        msg = _deliver(event)
        assert "HEADER_BODY" in msg.text
        assert "SECTION_BODY" in msg.text


class TestNestedBlockBudget:
    """Slack allows 20 attachments per message, each with its own blocks.

    The budget is spent across the whole array, not per attachment: otherwise
    one alert could cost many multiples of what the top-level ``blocks`` path
    spends once. Measured before this cap existed, 20 chatty attachments
    produced ~105k characters of message text.
    """

    @staticmethod
    def _chatty_attachment(idx, n_sections=8, chars=400):
        return {
            "id": idx,
            "fallback": "fb",
            "blocks": [
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": f"f{i} " + ("x" * chars)},
                }
                for i in range(n_sections)
            ],
        }

    def test_many_attachments_share_one_budget(self):
        event = _alert_event()
        event["attachments"] = [self._chatty_attachment(i) for i in range(20)]
        msg = _deliver(event)
        # Some framing (headers, separators, the message's own text) sits
        # outside the nested-blocks budget, so allow modest slack over it.
        assert len(msg.text) < _slack_mod._SLACK_ATTACHMENT_BLOCKS_MAX_CHARS * 2, (
            f"20 attachments produced {len(msg.text)} chars; the nested-block "
            "budget is not shared across the attachment array"
        )

    def test_first_attachment_still_carries_its_body(self):
        """The cap must not starve the common single-attachment alert."""
        event = _alert_event()
        event["attachments"] = [
            {"id": 0, "fallback": "fb", "blocks": [
                {"type": "context",
                 "elements": [{"type": "mrkdwn", "text": ALERT_BODY}]},
            ]},
        ] + [self._chatty_attachment(i) for i in range(1, 20)]
        msg = _deliver(event)
        assert ALERT_BODY in msg.text

    def test_nearly_spent_budget_cannot_be_overrun_by_the_next_attachment(self):
        """Adversarial: land the shared budget on a tiny positive remainder.

        ``_serialize_slack_blocks_for_agent`` truncates with
        ``payload[: max_chars - 18]``. Handed a remaining budget below 18, that
        slice index goes negative and Python keeps *almost the whole payload* —
        so a first attachment sized to leave 1..17 chars of budget let the next
        attachment blow straight through the cap. Measured: 26,153 chars
        delivered against a 6,000 budget.
        """
        max_chars = _slack_mod._SLACK_ATTACHMENT_BLOCKS_MAX_CHARS

        def section(n):
            return {"type": "section", "text": {"type": "mrkdwn", "text": "x" * n}}

        # Find a first-attachment size whose serialized form leaves 1..17 chars.
        for n in range(max_chars - 300, max_chars):
            left = max_chars - len(_slack_mod._serialize_slack_blocks_for_agent([section(n)]))
            if 1 <= left <= 17:
                break
        else:  # pragma: no cover - shape of the frame changed; re-derive
            raise AssertionError("could not construct a 1..17 char remainder")

        event = _alert_event()
        event["attachments"] = [
            {"id": 0, "fallback": "fb", "blocks": [section(n)]},
            {"id": 1, "fallback": "fb", "blocks": [section(20_000)]},
        ]
        msg = _deliver(event)
        assert len(msg.text) < max_chars + 500, (
            f"second attachment overran a nearly-spent budget: {len(msg.text)} chars"
        )
