"""Alert content nested in ``message.attachments[0].blocks`` must be read.

Alert apps post the message body as structured Block Kit blocks inside a
legacy attachment: top-level ``text`` empty, no top-level ``blocks``, and
``fallback`` set to "[no preview available]". Attachment-nested blocks were
only read for ``rich_text``, so the fallback stood in as the body and the
agent saw "[no preview available]" for the very alert it was asked to
investigate.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

_repo = str(Path(__file__).resolve().parents[2])
if _repo not in sys.path:
    sys.path.insert(0, _repo)


def _ensure_slack_mock():
    if "slack_bolt" in sys.modules:
        return
    slack_bolt = MagicMock()
    slack_bolt.async_app.AsyncApp = MagicMock
    sys.modules["slack_bolt"] = slack_bolt
    sys.modules["slack_bolt.async_app"] = slack_bolt.async_app
    handler_mod = MagicMock()
    handler_mod.AsyncSocketModeHandler = MagicMock
    sys.modules["slack_bolt.adapter"] = MagicMock()
    sys.modules["slack_bolt.adapter.socket_mode"] = MagicMock()
    sys.modules["slack_bolt.adapter.socket_mode.async_handler"] = handler_mod
    sdk_mod = MagicMock()
    sdk_mod.web = MagicMock()
    sdk_mod.web.async_client = MagicMock()
    sdk_mod.web.async_client.AsyncWebClient = MagicMock
    sys.modules["slack_sdk"] = sdk_mod
    sys.modules["slack_sdk.web"] = sdk_mod.web
    sys.modules["slack_sdk.web.async_client"] = sdk_mod.web.async_client


_ensure_slack_mock()

import plugins.platforms.slack.adapter as _slack_mod  # noqa: E402

_slack_mod.SLACK_AVAILABLE = True

from plugins.platforms.slack.adapter import SlackAdapter  # noqa: E402


NO_PREVIEW = "[no preview available]"


def _section(text):
    return {"type": "section", "text": {"type": "mrkdwn", "text": text}}


# Structure of the reported message; every value is invented, standing in for
# the internal service alert that surfaced the bug.
ALERT_MESSAGE = {
    "ts": "1700000000.000100",
    "bot_id": "B_ALERT",
    "subtype": "bot_message",
    "text": "",
    "blocks": None,
    "attachments": [
        {
            "fallback": NO_PREVIEW,
            "blocks": [
                _section(":rotating_light: *EXAMPLE API(/v1/example) failure*"),
                _section("environment: example-env"),
                _section("servlet_path: /v1/example"),
                {
                    "type": "section",
                    "fields": [
                        {"type": "mrkdwn", "text": "consumer_name: example-consumer"}
                    ],
                },
                {
                    "type": "context",
                    "elements": [
                        {"type": "mrkdwn", "text": "exception_message: example failure"}
                    ],
                },
                {
                    "type": "actions",
                    "elements": [
                        {
                            "type": "button",
                            "text": {"type": "plain_text", "text": "Example Logs"},
                            "url": "https://logs.example.com/q?id=X",
                        }
                    ],
                },
            ],
        }
    ],
}


def test_alert_body_in_attachment_blocks_is_rendered():
    rendered = SlackAdapter._render_message_text(ALERT_MESSAGE)

    for expected in (
        ":rotating_light: *EXAMPLE API(/v1/example) failure*",
        "environment: example-env",
        "servlet_path: /v1/example",
        "consumer_name: example-consumer",
        "exception_message: example failure",
        "Example Logs",
        "https://logs.example.com/q?id=X",
    ):
        assert expected in rendered
    assert NO_PREVIEW not in rendered
