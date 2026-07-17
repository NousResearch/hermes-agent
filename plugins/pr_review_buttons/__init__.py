"""pr_review_buttons — Approve / Request-changes / Comment buttons for PR reviews.

The cron PR-review sweep stages each PR's verbatim review (``hermes prreview
stage``) and posts a single Block Kit digest to the Slack home channel with
three buttons per PR (``hermes prreview publish``). Clicking a button posts the
*exact* staged review to GitHub via ``gh`` — a true what-you-approved gate.

This module only wires the surfaces:

* the ``hermes prreview`` CLI command (stage / publish / list), and
* three Slack Block Kit action handlers (approve / request-changes / comment).

All logic lives in the sibling modules (``store``, ``blocks``, ``github``,
``actions``, ``slackio``, ``cli``).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_ACTION_IDS = (
    "prreview_approve",
    "prreview_request_changes",
    "prreview_comment",
)


def register(ctx) -> None:
    from plugins.pr_review_buttons.cli import prreview_command, register_cli

    ctx.register_cli_command(
        name="prreview",
        help="Stage and post buttoned PR reviews to Slack",
        setup_fn=register_cli,
        handler_fn=lambda args: _exit(prreview_command(args)),
        description=(
            "Operator/agent CLI for the PR-review buttons flow. 'stage' records "
            "one PR's verbatim review; 'publish' posts the buttoned digest to "
            "Slack; buttons post the exact staged review to GitHub on click."
        ),
    )

    # Each Slack button click arrives as (ack, body, action) — the adapter's
    # plugin wrapper does not hand us slack_bolt's client, so we build our own
    # AsyncWebClient from the bot token the gateway already has in its env.
    async def _on_click(ack, body, action):
        from plugins.pr_review_buttons import actions, slackio

        try:
            client = slackio.make_client()
        except Exception as exc:
            logger.error("[pr_review_buttons] cannot build Slack client: %s", exc)
            await ack()
            return
        await actions.handle_action(ack, body, action, client=client)

    for action_id in _ACTION_IDS:
        ctx.register_slack_action_handler(action_id, _on_click)

    logger.debug("pr_review_buttons registered CLI + %d action handlers", len(_ACTION_IDS))


def _exit(code):
    """Normalise a CLI handler return into an exit code (0 when None)."""
    import sys

    sys.exit(int(code or 0))
