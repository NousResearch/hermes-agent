"""Teams side of plugin card actions (contract: ``gateway/plugin_card_actions.py``).

An ``Action.Execute`` click whose data carries ``hermes_plugin_action`` goes to the plugin that
registered that namespace with ``ctx.register_card_action_handler``. Teams waits for the answer
inside the invoke (it shows the replacement card or the text), so the handler gets
:data:`INVOKE_TIMEOUT_S`. The bot's allowlist gate (``TeamsAdapter._card_action_denied``, default
deny) runs first; it is necessary, not sufficient: the plugin still authorizes ``event.user_id``.
"""

from __future__ import annotations

import logging
from typing import Any

from agent.i18n import t
from gateway.plugin_card_actions import ROUTING_KEY, action_failed_text, run_card_action

logger = logging.getLogger(__name__)

# The SDK has no invoke deadline constant; Teams gives up on an unanswered invoke after about 10 s,
# so a slow handler is answered with the failure text before that.
INVOKE_TIMEOUT_S = 8.0


async def answer_plugin_card_action(adapter: Any, activity: Any, data: dict) -> Any:
    """The invoke response for a click on a plugin's button."""
    denied = adapter._card_action_denied(activity.from_)
    if denied:
        return adapter._invoke_message(denied)
    # The card's message is what the click answers: ctx.post_message replies there.
    message_id = getattr(activity, "reply_to_id", None) or ""
    source = adapter._source_for_activity(activity, message_id or None)
    if adapter._canonicalize(source) is None and getattr(source, "profile_route_rejected", False) is True:
        from gateway.parked_route_refusal import drop_rejected_route
        drop_rejected_route(adapter, source)
        return adapter._invoke_message(action_failed_text())
    namespace = data.get(ROUTING_KEY)
    result = await run_card_action(adapter.gateway_runner, source, namespace, data, message_id=message_id,
                                   timeout=INVOKE_TIMEOUT_S)
    if result is None:
        return adapter._invoke_message(t("platform.teams.approval.unknown_action"))
    if result.card is None:
        return adapter._invoke_message(result.text)
    try:
        return adapter._invoke_raw_card(result.card)
    except ValueError as exc:  # pydantic's ValidationError: not an Adaptive Card the SDK accepts
        logger.warning("[teams] replacement card for namespace %s refused by the SDK: %s", namespace,
                       type(exc).__name__)
        return adapter._invoke_message(action_failed_text())
