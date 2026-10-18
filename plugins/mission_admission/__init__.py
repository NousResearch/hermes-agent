"""Gateway fast-path for Fred missions; the core turn prologue is the mandatory backstop."""
from __future__ import annotations

import logging

from hermes_cli import mission_ownership as missions
from hermes_cli.kanban_db_connect import connect_closing

_LOG = logging.getLogger(__name__)


def _admit_message(*, session_key: str = "", platform: str = "", message_id: str = "",
                   text: str = "", **_kwargs):
    if platform != "telegram" or not missions.is_substantial_objective(text):
        return None
    try:
        with connect_closing() as conn:
            mission = missions.ensure_turn_mission(
                conn, session_id=session_key, text=text,
                ingress_id=missions.ingress_key(
                    session_id=session_key, platform=platform,
                    delivery_id=message_id or None,
                ),
            )
        if mission is None:
            return None
        return None
    except Exception as exc:
        _LOG.exception("Fred mission admission failed for Telegram message %s", message_id)
        return {"action": "handled", "reply": (
            "I could not establish or resume the durable mission, so I have not started "
            f"the objective. Admission error: {type(exc).__name__}."
        )}


def register(ctx) -> None:
    ctx.register_hook("post_gateway_admission", _admit_message)
