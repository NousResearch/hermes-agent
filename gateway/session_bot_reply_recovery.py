"""Recover a committed local Bot Chat reply without replaying an interrupted relay."""

import logging
import time

logger = logging.getLogger(__name__)


def saved_bot_reply(row):
    """Read the teammate's settled receipt through the trusted process-completion reference."""
    reference = (row['payload'].get('local_automation_v1') or {}).get('bot_reply_receipt')
    if not reference:
        return None

    from gateway.run import _redact_gateway_user_facing_secrets
    from tools.bot_live_delivery import read_delivery_result

    try:
        receipt = read_delivery_result(reference['home'], reference['delivery_id'])
    except (OSError, ValueError):
        logger.warning('Could not read the saved Bot Chat reply', exc_info=True)
        return None
    if not receipt or receipt.get('status') != 'settled' or receipt.get('delivery_id') != reference['delivery_id']:
        return None
    reply = receipt.get('reply')
    if isinstance(reply, str) and reply.strip():
        return _redact_gateway_user_facing_secrets(reply)


def close_with_saved_reply(db, conn, row, reply):
    """Write the recovered presentation inside the interrupted admission's settlement."""
    from hermes_state_local_lineage import local_physical_target
    from hermes_state_runtime import _canonical_chain

    target = _canonical_chain(conn, local_physical_target(conn, row['target_session_id']))[-1]
    # Scope display changes to this relay's own input. Never replace another turn.
    anchor = conn.execute(
        "SELECT id FROM messages WHERE session_id=? AND active=1 AND role='user' "
        "AND display_kind='process_complete' AND content=? ORDER BY id DESC LIMIT 1",
        (target, row['payload']['text']),
    ).fetchone()
    if anchor is not None:
        conn.execute(
            "UPDATE messages SET display_kind='hidden' WHERE session_id=? AND active=1 AND id>? "
            "AND role='assistant' AND (display_kind IS NULL OR display_kind='failed_turn') "
            "AND (tool_calls IS NULL OR tool_calls IN ('', '[]'))",
            (target, anchor[0]),
        )
    else:
        # A crash can occur after the claim but before it writes its user row.
        db._append_messages_in_transaction(conn, target, [{
            'role': 'user', 'content': row['payload']['text'], 'timestamp': time.time(),
            'display_kind': 'process_complete',
        }])
    db._append_messages_in_transaction(conn, target, [{
        'role': 'assistant', 'content': reply, 'timestamp': time.time(),
    }])
