"""Confirmed tool replies belong to one agent turn, never its cached successor."""


def mark_reply_delivered(agent):
    """Call only after the reply's transport acknowledges successful delivery."""
    turn_id = getattr(agent, '_current_turn_id', None)
    if isinstance(turn_id, str) and turn_id:
        agent._reply_delivered_turn_id = turn_id


def consume_reply_delivery(agent, turn_id):
    delivered_turn = getattr(agent, '_reply_delivered_turn_id', None)
    agent._reply_delivered_turn_id = None
    return bool(isinstance(turn_id, str) and turn_id and delivered_turn == turn_id)
