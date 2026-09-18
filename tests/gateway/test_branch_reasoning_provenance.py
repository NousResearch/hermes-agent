"""Messaging-gateway branches preserve reasoning replay provenance."""

from gateway.slash_commands_session import _branch_row
from hermes_state import SessionDB


def test_gateway_branch_round_trips_reasoning_route(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    route = "same-route-provenance"
    try:
        db.create_session("branch", source="telegram")
        db.append_messages_batch(
            "branch",
            [
                _branch_row(
                    {
                        "role": "assistant",
                        "content": "visible reply",
                        "reasoning": "private trace",
                        "reasoning_content": "private content",
                        "reasoning_details": [{"text": "private details"}],
                        "_reasoning_route": route,
                        "anthropic_content_blocks": [
                            {"type": "thinking", "signature": "sig"}
                        ],
                        "bedrock_content_blocks": [
                            {"reasoningContent": "signed"}
                        ],
                    }
                )
            ],
        )

        [assistant] = db.get_messages_as_conversation("branch")
        assert assistant["reasoning"] == "private trace"
        assert assistant["reasoning_content"] == "private content"
        assert assistant["reasoning_details"] == [{"text": "private details"}]
        assert assistant["_reasoning_route"] == route
        assert assistant["anthropic_content_blocks"] == [
            {"type": "thinking", "signature": "sig"}
        ]
        assert assistant["bedrock_content_blocks"] == [
            {"reasoningContent": "signed"}
        ]
    finally:
        db.close()
