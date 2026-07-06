"""The chat run.completed transcript carries decoded per-message tokens.

_turn_transcript_messages → _message_response must surface a `tokens` bucket
for each assistant/tool row (decoded from raw packed token_count), so web /
API chat clients can render a per-message breakdown.
"""
from gateway.platforms.api_server import APIServerAdapter
from hermes_token_codec import pack_input_tokens, pack_assistant_tokens


def test_turn_transcript_surfaces_tokens():
    user_msg = {"role": "user", "content": "q"}
    result = {
        "messages": [
            user_msg,
            {"role": "tool", "content": "r", "token_count": pack_input_tokens(8000, 6000)},
            {"role": "assistant", "content": "a", "token_count": pack_assistant_tokens(300, 40)},
        ]
    }
    out = APIServerAdapter._turn_transcript_messages([], user_msg, result)
    # tool + assistant rows are emitted (user is excluded), each with `tokens`.
    by_role = {m["role"]: m for m in out}
    assert by_role["assistant"]["tokens"] == {
        "input": 0, "output": 300, "cache_read": 0, "reasoning": 40,
    }
    assert by_role["tool"]["tokens"]["input"] == 8000
    assert by_role["tool"]["tokens"]["cache_read"] == 6000
    # Never leak the raw packed sentinel.
    assert all(m.get("token_count") is None or m["token_count"] >= 0 for m in out)
