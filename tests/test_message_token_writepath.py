"""Write-path tests for bit-packed message token accounting.

Covers the two producer-side touch points and the API-display decode:
  * build_assistant_message packs (output, reasoning) onto the assistant row
    from agent._last_usage (and skips cleanly when usage is absent).
  * APIServerAdapter._message_response never leaks a raw packed (negative) value.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock

from hermes_token_codec import unpack_token_count, pack_assistant_tokens, pack_input_tokens


def _make_agent():
    from run_agent import AIAgent
    agent = MagicMock(spec=AIAgent)
    agent._build_assistant_message = AIAgent._build_assistant_message.__get__(agent)
    agent._extract_reasoning = AIAgent._extract_reasoning.__get__(agent)
    agent.verbose_logging = False
    agent.reasoning_callback = None
    agent.stream_delta_callback = None
    agent._stream_callback = None
    return agent


def test_build_assistant_message_packs_usage():
    agent = _make_agent()
    agent._last_usage = SimpleNamespace(output_tokens=1500, reasoning_tokens=300)
    msg = agent._build_assistant_message(SimpleNamespace(content="hi", tool_calls=None), "stop")
    assert msg["token_count"] < 0  # packed -> negative
    assert unpack_token_count(msg["token_count"]) == {
        "output_tokens": 1500, "reasoning_tokens": 300,
    }


def test_build_assistant_message_without_usage_has_no_token_count():
    agent = _make_agent()
    # spec=AIAgent MagicMock would auto-create _last_usage as a Mock; force absent.
    agent._last_usage = None
    msg = agent._build_assistant_message(SimpleNamespace(content="hi", tool_calls=None), "stop")
    assert "token_count" not in msg


def test_message_response_decodes_packed_assistant():
    from gateway.platforms.api_server import APIServerAdapter
    row = {
        "id": 1, "role": "assistant", "content": "hi",
        "token_count": pack_assistant_tokens(900, 120),
    }
    resp = APIServerAdapter._message_response(row)
    # Raw negative sentinel must not leak; scalar stays legacy-compatible.
    assert resp["token_count"] == 900
    assert resp["tokens"] == {"input": 0, "output": 900, "cache_read": 0, "reasoning": 120}


def test_message_response_decodes_packed_input_row():
    from gateway.platforms.api_server import APIServerAdapter
    row = {
        "id": 2, "role": "user", "content": "q",
        "token_count": pack_input_tokens(8000, 6000),
    }
    resp = APIServerAdapter._message_response(row)
    # Non-assistant scalar token_count is suppressed; buckets carry the data.
    assert resp["token_count"] is None
    assert resp["tokens"]["input"] == 8000
    assert resp["tokens"]["cache_read"] == 6000


def test_message_response_legacy_and_null():
    from gateway.platforms.api_server import APIServerAdapter
    legacy = APIServerAdapter._message_response({"id": 3, "role": "assistant", "token_count": 1234})
    assert legacy["token_count"] == 1234
    assert legacy["tokens"]["output"] == 1234

    none_row = APIServerAdapter._message_response({"id": 4, "role": "user", "token_count": None})
    assert none_row["token_count"] is None
    assert none_row["tokens"] == {"input": 0, "output": 0, "cache_read": 0, "reasoning": 0}
