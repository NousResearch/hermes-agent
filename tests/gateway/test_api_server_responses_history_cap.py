"""Opt-in byte cap on tool outputs in the stored /v1/responses conversation history.

The persisted snapshot embeds the cumulative transcript with every tool output verbatim, so a
few large tool outputs made one response_store.db write ~677 KB (#82513). The cap is opt-in
(``gateway.api_server.history_tool_output_max_chars``, 0 = verbatim) because the stored
history is what the model is replayed on the next chained turn.
"""

import json
from unittest.mock import patch

from gateway.platforms.api_server import APIServerAdapter
from gateway.platforms.base import PlatformConfig

BIG = "x" * 20_000
PRIOR = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]


def _result():
    return {"messages": [
        *PRIOR,
        {"role": "user", "content": "read it", "timestamp": 1.0},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "write_file", "arguments": json.dumps({"path": "/tmp/f", "content": BIG})}}]},
        {"role": "tool", "tool_call_id": "c1", "name": "write_file", "content": BIG},
        {"role": "assistant", "content": "done " + BIG},
    ]}


def test_default_stores_tool_outputs_verbatim():
    with patch("hermes_cli.config.load_config", return_value={}):
        adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"port": 0}))
    assert adapter._history_tool_output_max_chars == 0
    result = _result()
    history = adapter._build_response_conversation_history(
        PRIOR, "read it", result, "done", tool_output_max_chars=adapter._history_tool_output_max_chars)
    assert history[4]["content"] == BIG
    assert json.loads(history[3]["tool_calls"][0]["function"]["arguments"])["content"] == BIG


def test_cap_trims_only_tool_rows_and_leaves_agent_transcript_intact():
    cfg = {"gateway": {"api_server": {"history_tool_output_max_chars": 1000}}}
    with patch("hermes_cli.config.load_config", return_value=cfg):
        adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"port": 0}))
    assert adapter._history_tool_output_max_chars == 1000
    result = _result()
    history = adapter._build_response_conversation_history(
        PRIOR, "read it", result, "done", tool_output_max_chars=adapter._history_tool_output_max_chars)
    tool_row = history[4]
    assert tool_row["content"].startswith("x" * 1000) and tool_row["content"].endswith("...[19000 more chars]")
    assert len(tool_row["content"]) < 1100
    args = json.loads(history[3]["tool_calls"][0]["function"]["arguments"])
    assert args["path"] == "/tmp/f" and args["content"].endswith("...[19000 more chars]")
    # Non-tool rows are untouched, and the agent's own transcript rows were copied, not mutated.
    assert history[5]["content"] == "done " + BIG
    assert result["messages"][4]["content"] == BIG
    assert [m["role"] for m in history] == ["user", "assistant", "user", "assistant", "tool", "assistant"]
