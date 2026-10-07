"""Default character cap on tool outputs in stored /v1/responses history.

A snapshot embeds the cumulative transcript, so large tool outputs can make one
response_store.db write several hundred KB (#82513). The cap applies by default;
an explicit gateway.api_server.history_tool_output_max_chars: 0 keeps verbatim
storage and replay on the next chained turn.
"""

import copy
import json
from unittest.mock import patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.platforms.api_server_openai_routes import _cap_text, _cap_history_tool_outputs

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


def test_default_caps_tool_outputs_with_missing_config():
    with patch("hermes_cli.config.load_config", return_value={}):
        adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"port": 0}))
    assert adapter._history_tool_output_max_chars == 1000
    result = _result()
    history = adapter._build_response_conversation_history(
        PRIOR, "read it", result, "done", tool_output_max_chars=adapter._history_tool_output_max_chars)
    assert history[4]["content"].startswith("x" * 1000)
    assert history[4]["content"].endswith("...[19000 more chars]")
    assert json.loads(history[3]["tool_calls"][0]["function"]["arguments"])["content"].endswith("...[19000 more chars]")


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


@pytest.mark.parametrize("keep", [1, 1000])
def test_recapping_preserves_original_loss_count(keep):
    capped = _cap_text(BIG, keep)
    for _ in range(4):
        assert _cap_text(capped, keep) == capped
    smaller = _cap_text(capped, keep // 2)
    assert smaller == BIG[:keep // 2] + f"...[{len(BIG) - keep // 2} more chars]"


def test_short_unicode_and_boundary_text_stays_unchanged():
    for text in ["", "short", "界" * 1000]:
        assert _cap_text(text, 1000) == text
    assert _cap_text("界" * 1001, 1000) == "界" * 1000 + "...[1 more chars]"


def test_unchanged_arguments_keep_wire_format_and_zero_disables_all_caps():
    raw = '{ "values" : [' + ', '.join(['42'] * 1000) + '] }'
    history = [{"role": "assistant", "content": "", "tool_calls": [{
        "id": "c1", "function": {"name": "tool", "arguments": raw}}]}]
    assert _cap_history_tool_outputs(history, 1000) == history
    result = _result()["messages"]
    assert _cap_history_tool_outputs(result, 0) == result


def test_marker_suffix_cannot_bypass_cap():
    text = BIG + "...[123 more chars]"
    assert _cap_text(text, 1000) == "x" * 1000 + "...[19123 more chars]"
    text = BIG + "...[" + "9" * 5000 + " more chars]"
    assert len(_cap_text(text, 1000)) < 1100


@pytest.mark.parametrize("arguments", [
    {"nested": {"items": [BIG, {"text": BIG}]}, "number": 42},
    [BIG, {"text": BIG}],
    json.dumps({"nested": {"items": [BIG]}}),
    json.dumps([BIG]),
    json.dumps(BIG),
    "not-json " + BIG,
], ids=["dict", "list", "json-dict", "json-list", "json-string", "malformed"])
def test_nested_and_malformed_arguments_are_capped_without_mutation(arguments):
    history = [{"role": "assistant", "content": BIG, "tool_calls": [{
        "id": "c1", "type": "function", "function": {"name": "tool", "arguments": arguments}}]}]
    original = copy.deepcopy(history)
    capped = _cap_history_tool_outputs(history, 1000)
    args = capped[0]["tool_calls"][0]["function"]["arguments"]
    assert type(args) is type(arguments)
    assert BIG not in json.dumps(args)
    assert "more chars]" in json.dumps(args)
    assert capped[0]["content"] == BIG
    assert history == original
    assert _cap_history_tool_outputs(capped, 1000) == capped


def test_tool_content_blocks_preserve_structure_and_zero_opt_out():
    history = [{"role": "tool", "tool_call_id": "c1", "content": [
        {"type": "text", "text": BIG}, {"type": "text", "text": "small"},
        {"type": "custom", "data": {"values": [BIG, 7, None, True]}}]}]
    original = copy.deepcopy(history)
    capped = _cap_history_tool_outputs(history, 1000)
    assert capped[0]["content"][0] == {"type": "text", "text": "x" * 1000 + "...[19000 more chars]"}
    assert capped[0]["content"][1] == original[0]["content"][1]
    assert capped[0]["content"][2]["data"]["values"][1:] == [7, None, True]
    assert BIG not in json.dumps(capped)
    assert history == original
    assert _cap_history_tool_outputs(history, 0) == original


@pytest.mark.parametrize("configured, expected", [(None, 1000), (0, 0)])
def test_real_merged_config_default_and_explicit_opt_out(tmp_path, monkeypatch, configured, expected):
    from hermes_cli.config import load_config
    from hermes_cli.config_defaults import DEFAULT_CONFIG
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    if configured is not None:
        (tmp_path / "config.yaml").write_text(
            "gateway:\n  api_server:\n    history_tool_output_max_chars: 0\n", encoding="utf-8")
    assert DEFAULT_CONFIG["gateway"]["api_server"]["history_tool_output_max_chars"] == 1000
    assert load_config()["gateway"]["api_server"]["history_tool_output_max_chars"] == expected
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"port": 0}))
    assert adapter._history_tool_output_max_chars == expected
    history = adapter._build_response_conversation_history(
        PRIOR, "read it", _result(), "done", tool_output_max_chars=expected)
    assert history[4]["content"] == (BIG if expected == 0 else "x" * 1000 + "...[19000 more chars]")


@pytest.mark.asyncio
async def test_chained_http_turns_replay_and_repersist_stable_caps(tmp_path, monkeypatch):
    from gateway.platforms.api_server import ResponseStore
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"port": 0}))
    db_path = tmp_path / "response_store.db"
    adapter._response_store = ResponseStore(db_path=db_path)
    replays = []

    async def run_agent(**kwargs):
        prior = kwargs["conversation_history"]
        replays.append(copy.deepcopy(prior))
        messages = [*prior, {"role": "user", "content": kwargs["user_message"]}]
        if not prior:
            messages.extend(_result()["messages"][3:5])
        messages.append({"role": "assistant", "content": "done"})
        return {"messages": messages, "final_response": "done"}, {}

    monkeypatch.setattr(adapter, "_run_agent", run_agent)
    app = web.Application()
    app.router.add_post("/v1/responses", adapter._handle_responses)
    previous_id = None
    first_tool_rows = None
    async with TestClient(TestServer(app)) as client:
        for turn in range(4):
            body = {"input": f"turn {turn}"}
            if previous_id:
                body["previous_response_id"] = previous_id
            response = await client.post("/v1/responses", json=body)
            assert response.status == 200, await response.text()
            previous_id = (await response.json())["id"]
            # Force a SQLite read instead of relying on the store's memory cache.
            adapter._response_store = ResponseStore(db_path=db_path)
            history = adapter._response_store.get(previous_id)["conversation_history"]
            rows = [row for row in history if row.get("tool_calls") or row["role"] == "tool"]
            if first_tool_rows is None:
                first_tool_rows = copy.deepcopy(rows)
            assert rows == first_tool_rows
            assert rows[1]["content"] == "x" * 1000 + "...[19000 more chars]"
            assert json.loads(rows[0]["tool_calls"][0]["function"]["arguments"])["content"] == rows[1]["content"]
            if turn:
                assert [row for row in replays[-1] if row.get("tool_calls") or row["role"] == "tool"] == rows
