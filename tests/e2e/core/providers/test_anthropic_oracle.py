"""The SDK-oracle fake itself: it must be able to fail.

Every wire-conformance cell in this directory ends with ``schema_errors() == []``;
that assertion is only worth something if the oracle reports what the real API
rejects. The real Messages API answers 400 ``Extra inputs are not permitted`` for
an unknown key on ANY content block, including replayed assistant blocks — which
the SDK's request union would otherwise accept through its permissive response
models. Record mode is exercised against a second local fake (never a real key).
"""

from __future__ import annotations

import copy
import json
import urllib.request

import pytest

from tests.fakes.providers.anthropic_messages import AnthropicMessagesServer, Reply, Text, validate_request

SIGNED = {"type": "thinking", "thinking": "weigh it", "signature": "EqSig+/=="}
TOOL_USE = {"type": "tool_use", "id": "toolu_1", "name": "read_file", "input": {"path": "a.txt"}}


def _body() -> dict:
    return {
        "model": "claude-sonnet-4-5", "max_tokens": 64,
        "system": [{"type": "text", "text": "you are terse", "cache_control": {"type": "ephemeral"}}],
        "tools": [{"name": "read_file", "description": "read", "input_schema": {"type": "object"}}],
        "messages": [
            {"role": "user", "content": "read a.txt"},
            {"role": "assistant", "content": [copy.deepcopy(SIGNED), {"type": "text", "text": "reading"},
                                              copy.deepcopy(TOOL_USE)]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1",
                                          "content": [{"type": "text", "text": "A"}]}]},
        ],
    }


@pytest.mark.parametrize("beta", [False, True], ids=["ga", "beta"])
def test_conformant_body_has_no_schema_errors(beta: bool) -> None:
    assert validate_request(_body(), beta=beta) == []


@pytest.mark.parametrize("beta", [False, True], ids=["ga", "beta"])
@pytest.mark.parametrize("where", [
    ("messages", 1, 0),  # assistant thinking (what _replay_thinking builds)
    ("messages", 1, 1),  # assistant text
    ("messages", 1, 2),  # assistant tool_use
    ("messages", 2, 0),  # user tool_result
    ("system", None, 0),  # system text block
], ids=["assistant_thinking", "assistant_text", "assistant_tool_use", "user_tool_result", "system_text"])
def test_unknown_key_in_any_content_block_is_reported(where: tuple, beta: bool) -> None:
    body = _body()
    field, msg, idx = where
    block = body["system"][idx] if field == "system" else body["messages"][msg]["content"][idx]
    block["e2e_bogus_field"] = 1
    path = f"$.system[{idx}]" if field == "system" else f"$.messages[{msg}].content[{idx}]"
    assert f"unknown key {path}.e2e_bogus_field" in validate_request(body, beta=beta)


@pytest.mark.parametrize("beta", [False, True], ids=["ga", "beta"])
def test_response_only_block_type_in_request_is_reported(beta: bool) -> None:
    body = _body()
    body["messages"][1]["content"][0]["type"] = "thinking_delta"
    assert any("$.messages[1].content[0]" in p for p in validate_request(body, beta=beta))


def test_record_mode_writes_a_sanitised_cassette(tmp_path) -> None:
    secret_key, secret_bearer = "sk-ant-api03-RECORD-PROBE-KEY", "sk-ant-oat01-RECORD-PROBE-BEARER"
    with AnthropicMessagesServer([Reply([Text("UPSTREAM-ANSWER")])]) as upstream, \
            AnthropicMessagesServer(record_upstream=upstream.base_url, cassette="probe",
                                    cassette_dir=tmp_path) as recorder:
        req = urllib.request.Request(
            recorder.base_url + "/v1/messages", data=json.dumps(_body()).encode(), method="POST",
            headers={"content-type": "application/json", "x-api-key": secret_key,
                     "authorization": f"Bearer {secret_bearer}", "anthropic-version": "2023-06-01"})
        with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 - loopback fake
            answer = json.loads(resp.read())
        assert upstream.requests[0]["x_api_key"] == secret_key, "record mode must forward the request's auth"
    assert answer["content"][0]["text"] == "UPSTREAM-ANSWER"
    lines = (tmp_path / "probe.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    exchange = json.loads(lines[0])
    assert exchange["request"]["body"] == _body() and exchange["response"]["status"] == 200
    assert "UPSTREAM-ANSWER" in exchange["response"]["body"]
    raw = lines[0].lower()
    assert secret_key.lower() not in raw and secret_bearer.lower() not in raw
    assert not {"authorization", "x-api-key"} & set(exchange["request"]["headers"])
