"""Serialized reasoning parameters are observable without logging request content."""

import asyncio
import json

import httpx
import pytest

from agent.outbound_reasoning_audit import EVENT_PREFIX
from agent.process_bootstrap import build_keepalive_http_client


OFFICIAL = "https://chatgpt.com/backend-api/codex"


def _event(capsys):
    captured = capsys.readouterr().err.strip()
    assert captured.startswith(EVENT_PREFIX)
    return json.loads(captured[len(EVENT_PREFIX):])


def test_sync_hook_reports_exact_serialized_value_without_sensitive_request_data(capsys):
    client = build_keepalive_http_client("https://api.example.test/v1")
    try:
        request = client.build_request(
            "POST",
            "https://api.example.test/v1/responses?private=query-value",
            json={"model": "gpt-example", "input": "private prompt", "reasoning": {"effort": "high"}},
            headers={"Authorization": "Bearer private-token"},
        )
        for hook in client.event_hooks["request"]:
            hook(request)
    finally:
        client.close()

    event = _event(capsys)
    assert event["model"] == "gpt-example"
    assert event["host"] == "api.example.test"
    assert event["path"] == "/v1/responses"
    assert event["reasoning_fields"] == [{"path": "reasoning.effort", "value": "high"}]
    assert event["reasoning_status"] == "explicit"
    assert event["stage"] == "before_transport_send"
    assert "private prompt" not in json.dumps(event)
    assert "private-token" not in json.dumps(event)
    assert "query-value" not in json.dumps(event)


def test_absent_reasoning_is_distinguished_from_unavailable(capsys):
    client = build_keepalive_http_client("https://api.example.test/v1")
    try:
        request = client.build_request("POST", "https://api.example.test/v1/chat", json={"model": "x"})
        for hook in client.event_hooks["request"]:
            hook(request)
    finally:
        client.close()
    event = _event(capsys)
    assert event["reasoning_fields"] == []
    assert event["reasoning_status"] == "not_present"


def test_unparseable_post_is_logged_as_unavailable(capsys):
    from agent.outbound_reasoning_audit import audit_outbound_reasoning_request

    request = httpx.Request("POST", "https://api.example.test/v1/responses", content=b"not-json")
    audit_outbound_reasoning_request(request)
    event = _event(capsys)
    assert event["model"] is None
    assert event["reasoning_fields"] == []
    assert event["reasoning_status"] == "unavailable"


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"reasoning_effort": "medium"}, ("reasoning_effort", "medium")),
        ({"output_config": {"effort": "high"}}, ("output_config.effort", "high")),
        (
            {"thinking": {"type": "enabled", "budget_tokens": 4096}},
            (("thinking.type", "enabled"), ("thinking.budget_tokens", 4096)),
        ),
        ({"thinkingConfig": {"thinkingLevel": "HIGH"}}, ("thinkingConfig.thinkingLevel", "HIGH")),
        ({"thinking_level": "high"}, ("thinking_level", "high")),
    ],
)
def test_supported_reasoning_fields_are_captured(payload, expected, capsys):
    from agent.outbound_reasoning_audit import audit_outbound_reasoning_request

    request = httpx.Request("POST", "https://api.example.test/v1/responses", json=payload)
    audit_outbound_reasoning_request(request)
    event = _event(capsys)
    expected_fields = expected if isinstance(expected[0], tuple) else (expected,)
    assert event["reasoning_fields"] == [
        {"path": path, "value": value} for path, value in expected_fields
    ]
    assert event["reasoning_status"] == "explicit"


def test_audit_hook_precedes_codex_compression_hook(monkeypatch, capsys):
    from agent import codex_headers

    order = []

    def compress(request):
        order.append("compress")

    monkeypatch.setattr(
        codex_headers,
        "codex_request_compression_hooks",
        lambda _url, async_mode=False: {"request": [compress]},
    )
    client = build_keepalive_http_client(OFFICIAL)
    try:
        request = client.build_request(
            "POST", f"{OFFICIAL}/responses", json={"model": "gpt-codex", "reasoning": {"effort": "high"}}
        )
        hooks = client.event_hooks["request"]
        assert len(hooks) == 2
        hooks[0](request)
        order.append("audit")
        hooks[1](request)
    finally:
        client.close()

    assert order == ["audit", "compress"]
    assert _event(capsys)["reasoning_fields"] == [{"path": "reasoning.effort", "value": "high"}]


def test_async_client_uses_async_audit_hook_before_existing_hooks(monkeypatch, capsys):
    from agent import codex_headers

    order = []

    async def existing(request):
        order.append("existing")

    monkeypatch.setattr(
        codex_headers,
        "codex_request_compression_hooks",
        lambda _url, async_mode=False: {"request": [existing]},
    )
    client = build_keepalive_http_client(OFFICIAL, async_mode=True)

    async def run_hooks():
        request = httpx.Request(
            "POST", f"{OFFICIAL}/responses", json={"model": "gpt-codex", "reasoning": {"effort": "high"}}
        )
        hooks = client.event_hooks["request"]
        assert len(hooks) == 2
        await hooks[0](request)
        order.append("audit")
        await hooks[1](request)

    try:
        asyncio.run(run_hooks())
    finally:
        asyncio.run(client.aclose())

    assert order == ["audit", "existing"]
    assert _event(capsys)["reasoning_fields"] == [{"path": "reasoning.effort", "value": "high"}]
