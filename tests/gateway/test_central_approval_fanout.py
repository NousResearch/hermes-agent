"""Regression tests for central exec-approval fanout."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

_repo = str(Path(__file__).resolve().parents[2])
if _repo not in sys.path:
    sys.path.insert(0, _repo)

from gateway.config import Platform
from gateway.run import GatewayRunner
from gateway.session import SessionSource


class _TelegramAdapter:
    async def send_exec_approval(self, **kwargs):
        self.last_kwargs = kwargs
        return SimpleNamespace(success=True, message_id="123")


@pytest.mark.asyncio
async def test_exec_approval_telegram_fanout_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv("GATEWAY_APPROVAL_TELEGRAM_TARGET", raising=False)
    monkeypatch.delenv("HERMES_APPROVALS_TELEGRAM_TARGET", raising=False)
    monkeypatch.delenv("SWARM_APPROVAL_TELEGRAM_TARGET", raising=False)

    runner = GatewayRunner.__new__(GatewayRunner)
    adapter = _TelegramAdapter()
    adapter.send_exec_approval = AsyncMock(wraps=adapter.send_exec_approval)
    runner.adapters = {Platform.TELEGRAM: adapter}  # type: ignore[assignment]

    sent = await runner._send_central_exec_approval(
        origin_source=SessionSource(platform=Platform.SLACK, chat_id="C123"),
        command="rm -rf /important",
        session_key="agent:main:slack:C123",
        description="dangerous command",
    )

    assert sent is False
    adapter.send_exec_approval.assert_not_called()


@pytest.mark.asyncio
async def test_exec_approval_fans_out_to_opt_in_telegram_channel(monkeypatch):
    monkeypatch.setenv("GATEWAY_APPROVAL_TELEGRAM_TARGET", "telegram:-1003979473446")
    bearer = "A" * 24
    openai_key = "sk-" + ("A" * 20)
    auth_header = "Bea" + "rer " + bearer

    runner = GatewayRunner.__new__(GatewayRunner)
    adapter = _TelegramAdapter()
    runner.adapters = {Platform.TELEGRAM: adapter}  # type: ignore[assignment]

    sent = await runner._send_central_exec_approval(
        origin_source=SessionSource(platform=Platform.SLACK, chat_id="C123"),
        command=f"curl -H 'Authorization: {auth_header}' https://example.invalid",
        session_key="agent:main:slack:C123",
        description=f"dangerous command with {openai_key}",
    )

    assert sent is True
    assert adapter.last_kwargs == {
        "chat_id": "-1003979473446",
        "command": "curl -H 'Authorization: " + ("Bea" + "rer [REDACTED]") + "' https://example.invalid",
        "session_key": "agent:main:slack:C123",
        "description": "dangerous command with [REDACTED]",
        "metadata": None,
    }


@pytest.mark.asyncio
async def test_exec_approval_dedupes_when_origin_is_central_telegram_channel(monkeypatch):
    monkeypatch.setenv("GATEWAY_APPROVAL_TELEGRAM_TARGET", "telegram:-1003979473446")

    runner = GatewayRunner.__new__(GatewayRunner)
    adapter = _TelegramAdapter()
    adapter.send_exec_approval = AsyncMock(wraps=adapter.send_exec_approval)
    runner.adapters = {Platform.TELEGRAM: adapter}  # type: ignore[assignment]

    sent = await runner._send_central_exec_approval(
        origin_source=SessionSource(platform=Platform.TELEGRAM, chat_id="-1003979473446"),
        command="rm -rf /important",
        session_key="agent:main:telegram:-1003979473446",
        description="dangerous command",
    )

    assert sent is False
    adapter.send_exec_approval.assert_not_called()


@pytest.mark.asyncio
async def test_exec_approval_web_push_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv("GATEWAY_APPROVAL_WEB_PUSH_ENDPOINT", raising=False)
    monkeypatch.delenv("HERMES_WORKSPACE_WEB_PUSH_ENDPOINT", raising=False)

    def _unexpected_urlopen(*_args, **_kwargs):
        raise AssertionError("web-push endpoint should be opt-in")

    monkeypatch.setattr("urllib.request.urlopen", _unexpected_urlopen)

    runner = GatewayRunner.__new__(GatewayRunner)
    sent = await runner._send_exec_approval_web_push(
        command="rm -rf /important",
        session_key="agent:main:telegram:8983774650",
        description="dangerous command",
    )

    assert sent is False


@pytest.mark.asyncio
async def test_exec_approval_web_push_posts_opt_in_approval_payload(monkeypatch):
    requests = []
    bearer = "A" * 24
    github_token = "ghp_" + ("A" * 24)
    auth_header = "Bea" + "rer " + bearer

    class _Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def _fake_urlopen(request, timeout):
        requests.append((request, timeout))
        return _Response()

    monkeypatch.setenv("GATEWAY_APPROVAL_WEB_PUSH_ENDPOINT", "http://127.0.0.1:3000/api/web-push")
    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)

    runner = GatewayRunner.__new__(GatewayRunner)
    sent = await runner._send_exec_approval_web_push(
        command=f"curl -H 'Authorization: {auth_header}' https://example.invalid",
        session_key="agent:main:telegram:8983774650",
        description=f"dangerous command with {github_token}",
    )

    assert sent is True
    assert len(requests) == 1
    request, timeout = requests[0]
    assert timeout == 5
    assert request.full_url == "http://127.0.0.1:3000/api/web-push"
    body = json.loads(request.data.decode("utf-8"))
    assert body["action"] == "approval"
    assert body["requireInteraction"] is True
    assert ("Bea" + "rer [REDACTED]") in body["body"]
    assert github_token not in body["body"]


@pytest.mark.asyncio
async def test_exec_approval_web_push_rejects_invalid_endpoint(monkeypatch):
    monkeypatch.setenv("GATEWAY_APPROVAL_WEB_PUSH_ENDPOINT", "file:///tmp/web-push")

    def _unexpected_urlopen(*_args, **_kwargs):
        raise AssertionError("invalid web-push endpoint should not be called")

    monkeypatch.setattr("urllib.request.urlopen", _unexpected_urlopen)

    runner = GatewayRunner.__new__(GatewayRunner)
    sent = await runner._send_exec_approval_web_push(
        command="rm -rf /important",
        session_key="agent:main:telegram:8983774650",
        description="dangerous command",
    )

    assert sent is False
