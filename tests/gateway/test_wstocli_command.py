"""Tests for /wstocli Workspace/API-to-CLI transfer command."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import Platform, PlatformConfig
from gateway.platforms.api_server import APIServerAdapter, cors_middleware, security_headers_middleware
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


def _make_gateway_event(text="/wstocli", platform=Platform.API_SERVER):
    source = SessionSource(
        platform=platform,
        user_id="workspace",
        chat_id="home",
        user_name="Workspace",
    )
    return MessageEvent(text=text, source=source)


def _make_gateway_runner(session_id="api_chat_123", title="Workspace work"):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.session_store = SimpleNamespace(
        get_or_create_session=lambda source: SimpleNamespace(
            session_id=session_id,
            session_key="agent:main:api_server:workspace:home",
        )
    )
    runner._session_db = SimpleNamespace(
        get_session_title=lambda sid: title,
        _get_session_rich_row=lambda sid: {"preview": "workspace repair flow"},
    )
    return runner


def _make_api_adapter():
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "test-key"}))
    adapter._session_db = SimpleNamespace(
        get_session_title=lambda sid: "Workspace work",
        _get_session_rich_row=lambda sid: {"preview": "workspace repair flow"},
    )
    return adapter


def _create_api_app(adapter):
    mws = [mw for mw in (cors_middleware, security_headers_middleware) if mw is not None]
    app = web.Application(middlewares=mws)
    app.router.add_post("/v1/chat/completions", adapter._handle_chat_completions)
    return app


def _wstocli_payload(message="/wstocli", stream=False):
    return {
        "model": "test-model",
        "stream": stream,
        "messages": [{"role": "user", "content": message}],
    }


@pytest.mark.asyncio
async def test_wstocli_is_registered_as_gateway_only_command():
    from hermes_cli.commands import COMMANDS, GATEWAY_KNOWN_COMMANDS, resolve_command

    wstocli = resolve_command("wstocli")

    assert wstocli is not None
    assert wstocli.name == "wstocli"
    assert wstocli.gateway_only is True
    assert "wstocli" in GATEWAY_KNOWN_COMMANDS
    assert "/wstocli" not in COMMANDS


@pytest.mark.asyncio
async def test_gateway_wstocli_rejects_non_workspace_sources():
    runner = _make_gateway_runner()

    result = await runner._handle_wstocli_command(_make_gateway_event(platform=Platform.TELEGRAM))

    assert "/wstocli" in result
    assert "Workspace" in result


@pytest.mark.asyncio
async def test_gateway_wstocli_lists_targets_for_api_server(monkeypatch):
    runner = _make_gateway_runner(session_id="api_chat_123")
    monkeypatch.setattr(
        runner,
        "_tgtocli_discover_targets",
        lambda: [
            {
                "kind": "tmux",
                "name": "desk-cli",
                "pane_id": "%7",
                "cwd": "/repo",
                "summary": "desktop Hermes CLI (`20260522_111111_facefeed`)",
            }
        ],
    )

    result = await runner._handle_wstocli_command(_make_gateway_event())

    assert "Pick CLI target for Workspace session" in result
    assert "WS: Workspace work (`api_chat_123`)" in result
    assert "1. desk-cli pane `%7`" in result
    assert "2. New tmux CLI" in result
    assert "Reply: `/wstocli 1`" in result


@pytest.mark.asyncio
async def test_gateway_wstocli_sends_resume_to_selected_tmux(monkeypatch):
    runner = _make_gateway_runner(session_id="api_chat_123")
    monkeypatch.setattr(
        runner,
        "_tgtocli_discover_targets",
        lambda: [
            {
                "kind": "tmux",
                "name": "desk-cli",
                "pane_id": "%7",
                "cwd": "/repo",
                "summary": "desktop Hermes CLI",
            }
        ],
    )

    with patch("gateway.run.subprocess.run") as run:
        run.return_value = SimpleNamespace(returncode=0, stdout="", stderr="")
        result = await runner._handle_wstocli_command(_make_gateway_event("/wstocli 1"))

    assert "DONE" in result
    assert "Sent to `desk-cli`" in result
    assert run.call_args.args[0] == [
        "tmux",
        "send-keys",
        "-t",
        "%7",
        "/resume api_chat_123",
        "Enter",
    ]


def test_api_server_wstocli_lists_targets_without_running_agent(monkeypatch):
    adapter = _make_api_adapter()
    monkeypatch.setattr(
        adapter,
        "_wstocli_discover_targets",
        lambda: [
            {
                "kind": "tmux",
                "name": "desk-cli",
                "pane_id": "%7",
                "cwd": "/repo",
                "summary": "desktop Hermes CLI",
            }
        ],
    )

    result = adapter._handle_wstocli_command("/wstocli", "api_chat_123")

    assert "Pick CLI target for Workspace session" in result
    assert "WS: Workspace work (`api_chat_123`)" in result
    assert "Reply: `/wstocli 1`" in result


def test_api_server_wstocli_sends_to_selected_tmux(monkeypatch):
    adapter = _make_api_adapter()
    monkeypatch.setattr(
        adapter,
        "_wstocli_discover_targets",
        lambda: [
            {
                "kind": "tmux",
                "name": "desk-cli",
                "pane_id": "%7",
                "cwd": "/repo",
                "summary": "desktop Hermes CLI",
            }
        ],
    )

    with patch("gateway.platforms.api_server.subprocess.run") as run:
        run.return_value = SimpleNamespace(returncode=0, stdout="", stderr="")
        result = adapter._handle_wstocli_command("/wstocli 1", "api_chat_123")

    assert "DONE" in result
    assert "Sent to `desk-cli`" in result
    assert run.call_args.args[0] == [
        "tmux",
        "send-keys",
        "-t",
        "%7",
        "/resume api_chat_123",
        "Enter",
    ]


@pytest.mark.asyncio
async def test_chat_completions_wstocli_intercepts_before_agent(monkeypatch):
    adapter = _make_api_adapter()
    monkeypatch.setattr(adapter, "_wstocli_discover_targets", lambda: [])

    with patch.object(adapter, "_run_agent") as run_agent:
        async with TestClient(TestServer(_create_api_app(adapter))) as cli:
            resp = await cli.post(
                "/v1/chat/completions",
                json=_wstocli_payload(),
                headers={"Authorization": "Bearer test-key", "X-Hermes-Session-Id": "api_chat_123"},
            )
            body = await resp.json()

    assert resp.status == 200
    assert "Pick CLI target for Workspace session" in body["choices"][0]["message"]["content"]
    assert resp.headers["X-Hermes-Session-Id"] == "api_chat_123"
    run_agent.assert_not_called()


@pytest.mark.asyncio
async def test_streaming_chat_completions_wstocli_intercepts_before_agent(monkeypatch):
    adapter = _make_api_adapter()
    monkeypatch.setattr(adapter, "_wstocli_discover_targets", lambda: [])

    with patch.object(adapter, "_run_agent") as run_agent:
        async with TestClient(TestServer(_create_api_app(adapter))) as cli:
            resp = await cli.post(
                "/v1/chat/completions",
                json=_wstocli_payload(stream=True),
                headers={"Authorization": "Bearer test-key", "X-Hermes-Session-Id": "api_chat_123"},
            )
            body = await resp.text()

    assert resp.status == 200
    assert "Pick CLI target for Workspace session" in body
    assert "data: [DONE]" in body
    assert resp.headers["X-Hermes-Session-Id"] == "api_chat_123"
    run_agent.assert_not_called()
