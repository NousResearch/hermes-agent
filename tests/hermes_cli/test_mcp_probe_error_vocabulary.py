"""A probe failure must describe itself in the vocabulary of what actually failed (#119232).

The Desktop MCP card decides between "Needs authentication" (amber, with an Authenticate button)
and a plain connectivity error from the probe's error text
(``apps/desktop/src/lib/mcp-probe-cache.ts``). Hermes' own connect-timeout hint mentioned an OAuth
login, and the dashboard's failed-sign-in message named an OAuth token, so both read as
"this server wants a login": an auth-free server that merely timed out was pinned to an auth state
it could never leave, while a server that really refuses with 401 produced the same timeout text
and was indistinguishable from an unreachable host.
"""

import re

import pytest

# apps/desktop/src/lib/mcp-probe-cache.ts:10 — kept in sync deliberately: these tests pin the
# backend's side of that contract (the text it emits), not the classifier itself.
NEEDS_AUTH_RE = re.compile(r"\b(401|unauthorized|forbidden|invalid[_ ]?token|authentication|oauth)\b", re.I)


def _reads_as_needs_auth(text: str) -> bool:
    return NEEDS_AUTH_RE.search(text) is not None


def test_connect_timeout_does_not_read_as_an_auth_failure(monkeypatch):
    """The real probe path: a server that never answers is a connectivity error."""
    import asyncio

    from hermes_cli.mcp_config import _probe_single_server

    async def _hang(name, config):
        await asyncio.sleep(30)

    monkeypatch.setattr("tools.mcp_tool_discovery._connect_server", _hang)

    with pytest.raises(TimeoutError) as info:
        _probe_single_server("inspo", {"url": "https://inspomcp.dev/api/mcp"}, connect_timeout=1)

    message = str(info.value)
    assert "timed out after 1s" in message  # still says what happened, and what bounds it
    assert "connect_timeout" in message
    assert not _reads_as_needs_auth(message)


def test_failed_sign_in_message_does_not_re_pin_the_card():
    """The dashboard writes this back onto the probe result, so it must not read as needs-auth."""
    from hermes_cli.mcp_config import NO_TOKEN_STORED_MESSAGE

    assert "no access token was stored" in NO_TOKEN_STORED_MESSAGE
    assert "config.yaml" in NO_TOKEN_STORED_MESSAGE  # still names the fix
    assert not _reads_as_needs_auth(NO_TOKEN_STORED_MESSAGE)


def test_both_sign_in_paths_share_that_message():
    """The dashboard flow and the connectors flow both reach the Desktop card."""
    import inspect

    from hermes_cli import web_server_mcp
    from tools.connectors import mcp_oauth

    for module in (web_server_mcp, mcp_oauth):
        source = inspect.getsource(module)
        assert "NO_TOKEN_STORED_MESSAGE" in source
        assert "no OAuth token was obtained" not in source


@pytest.mark.asyncio
async def test_a_real_401_does_read_as_an_auth_failure(monkeypatch):
    """The other half: the endpoint that genuinely refuses must keep the Authenticate affordance."""
    from tools.mcp_tool import MCPServerTask
    from tools.mcp_tool_errors import McpAuthRequiredError

    class _SdkInternalError(Exception):
        def __init__(self):
            super().__init__("Server returned an error response")
            self.error = type("E", (), {"code": -32603})()

    task = MCPServerTask("inspo")
    task._config = {}

    async def fake_serve(self, cm, label, timeout):
        self._http_rejection.update(status=401, method="POST", url="https://inspomcp.dev/api/mcp",
                                    body='{"error":"unauthorized"}')
        raise ExceptionGroup("g", [_SdkInternalError()])

    monkeypatch.setattr(MCPServerTask, "_serve_transport", fake_serve)
    monkeypatch.setattr(MCPServerTask, "_streamable_http_transport", lambda self, *a, **k: object())
    monkeypatch.setattr(MCPServerTask, "_sse_transport", lambda self, *a, **k: object())
    monkeypatch.setattr(MCPServerTask, "_build_oauth_auth", lambda self, *a: None)

    with pytest.raises(McpAuthRequiredError) as info:
        await task._run_http({"url": "https://inspomcp.dev/api/mcp", "connect_timeout": 1})

    assert _reads_as_needs_auth(str(info.value))
