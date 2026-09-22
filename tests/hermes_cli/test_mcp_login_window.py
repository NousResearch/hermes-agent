"""An OAuth login waits as long as ``oauth.timeout`` says, and a probe timeout names itself (#116278).

``hermes mcp login`` (and the dashboard / Desktop re-auth twins) bounded the login probe at a fixed
315 s floor, so ``oauth: {timeout: 3600}`` changed nothing; the expiry then surfaced as a bare
``asyncio.TimeoutError`` whose ``str()`` is empty — a blank ``✗ Authentication failed:`` line.
"""

import asyncio
import contextlib
import io

import pytest


def test_login_probe_window_follows_oauth_timeout(monkeypatch, tmp_path):
    """Production entry point ``_reauth_oauth_server``: with ``oauth.timeout: 3600`` the probe's
    connect bound is the callback window plus exchange headroom, not the old 315 s floor."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import hermes_cli.mcp_config as mc

    seen = {}

    def _recording_probe(name, config, connect_timeout=None, **kw):
        seen["connect_timeout"] = connect_timeout
        raise RuntimeError("probe body replaced by recorder")

    monkeypatch.setattr(mc, "_probe_single_server", _recording_probe)
    with contextlib.redirect_stdout(io.StringIO()):
        mc._reauth_oauth_server(
            "traveler",
            {"url": "https://mcp.example.test/mcp", "auth": "oauth",
             "oauth": {"timeout": 3600, "redirect_port": 27891}},
        )
    assert seen["connect_timeout"] == 3615.0


@pytest.mark.parametrize("config, names_oauth_timeout", [
    ({"url": "https://mcp.example.test/mcp", "auth": "oauth"}, True),
    ({"url": "https://mcp.example.test/mcp", "oauth": {"timeout": 3600}}, True),
    # A server with no sign-in has no oauth.timeout to wait on, and naming it makes a plain
    # connectivity failure read as an authentication one on every surface (#119232).
    ({"url": "https://mcp.example.test/mcp"}, False),
])
def test_probe_timeout_names_server_and_the_knobs_that_apply(monkeypatch, config, names_oauth_timeout):
    """A probe that outlives its bound raises a TimeoutError whose message names the server and the
    governing settings — never the empty ``str(asyncio.TimeoutError())``."""
    import hermes_cli.mcp_config as mc

    async def _hang(name, config):
        await asyncio.sleep(3600)

    monkeypatch.setattr("tools.mcp_tool_discovery._connect_server", _hang)
    with pytest.raises(TimeoutError) as info:
        mc._probe_single_server("hangsrv", dict(config), connect_timeout=0.2)
    message = str(info.value)
    assert "hangsrv" in message and "timed out" in message
    assert "connect_timeout" in message
    assert ("oauth.timeout" in message) is names_oauth_timeout
