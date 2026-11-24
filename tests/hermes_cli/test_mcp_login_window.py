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


def test_probe_timeout_names_server_and_knobs(monkeypatch):
    """A probe that outlives its bound raises a TimeoutError whose message names the server and the
    governing settings — never the empty ``str(asyncio.TimeoutError())``."""
    import hermes_cli.mcp_config as mc

    async def _hang(name, config):
        await asyncio.sleep(3600)

    monkeypatch.setattr("tools.mcp_tool_discovery._connect_server", _hang)
    with pytest.raises(TimeoutError) as info:
        mc._probe_single_server("hangsrv", {"url": "https://mcp.example.test/mcp"}, connect_timeout=0.2)
    message = str(info.value)
    assert "hangsrv" in message and "timed out" in message
    assert "connect_timeout" in message and "oauth.timeout" in message


def _record_add_probe(monkeypatch, tmp_path, *, oauth: bool):
    """Run ``cmd_mcp_add`` for an HTTP server with the probe replaced by a recorder; returns its bound."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import argparse
    import hermes_cli.mcp_config as mc

    def _fake_http_auth(name, url, auth_type, server_config):
        if oauth:
            server_config["auth"] = "oauth"
        return True

    seen = {}

    def _recording_probe(name, config, connect_timeout=None, **kw):
        seen["connect_timeout"] = connect_timeout
        raise RuntimeError("probe body replaced by recorder")

    monkeypatch.setattr(mc, "_configure_http_auth", _fake_http_auth)
    monkeypatch.setattr(mc, "_probe_single_server", _recording_probe)
    monkeypatch.setattr("builtins.input", lambda _: "n")
    args = argparse.Namespace(name="traveler", url="https://mcp.example.test/mcp", mcp_command=None, args=None,
                              auth="oauth" if oauth else None, connect_timeout=None, env=None, preset=None)
    with contextlib.redirect_stdout(io.StringIO()):
        mc.cmd_mcp_add(args)
    return seen["connect_timeout"]


def test_add_with_oauth_waits_like_login(monkeypatch, tmp_path):
    """``hermes mcp add --auth oauth``: the first connect is the browser sign-in, so its probe gets the login
    window (oauth.timeout 300 s + 15 s), not the 30 s connect_timeout that closed the callback listener while
    the user was still on the consent page."""
    assert _record_add_probe(monkeypatch, tmp_path, oauth=True) == 315.0


def test_add_without_oauth_keeps_the_connect_timeout(monkeypatch, tmp_path):
    """A server without OAuth keeps the plain connect bound (None -> config connect_timeout, 30 s default)."""
    assert _record_add_probe(monkeypatch, tmp_path, oauth=False) is None
