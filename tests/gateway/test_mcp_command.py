"""``/mcp``: connect an OAuth MCP server from a chat with the device flow (#78169).

The sign-in link and code go to the DM that asked; the server is saved and reloaded only after
a token was granted, so a failed, denied or abandoned attempt leaves config.yaml untouched.
"""
import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway import slash_commands_mcp as mcp_cmd
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from hermes_cli.commands import resolve_command


def _source(*, chat_type="dm", chat_id="chat-1", user_id="user-1", platform=Platform.TELEGRAM):
    return SessionSource(platform=platform, chat_id=chat_id, chat_type=chat_type, user_id=user_id)


def _event(args="", **source_kwargs):
    return SimpleNamespace(source=_source(**source_kwargs), get_command_args=lambda: args)


def _runner(monkeypatch, *, extra=None, servers=None, login=None, saved=None):
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(platforms={
        Platform.TELEGRAM: PlatformConfig(enabled=True, token="token", extra=extra or {})})
    runner.adapters = {}
    runner._agent_cache = {}
    runner._agent_cache_lock = threading.Lock()
    runner._deliver_platform_notice = AsyncMock()
    runner._execute_mcp_reload = AsyncMock(return_value="MCP Servers Reloaded")
    monkeypatch.setattr(mcp_cmd, "_configured_servers", lambda: dict(servers or {}))
    saved = saved if saved is not None else {}

    def save(name, cfg):
        saved[name] = cfg
        return True

    from hermes_cli import mcp_config
    monkeypatch.setattr(mcp_config, "_save_mcp_server", save)
    import tools.mcp_oauth_device as device
    monkeypatch.setattr(device, "login_device", login or _granting())
    return runner, saved


def _granting(link="https://as.example/device?user_code=ABCD-EFGH", code="ABCD-EFGH"):
    calls = []

    async def login_device(name, url, oauth, on_code=None):
        calls.append((name, url, oauth))
        await on_code(link, code, 10)

    login_device.calls = calls
    return login_device


async def _finish(runner):
    tasks = list(getattr(runner, "_background_tasks", ()))
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


def _pushed(runner):
    return [c.args[1] for c in runner._deliver_platform_notice.await_args_list]


@pytest.mark.asyncio
@pytest.mark.parametrize("chat_type", ["group", "channel", "thread"])
async def test_only_a_direct_message_can_start_a_sign_in(monkeypatch, chat_type):
    login = _granting()
    runner, saved = _runner(monkeypatch, login=login)
    reply = await runner._handle_mcp_command(_event("add notes https://mcp.example/mcp", chat_type=chat_type))
    await _finish(runner)
    assert reply == mcp_cmd.MCP_DM_ONLY
    assert not login.calls and not saved


@pytest.mark.asyncio
async def test_a_non_admin_is_refused_when_gating_is_on(monkeypatch):
    runner, _ = _runner(monkeypatch, extra={"allow_admin_from": ["operator"]})
    reply = await runner._handle_mcp_command(_event("add notes https://mcp.example/mcp", user_id="other"))
    assert reply == mcp_cmd.MCP_NOT_ALLOWED


@pytest.mark.asyncio
@pytest.mark.parametrize("args, expected", [
    ("", mcp_cmd.MCP_USAGE),
    ("delete notes", mcp_cmd.MCP_USAGE),
    ("remove", mcp_cmd.MCP_USAGE),
    ("add notes", mcp_cmd.MCP_USAGE),
    ("add bad/name https://mcp.example/mcp", mcp_cmd.MCP_BAD_NAME),
    ("add notes http://mcp.example/mcp", mcp_cmd.MCP_BAD_URL),
    ("add notes mcp.example", mcp_cmd.MCP_BAD_URL),
])
async def test_bad_arguments_get_a_reply_and_start_nothing(monkeypatch, args, expected):
    login = _granting()
    runner, _ = _runner(monkeypatch, login=login)
    assert await runner._handle_mcp_command(_event(args)) == expected
    await _finish(runner)
    assert not login.calls


@pytest.mark.asyncio
async def test_add_sends_the_link_then_saves_reloads_and_says_connected(monkeypatch):
    login = _granting()
    runner, saved = _runner(monkeypatch, login=login)

    reply = await runner._handle_mcp_command(_event("add notes https://mcp.example/mcp"))
    await _finish(runner)

    assert reply == mcp_cmd.MCP_STARTING.format(name="notes")
    assert login.calls == [("notes", "https://mcp.example/mcp", {"flow": "device"})]
    pushed = _pushed(runner)
    assert "https://as.example/device?user_code=ABCD-EFGH" in pushed[0] and "ABCD-EFGH" in pushed[0]
    assert pushed[1].startswith("Connected notes.") and "MCP Servers Reloaded" in pushed[1]
    assert saved == {"notes": {"url": "https://mcp.example/mcp", "auth": "oauth",
                               "oauth": {"flow": "device"}, "enabled": True}}
    runner._execute_mcp_reload.assert_awaited_once()
    # Every push goes to the DM that asked.
    assert all(c.args[0].chat_id == "chat-1" for c in runner._deliver_platform_notice.await_args_list)


@pytest.mark.asyncio
async def test_a_denied_or_expired_sign_in_saves_nothing_and_says_why(monkeypatch):
    async def denied(name, url, oauth, on_code=None):
        await on_code("https://as.example/device", "ABCD-EFGH", 10)
        raise RuntimeError("Device authorization failed: access_denied")

    runner, saved = _runner(monkeypatch, login=denied)
    await runner._handle_mcp_command(_event("add notes https://mcp.example/mcp"))
    await _finish(runner)

    assert not saved
    runner._execute_mcp_reload.assert_not_awaited()
    assert _pushed(runner)[-1] == mcp_cmd.MCP_FAILED.format(
        name="notes", error="Device authorization failed: access_denied")


@pytest.mark.asyncio
async def test_an_unexpected_error_is_not_echoed_to_the_chat(monkeypatch):
    async def broken(name, url, oauth, on_code=None):
        raise ValueError("secret-looking detail")

    runner, saved = _runner(monkeypatch, login=broken)
    await runner._handle_mcp_command(_event("add notes https://mcp.example/mcp"))
    await _finish(runner)
    assert not saved
    assert "secret-looking" not in _pushed(runner)[-1]


@pytest.mark.asyncio
async def test_a_rejected_configuration_is_reported_and_not_reloaded(monkeypatch):
    runner, _ = _runner(monkeypatch)
    from hermes_cli import mcp_config
    monkeypatch.setattr(mcp_config, "_save_mcp_server", lambda name, cfg: False)
    await runner._handle_mcp_command(_event("add notes https://mcp.example/mcp"))
    await _finish(runner)
    runner._execute_mcp_reload.assert_not_awaited()
    assert _pushed(runner)[-1] == mcp_cmd.MCP_REJECTED.format(name="notes")


@pytest.mark.asyncio
async def test_a_configured_name_is_not_overwritten(monkeypatch):
    login = _granting()
    runner, _ = _runner(monkeypatch, login=login, servers={"notes": {"url": "https://x.example/mcp"}})
    for args in ("add notes https://mcp.example/mcp", "install notes"):
        assert "already configured" in await runner._handle_mcp_command(_event(args))
    assert not login.calls


@pytest.mark.asyncio
async def test_login_signs_in_again_and_keeps_the_configured_block(monkeypatch):
    login = _granting()
    existing = {"url": "https://mcp.example/mcp", "auth": "oauth", "oauth": {"scope": "read"}}
    runner, saved = _runner(monkeypatch, login=login, servers={"notes": existing})
    await runner._handle_mcp_command(_event("login notes"))
    await _finish(runner)
    assert login.calls == [("notes", "https://mcp.example/mcp", {"scope": "read", "flow": "device"})]
    assert not saved
    runner._execute_mcp_reload.assert_awaited_once()


@pytest.mark.asyncio
async def test_login_refuses_unknown_and_non_oauth_servers(monkeypatch):
    runner, _ = _runner(monkeypatch, servers={"local": {"command": "npx", "args": ["x"]}})
    assert "not configured" in await runner._handle_mcp_command(_event("login missing"))
    assert "not a remote server" in await runner._handle_mcp_command(_event("login local"))


@pytest.mark.asyncio
async def test_install_uses_the_catalog_entry_with_the_device_flow(monkeypatch):
    login = _granting()
    runner, saved = _runner(monkeypatch, login=login)
    await runner._handle_mcp_command(_event("install knowsme"))
    await _finish(runner)
    assert login.calls[0][:2] == ("knowsme", "https://mcp.knowsme.dev/mcp")
    assert saved["knowsme"]["url"] == "https://mcp.knowsme.dev/mcp"
    assert saved["knowsme"]["auth"] == "oauth" and saved["knowsme"]["oauth"]["flow"] == "device"


@pytest.mark.asyncio
@pytest.mark.parametrize("transport, auth, expected", [
    ("stdio", SimpleNamespace(type="none", provider=None, env=[]), "not a remote server"),
    ("http", SimpleNamespace(type="api_key", provider=None, env=[]), "not a remote server"),
    ("http", SimpleNamespace(type="oauth", provider="google", env=[]), "hermes mcp install"),
    ("http", SimpleNamespace(type="oauth", provider=None, env=["CLIENT_ID"]), "hermes mcp install"),
])
async def test_install_refuses_unknown_and_terminal_only_entries(monkeypatch, transport, auth, expected):
    from hermes_cli import mcp_catalog
    login = _granting()
    runner, _ = _runner(monkeypatch, login=login)
    assert "No MCP catalog entry" in await runner._handle_mcp_command(_event("install no-such-server"))
    entry = SimpleNamespace(name="vendor", transport=SimpleNamespace(type=transport, url="https://v.example/mcp"),
                            auth=auth, install=None)
    monkeypatch.setattr(mcp_catalog, "get_entry", lambda name: entry if name == "vendor" else None)
    assert expected in await runner._handle_mcp_command(_event("install vendor"))
    assert not login.calls


@pytest.mark.asyncio
async def test_another_identity_cannot_supersede_a_live_attempt(monkeypatch):
    gate = asyncio.Event()

    async def waiting(name, url, oauth, on_code=None):
        await gate.wait()

    runner, _ = _runner(monkeypatch, login=waiting)
    await runner._handle_mcp_command(_event("add notes https://mcp.example/mcp"))
    reply = await runner._handle_mcp_command(_event("add other https://mcp.example/mcp", user_id="user-2"))
    assert reply == mcp_cmd.MCP_BUSY_ELSEWHERE
    gate.set()
    await _finish(runner)


@pytest.mark.asyncio
async def test_the_same_identity_asking_again_replaces_its_attempt(monkeypatch):
    gate = asyncio.Event()
    started = []

    async def waiting(name, url, oauth, on_code=None):
        started.append(name)
        await gate.wait()

    runner, _ = _runner(monkeypatch, login=waiting)
    await runner._handle_mcp_command(_event("add notes https://mcp.example/mcp"))
    await asyncio.sleep(0)
    first = runner._mcp_attempts["mcp"][1]
    await runner._handle_mcp_command(_event("add notes2 https://mcp.example/mcp"))
    await asyncio.sleep(0)
    assert first.cancelled() or first.done()
    gate.set()
    await _finish(runner)
    assert started == ["notes", "notes2"]


@pytest.mark.asyncio
async def test_mcp_dispatches_mid_turn():
    runner = object.__new__(GatewayRunner)
    runner._handle_mcp_command = AsyncMock(return_value="started")
    command = resolve_command("mcp")
    event = _event("install knowsme")
    assert await runner._dispatch_busy_slash_command(event, command, "", event.source) == "started"
    runner._handle_mcp_command.assert_awaited_once_with(event)


@pytest.mark.asyncio
async def test_remove_drops_the_server_and_its_tokens_then_reloads(monkeypatch):
    removed, forgot = [], []
    runner, _ = _runner(monkeypatch, servers={"notes": {"url": "https://mcp.example/mcp", "auth": "oauth"}})
    from hermes_cli import mcp_config
    monkeypatch.setattr(mcp_config, "_remove_mcp_server", lambda name: removed.append(name) or True)
    import tools.mcp_oauth_manager as manager
    monkeypatch.setattr(manager, "get_manager", lambda: SimpleNamespace(remove=forgot.append))

    reply = await runner._handle_mcp_command(_event("remove notes"))

    assert reply.startswith(mcp_cmd.MCP_REMOVED.format(name="notes")) and "MCP Servers Reloaded" in reply
    assert removed == ["notes"] and forgot == ["notes"]
    runner._execute_mcp_reload.assert_awaited_once()


@pytest.mark.asyncio
async def test_remove_refuses_an_unknown_server_and_outside_a_dm(monkeypatch):
    runner, _ = _runner(monkeypatch)
    from hermes_cli import mcp_config
    monkeypatch.setattr(mcp_config, "_remove_mcp_server", lambda name: pytest.fail("removed"))
    assert "nothing to remove" in await runner._handle_mcp_command(_event("remove notes"))
    assert await runner._handle_mcp_command(_event("remove notes", chat_type="group")) == mcp_cmd.MCP_DM_ONLY
    runner._execute_mcp_reload.assert_not_awaited()


@pytest.mark.asyncio
async def test_remove_cancels_a_sign_in_still_waiting_for_that_server(monkeypatch):
    gate = asyncio.Event()

    async def waiting(name, url, oauth, on_code=None):
        await gate.wait()

    runner, saved = _runner(monkeypatch, login=waiting)
    from hermes_cli import mcp_config
    monkeypatch.setattr(mcp_config, "_remove_mcp_server", lambda name: True)
    await runner._handle_mcp_command(_event("add notes https://mcp.example/mcp"))
    await asyncio.sleep(0)
    pending = runner._mcp_attempts["mcp"][1]
    monkeypatch.setattr(mcp_cmd, "_configured_servers", lambda: {"notes": {"url": "https://mcp.example/mcp"}})
    await runner._handle_mcp_command(_event("remove notes"))
    await asyncio.sleep(0)
    assert pending.cancelled()
    gate.set()
    await _finish(runner)
    assert not saved
