"""``gateway.home: bot-chat``: the profile's Bot Chat as its primary home.

``/sethome`` typed in a Desktop/Webapp chat (a local session, through the real ``slash.exec`` RPC)
makes the Bot Chat the home; ``/sethome`` in a messaging chat takes it back. While it is the home, the
messaging prompt names it as the default destination and a DM without a platform home channel is not
nagged to run ``/sethome``.
"""
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import hermes_yaml as yaml
import pytest
import pytest_asyncio

from gateway.config import GatewayConfig, HomeChannel, Platform, PlatformConfig, load_gateway_config
from gateway.run import GatewayRunner
from gateway.session import SessionSource, SessionStore, build_session_context, build_session_context_prompt


def _gateway_section(home):
    return (yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8")) or {}).get("gateway") or {}


@pytest.mark.parametrize("written, loaded", [("bot-chat", "bot-chat"), ("Bot-Chat", "bot-chat"), ("telegram", None)])
def test_gateway_home_loads_from_the_profile_config(tmp_path, monkeypatch, written, loaded):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(f"gateway:\n  home: {written}\n", encoding="utf-8")

    config = load_gateway_config()

    assert config.home == loaded
    assert GatewayConfig.from_dict(config.to_dict()).home == loaded


@pytest_asyncio.fixture
async def desktop(tmp_path, monkeypatch):
    """A gateway runtime with one local (app) session and a Telegram DM source."""
    from pathlib import Path
    from gateway.run_runtime import initialize_gateway_runtime
    from gateway.runtime_ownership import process_ownership
    from gateway.session_authorities import owner_scope
    from gateway.session_controls import AuthorityConnection
    from gateway.session_local import create_local_session
    import hermes_state

    home = tmp_path / ".hermes"
    home.mkdir(mode=0o700)
    (home / "config.yaml").write_text(json.dumps({"model": {
        "default": "fixture", "provider": "custom", "base_url": "http://127.0.0.1:9/v1", "api_key": "loopback-only"}}))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", hermes_state._IMPORT_DEFAULT_DB_PATH)
    config = GatewayConfig()
    config.multiplex_profiles = True
    config._runtime_profile_homes = (("default", home),)
    runner = GatewayRunner(config)
    runner._note_served_profiles(config._runtime_profile_homes)
    process_ownership.reserve([home])
    await initialize_gateway_runtime(runner)
    authority = runner.session_authorities.for_home(home)
    connection = AuthorityConnection(authority, object(), {"user_id": "uid:1000"})
    with owner_scope(authority):
        ref = create_local_session(authority, connection.actor, dict(
            request_id="desktop", source="cli", cwd=str(tmp_path), toolsets=[]))
    telegram = SessionSource(platform=Platform.TELEGRAM, chat_id="42", chat_name="me", user_id="42", chat_type="dm")
    try:
        yield SimpleNamespace(runner=runner, connection=connection, session_id=ref.session_id, home=home,
                              telegram=telegram)
    finally:
        await connection.close()
        authority.db.close()
        process_ownership.release(home)


async def _slash(desktop, command):
    reply = await desktop.connection.dispatch({"id": "1", "method": "slash.exec", "params": {
        "session_id": desktop.session_id, "command": command}})
    assert "result" in reply, reply
    return reply["result"]


async def _rpc(desktop, method, **params):
    reply = await desktop.connection.dispatch({"id": "1", "method": method, "params": {
        "session_id": desktop.session_id, **params}})
    assert "result" in reply, reply
    return reply["result"]


@pytest.mark.asyncio
async def test_the_app_palette_offers_sethome(desktop):
    catalog = await _rpc(desktop, "commands.catalog")
    completions = await _rpc(desktop, "complete.slash", text="/seth")

    assert "/sethome" in dict(catalog["pairs"])
    assert catalog["canon"]["/set-home"] == "/sethome"
    assert "/sethome" in [item["text"] for item in completions["items"]]
    assert "/approve" not in dict(catalog["pairs"])


@pytest.mark.asyncio
async def test_sethome_in_a_desktop_chat_makes_the_bot_chat_the_home(desktop):
    from agent.i18n import t

    result = await _slash(desktop, "/sethome")

    assert result == {"type": "exec", "output": t("gateway.set_home.bot_chat_success")}
    assert _gateway_section(desktop.home)["home"] == "bot-chat"
    assert desktop.runner.config.home == "bot-chat"


@pytest.mark.asyncio
async def test_a_messaging_sethome_takes_the_home_back(desktop):
    from agent.i18n import t
    from gateway.platforms.base import MessageEvent

    await _slash(desktop, "/sethome")
    reply = await desktop.runner._handle_set_home_command(MessageEvent(text="/sethome", source=desktop.telegram))

    assert reply.endswith(t("gateway.set_home.bot_chat_released"))
    assert _gateway_section(desktop.home).get("home") is None
    assert desktop.runner.config.home is None
    assert desktop.runner.config.get_home_channel(Platform.TELEGRAM).chat_id == "42"


@pytest.mark.asyncio
async def test_taking_the_home_back_is_one_save(desktop, monkeypatch):
    """The channel and the cleared ``gateway.home`` land together: a failure after the first write
    must not leave the channel saved while the Bot Chat stays home."""
    import hermes_cli.config as cli_config
    from gateway.platforms.base import MessageEvent

    await _slash(desktop, "/sethome")
    real_save, saves = cli_config.save_config, []

    def save_once(config, *args, **kwargs):
        saves.append(1)
        if len(saves) > 1:
            raise OSError("disk full")
        return real_save(config, *args, **kwargs)

    monkeypatch.setattr(cli_config, "save_config", save_once)
    await desktop.runner._handle_set_home_command(MessageEvent(text="/sethome", source=desktop.telegram))

    on_disk = yaml.safe_load((desktop.home / "config.yaml").read_text(encoding="utf-8"))
    assert (on_disk.get("gateway") or {}).get("home") is None
    assert on_disk["platforms"]["telegram"]["home_channel"]["chat_id"] == "42"
    assert desktop.runner.config.home is None


def _context(home=None):
    config = GatewayConfig(home=home)
    config.platforms[Platform.TELEGRAM] = PlatformConfig(
        enabled=True, token="***", home_channel=HomeChannel(Platform.TELEGRAM, "42", "me"))
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="42", chat_name="me", user_id="42", chat_type="dm")
    return build_session_context(source, config)


def test_the_messaging_prompt_names_the_bot_chat_as_the_primary_home():
    with_home, without = build_session_context_prompt(_context("bot-chat")), build_session_context_prompt(_context())

    assert '`"bot-chat"`' in with_home and "primary home" in with_home
    assert "bot-chat" not in without
    assert '`"telegram"` → Home channel' in with_home


def test_the_pinned_prompt_re_renders_when_the_home_changes():
    runner = GatewayRunner.__new__(GatewayRunner)

    assert runner._ephemeral_change_key(_context("bot-chat"), False) != runner._ephemeral_change_key(_context(), False)


@pytest.mark.asyncio
@pytest.mark.parametrize("home, nudged", [("bot-chat", False), (None, True)])
async def test_a_dm_without_a_home_channel_is_nudged_only_when_the_bot_chat_is_not_home(tmp_path, monkeypatch, home, nudged):
    monkeypatch.setattr("gateway.run._hermes_home", tmp_path)
    monkeypatch.delenv("TELEGRAM_HOME_CHANNEL", raising=False)
    (tmp_path / "config.yaml").write_text("model: test\n", encoding="utf-8")
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(home=home)
    runner.session_store = SessionStore(sessions_dir=tmp_path / "sessions", config=runner.config)
    runner._deliver_platform_notice = AsyncMock()
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="dm", user_id="u", chat_type="dm")

    await runner._hmwa_first_contact_notes(source, [], [])

    assert runner._deliver_platform_notice.await_count == (1 if nudged else 0)
