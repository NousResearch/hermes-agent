"""Discord /agents links use structured destinations."""
from types import SimpleNamespace

import pytest

from gateway.config import Platform
from gateway.session import SessionSource, build_session_key


def _runner(source, *, pending=False):
    from gateway.run import GatewayRunner, _AGENT_PENDING_SENTINEL

    runner = object.__new__(GatewayRunner)
    runner._session_key_for_source = lambda value: build_session_key(value, profile=value.profile)
    key = runner._session_key_for_source(source)
    runner._cache_session_source(key, source)
    runner._running_agents = {
        key: _AGENT_PENDING_SENTINEL if pending else SimpleNamespace(session_id="session-1", model="test-model"),
    }
    runner._running_agents_ts = {key: 100}
    runner._background_tasks = set()
    return runner, key


@pytest.mark.asyncio
@pytest.mark.parametrize("profile", [None, "research"])
async def test_discord_agents_link_thread_without_changing_row_details(monkeypatch, profile):
    from gateway import slash_commands_status

    source = SessionSource(
        platform=Platform.DISCORD, chat_id="111111111111111111", chat_type="thread",
        thread_id="222222222222222222", parent_chat_id="333333333333333333", profile=profile,
    )
    runner, key = _runner(source)
    monkeypatch.setattr(slash_commands_status.time, "time", lambda: 165)

    result = await runner._handle_agents_command(SimpleNamespace(source=source))

    assert "1. <#222222222222222222> · running · 1m 5s · `session-1` · `test-model`" in result
    assert " · this chat" in result
    assert key not in result
    assert "<#111111111111111111>" not in result
    assert "<#333333333333333333>" not in result


@pytest.mark.asyncio
@pytest.mark.parametrize("pending", [False, True])
@pytest.mark.parametrize("chat_type,prospective,expected", [
    ("channel", None, "111111111111111111"),
    ("thread", None, "111111111111111111"),
    ("channel", "222222222222222222", "222222222222222222"),
])
async def test_discord_agents_destination_without_thread_id(chat_type, prospective, expected, pending):
    source = SessionSource(
        platform=Platform.DISCORD, chat_id="111111111111111111", chat_type=chat_type,
        prospective_thread_id=prospective, parent_chat_id="333333333333333333",
    )
    runner, key = _runner(source, pending=pending)

    result = await runner._handle_agents_command(SimpleNamespace(source=source))

    state = "starting" if pending else "running"
    assert f"1. <#{expected}> · {state} · " in result
    assert key not in result
    assert ("`test-model`" in result) is not pending


@pytest.mark.asyncio
@pytest.mark.parametrize("pending", [False, True])
@pytest.mark.parametrize("missing", ["cache", "destination", "profile"])
async def test_discord_agents_keep_key_without_matching_metadata(missing, pending):
    source = SessionSource(platform=Platform.DISCORD, chat_id="111111111111111111", profile="research")
    runner, key = _runner(source, pending=pending)
    if missing == "cache":
        runner._session_sources.clear()
    elif missing == "destination":
        runner._cache_session_source(key, SessionSource(platform=Platform.DISCORD, chat_id=""))
    else:
        cached = runner._session_sources.pop(key)
        runner._cache_session_source(build_session_key(source), cached)

    result = await runner._handle_agents_command(SimpleNamespace(source=source))

    assert f"1. `{key}` · " in result
    assert "<#" not in result


@pytest.mark.asyncio
@pytest.mark.parametrize("request_platform", [
    Platform.DISCORD, Platform.TELEGRAM, Platform.SLACK, Platform.WHATSAPP, Platform.SIGNAL, Platform.LOCAL,
])
async def test_agents_mentions_are_discord_only_in_mixed_rows(request_platform, monkeypatch):
    from gateway import slash_commands_status

    source = SessionSource(platform=Platform.DISCORD, chat_id="111111111111111111", thread_id="222222222222222222")
    runner, key = _runner(source)
    other = SessionSource(platform=Platform.TELEGRAM, chat_id="333333333333333333", thread_id="444444444444444444")
    other_key = runner._session_key_for_source(other)
    runner._cache_session_source(other_key, other)
    runner._running_agents[other_key] = SimpleNamespace(session_id="session-2", model="other-model")
    request = SimpleNamespace(source=SessionSource(platform=request_platform, chat_id="555555555555555555"))
    monkeypatch.setattr(slash_commands_status.time, "time", lambda: 165)

    result = await runner._handle_agents_command(request)
    runner._session_sources.clear()
    without_metadata = await runner._handle_agents_command(request)

    if request_platform == Platform.DISCORD:
        assert result == without_metadata.replace(f"`{key}`", "<#222222222222222222>")
    else:
        assert result == without_metadata
    assert f"`{other_key}`" in result
