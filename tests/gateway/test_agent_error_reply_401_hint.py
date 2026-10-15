"""The gateway's 401 exception reply names the turn's provider and the failing profile's own
sign-in (93889b770da: a bare ``hermes auth add <provider>`` from a named profile re-signs the
ROOT store — the loop #114012 measured)."""

import asyncio
from types import SimpleNamespace

import pytest

from gateway.config import Platform
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource


def _reply_for_status(provider, status):
    runner = object.__new__(GatewayRunner)

    async def stop_typing(event, source):
        return None

    runner._hmwa_stop_typing_for_turn = stop_typing
    runner._session_state = lambda key: SimpleNamespace(
        turn=SimpleNamespace(agent=SimpleNamespace(provider=provider) if provider else None))
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="c", user_id="u")
    prepared = runner._PreparedTurn([], "", None, None, None, None)
    err = RuntimeError(f"HTTP {status}: token_revoked")
    err.status_code = status
    return asyncio.run(runner._hmwa_agent_error_reply(
        err, MessageEvent(text="x", source=source), source, None, "k", prepared,
    ))


def _reply_for_401(provider):
    return _reply_for_status(provider, 401)


@pytest.mark.parametrize("status,key", [(401, "hint_auth"), (402, "hint_quota"), (529, "hint_overloaded")])
def test_gateway_status_hints_use_translated_catalog(monkeypatch, status, key):
    from agent.i18n import t

    monkeypatch.setenv("HERMES_LANGUAGE", "de")
    reply = str(_reply_for_status("openai-codex", status))
    expected = t(f"gateway.errors.{key}", relogin="hermes auth add openai-codex --type oauth") if status == 401 else t(f"gateway.errors.{key}")
    # The host command may include a profile flag; check the translated words and valid command separately.
    assert expected.split("`")[0].strip() in reply
    assert "Your AI model service" not in reply and "Your sign-in to" not in reply


def test_gateway_401_reply_names_the_provider_and_the_profile(monkeypatch, tmp_path):
    profile_home = tmp_path / ".hermes" / "profiles" / "codex"
    profile_home.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(profile_home))

    known = _reply_for_401("openai-codex")
    assert "/login" in known
    assert "`hermes -p codex auth add openai-codex --type oauth`" in known, known
    assert "<provider>" not in known and "{relogin}" not in known

    # No agent yet (failure before the turn claimed one): keep the placeholder, still profile-pinned.
    unknown = _reply_for_401(None)
    assert "`hermes -p codex auth add <provider>`" in unknown, unknown
