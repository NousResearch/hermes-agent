"""The one-time "no home channel, run /sethome" nudge only goes to senders who may run /sethome."""

from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.run import GatewayRunner
from gateway.session import SessionSource, SessionStore


def _runner(tmp_path, monkeypatch, extra):
    monkeypatch.setattr("gateway.run._hermes_home", tmp_path)
    monkeypatch.delenv("WHATSAPP_HOME_CHANNEL", raising=False)
    (tmp_path / "config.yaml").write_text("model: test\n", encoding="utf-8")
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(platforms={Platform.WHATSAPP: PlatformConfig(enabled=True, extra=extra)})
    runner.session_store = SessionStore(sessions_dir=tmp_path / "sessions", config=runner.config)
    runner._deliver_platform_notice = AsyncMock()
    return runner


async def _first_contact(runner, user_id, *, chat_type="dm"):
    source = SessionSource(platform=Platform.WHATSAPP, chat_id=f"{user_id}@s.whatsapp.net",
                           user_id=user_id, chat_type=chat_type)
    await runner._hmwa_first_contact_notes(source, [], [])
    return runner._deliver_platform_notice.await_args_list


@pytest.mark.asyncio
async def test_non_admin_customer_never_sees_the_sethome_nudge(tmp_path, monkeypatch):
    runner = _runner(tmp_path, monkeypatch, {"allow_admin_from": ["operator"]})
    assert await _first_contact(runner, "customer") == []


@pytest.mark.asyncio
@pytest.mark.parametrize("extra, user_id", [
    ({"allow_admin_from": ["operator"]}, "operator"),
    ({"allow_admin_from": ["operator"], "user_allowed_commands": ["sethome"]}, "customer"),
    ({}, "anyone"),  # no admin list: gating off, historical behaviour
])
async def test_senders_who_may_run_sethome_still_get_the_nudge(tmp_path, monkeypatch, extra, user_id):
    runner = _runner(tmp_path, monkeypatch, extra)
    calls = await _first_contact(runner, user_id)
    assert len(calls) == 1 and "sethome" in calls[0].args[1]


@pytest.mark.asyncio
async def test_group_nudge_uses_group_admins_not_dm_admins(tmp_path, monkeypatch):
    runner = _runner(tmp_path, monkeypatch, {
        "allow_admin_from": ["dm_operator"],
        "group_allow_admin_from": ["group_operator"],
    })
    # Permission to run /sethome in a DM does not confer group admin rights.
    assert await _first_contact(runner, "dm_operator", chat_type="group") == []
    calls = await _first_contact(runner, "group_operator", chat_type="group")
    assert len(calls) == 1 and "sethome" in calls[0].args[1]
