"""A messaging gateway's first-ever DM carries the baseline profile-build offer."""

from unittest.mock import AsyncMock

import hermes_yaml as yaml
import pytest

from gateway.config import GatewayConfig, Platform
from gateway.run import GatewayRunner
from gateway.session import SessionSource, SessionStore


def _runner(tmp_path, monkeypatch):
    monkeypatch.setattr("gateway.run._hermes_home", tmp_path)
    monkeypatch.setenv("TELEGRAM_HOME_CHANNEL", "home")
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    runner.session_store = SessionStore(sessions_dir=tmp_path / "sessions", config=runner.config)
    runner._deliver_platform_notice = AsyncMock()
    return runner


@pytest.mark.asyncio
@pytest.mark.parametrize("guest", ["1", "0"])
async def test_first_dm_gets_the_profile_build_offer_once(tmp_path, monkeypatch, guest):
    from agent.onboarding import PROFILE_BUILD_FLAG, profile_build_directive

    monkeypatch.setenv("HERMES_GUEST_ONBOARDING", guest)
    (tmp_path / "config.yaml").write_text("model: test\n", encoding="utf-8")
    runner = _runner(tmp_path, monkeypatch)
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="dm", user_id="u", chat_type="dm")

    first: list[str] = []
    await runner._hmwa_first_contact_notes(source, [], first)
    second: list[str] = []
    await runner._hmwa_first_contact_notes(source, [], second)

    assert first == [profile_build_directive().strip()]
    assert len(second) == 1 and second[0] != first[0]
    seen = yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))["onboarding"]["seen"]
    assert seen[PROFILE_BUILD_FLAG] is True
