"""First-contact onboarding does not ask chats to configure delivery defaults."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from gateway.config import Platform
from gateway.run_turn import GatewayTurnMixin


@pytest.mark.parametrize("platform", [Platform.TELEGRAM, Platform.SLACK])
@pytest.mark.parametrize("profile", ["default", "secondary"])
@pytest.mark.parametrize("prior_sessions", [False, True])
def test_first_contact_keeps_onboarding_without_platform_notice(
    monkeypatch, tmp_path, platform, profile, prior_sessions,
):
    from agent import onboarding
    from gateway import run

    config = SimpleNamespace()
    monkeypatch.setattr(run, "_hermes_home", tmp_path)
    monkeypatch.setattr(run, "_load_gateway_config", lambda: config)
    first_contact_note = Mock(return_value="First-contact onboarding")
    monkeypatch.setattr(onboarding, "first_contact_turn_note", first_contact_note)
    runner = SimpleNamespace(
        async_session_store=SimpleNamespace(
            has_any_sessions=AsyncMock(return_value=prior_sessions),
        ),
        _deliver_platform_notice=AsyncMock(),
    )
    source = SimpleNamespace(platform=platform, profile=profile)
    notes = []

    asyncio.run(GatewayTurnMixin._hmwa_first_contact_notes(runner, source, [], notes))

    runner._deliver_platform_notice.assert_not_awaited()
    if prior_sessions:
        assert notes == []
        first_contact_note.assert_not_called()
    else:
        assert notes == ["First-contact onboarding"]
        first_contact_note.assert_called_once_with(
            config, tmp_path / "config.yaml",
            session_history_empty=True, install_has_prior_sessions=False,
        )
