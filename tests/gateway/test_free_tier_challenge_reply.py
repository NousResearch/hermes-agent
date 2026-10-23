"""What a messaging user is told when the free tier's browser check blocks a turn.

The trial runs every surface once the Desktop claimed it, but only the Desktop can clear the account
service's browser challenge: the gateway never waits on one and opens nothing. A messaging turn that
hits it must say where the check is finished, not blame the sign-in the user never had.
"""
from __future__ import annotations

import pytest

from gateway.config import Platform
from tests.hermes_cli.anon_portal import claimed_trial, install_portal, off_local_desktop


@pytest.fixture
def challenged_gateway(monkeypatch, tmp_path):
    from gateway import status
    from hermes_cli import anon_challenge

    nas = install_portal(monkeypatch, tmp_path)
    claimed_trial(monkeypatch)
    off_local_desktop(monkeypatch)  # a bot's turn, not the Desktop's
    monkeypatch.setattr(status, "owns_gateway_runtime_lock", lambda: True)
    monkeypatch.setattr(anon_challenge, "CHALLENGE_WAIT_SECONDS", 0.05)
    nas.challenge_required = True  # the Desktop's clearance has lapsed: the next exchange is challenged
    nas.challenge_never_clears = True
    return nas


def test_a_messaging_turn_blocked_by_the_browser_check_says_to_open_the_desktop_app(challenged_gateway):
    from agent.i18n import t
    from gateway.run import _resolve_runtime_agent_kwargs
    from gateway.run_turn_runner import _resolution_failure_reply

    with pytest.raises(Exception) as exc:
        _resolve_runtime_agent_kwargs()
    text, reason, retryable = _resolution_failure_reply(exc.value, Platform.TELEGRAM)

    assert text == t("gateway.errors.free_tier_check_pending")
    assert (reason, retryable) == ("auth", True)
    assert "/api/anonymous/challenge/status" not in [p for _, p in challenged_gateway.calls]
