"""Slack capability metadata must not invent access or prohibit unprobed routes."""

from dataclasses import replace

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.session import (
    SessionSource,
    build_session_context,
    build_session_context_prompt,
)


def _slack_prompt(*, shared=False, message_id="1790000000.000001"):
    source = SessionSource(
        platform=Platform.SLACK,
        chat_id="C_TEST_CHANNEL",
        chat_type="group" if shared else "dm",
        user_name="alice",
        thread_id="1789999999.000001",
        message_id=message_id,
    )
    context = replace(
        build_session_context(source, GatewayConfig()),
        shared_multi_user_session=shared,
    )
    return build_session_context_prompt(context)


@pytest.mark.parametrize("shared", [False, True])
def test_no_dedicated_slack_tools_does_not_deny_other_authorized_routes(monkeypatch, shared):
    monkeypatch.setattr("gateway.session._slack_tools_loaded", lambda: False)

    prompt = _slack_prompt(shared=shared)

    # The registry cannot determine availability or authorization of terminal clients.
    assert "No dedicated Slack tools are loaded" in prompt
    assert "CLI" in prompt and "API client" in prompt
    assert "already configured and authorized" in prompt
    assert "instructions and permissions" in prompt
    assert "do not bypass" in prompt
    assert "cannot call Slack APIs" not in prompt
    assert "do NOT have access to Slack-specific APIs" not in prompt
    assert "have access to Slack-specific tools this session" not in prompt
    assert "that alone does not provide live Slack access" in prompt
    assert ("current turn's sender prefix" in prompt) is shared


@pytest.mark.parametrize("tools_loaded", [False, True])
def test_slack_capability_note_remains_stable_and_reports_dedicated_tools(monkeypatch, tools_loaded):
    monkeypatch.setattr("gateway.session._slack_tools_loaded", lambda: tools_loaded)

    first_id, second_id = "1790000000.000001", "1790000000.000002"
    first = _slack_prompt(shared=True, message_id=first_id)
    second = _slack_prompt(shared=True, message_id=second_id)

    assert first == second
    assert first_id not in first and second_id not in first
    assert "current turn's sender prefix" in first
    assert ("have access to Slack-specific tools this session" in first) is tools_loaded
    if tools_loaded:
        assert "available Slack tool schemas" in first
        assert "No dedicated Slack tools are loaded" not in first
