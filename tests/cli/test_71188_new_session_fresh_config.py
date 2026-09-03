"""#71188: /new must read fresh config, not the stale CLI_CONFIG snapshot."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch


def _make_stub(*, model: str = "session-switched-model"):
    agent = SimpleNamespace(
        reasoning_config={"enabled": True, "effort": "high"},
        reset_session_state=MagicMock(),
        switch_model=MagicMock(),
    )
    stub = SimpleNamespace(
        agent=agent,
        conversation_history=[],
        session_id="old-session",
        _session_db=None,
        _pending_title=None,
        _resumed=False,
        reasoning_config={"enabled": True, "effort": "high"},
        _notify_session_boundary=MagicMock(),
        service_tier="priority",
        _pending_one_turn_model_restore={"model": "stale"},
        model=model,
        provider="openrouter",
        requested_provider="openrouter",
        api_key="k",
        base_url="",
        api_mode="",
    )
    return stub, agent


def test_new_session_uses_fresh_load_cli_config_not_stale_CLI_CONFIG():
    """Keep CLI_CONFIG stale; fresh load_cli_config() must drive /new defaults."""
    from cli import CLI_CONFIG, HermesCLI

    stub, agent = _make_stub(model="session-switched-model")

    stale_model = {"default": "stale-import-model", "provider": "openrouter"}
    fresh_config = {
        "agent": {
            "reasoning_effort": "medium",
            "service_tier": "normal",
        },
        "model": {
            "default": "fresh-config-model",
            "provider": "openrouter",
        },
    }
    fake_result = SimpleNamespace(
        success=True,
        new_model="fresh-config-model",
        target_provider="openrouter",
        api_key="k2",
        base_url="https://openrouter.ai/api/v1",
        api_mode="chat_completions",
    )

    with patch.dict(
        CLI_CONFIG.setdefault("agent", {}),
        {"reasoning_effort": "low", "service_tier": "priority"},
    ), patch.dict(
        CLI_CONFIG,
        {"model": stale_model},
    ), patch(
        "cli.load_cli_config",
        return_value=fresh_config,
    ) as load_fresh, patch(
        "hermes_cli.model_switch.switch_model",
        return_value=fake_result,
    ) as switch_model:
        HermesCLI.new_session(stub, silent=True)

    load_fresh.assert_called()
    assert stub.reasoning_config == {"enabled": True, "effort": "medium"}
    assert agent.reasoning_config == {"enabled": True, "effort": "medium"}
    assert stub.service_tier is None
    assert stub._pending_one_turn_model_restore is None
    assert stub.model == "fresh-config-model"
    assert stub.model != "stale-import-model"
    switch_model.assert_called_once()
    assert switch_model.call_args.kwargs["raw_input"] == "fresh-config-model"
    agent.switch_model.assert_called_once()
    agent.reset_session_state.assert_called_once()


def test_new_session_ignores_stale_CLI_CONFIG_when_fresh_differs_only_in_model():
    from cli import CLI_CONFIG, HermesCLI

    stub, agent = _make_stub(model="old-session-model")
    fresh_config = {
        "agent": {"reasoning_effort": "medium", "service_tier": ""},
        "model": {"default": "only-fresh-model", "provider": "openrouter"},
    }
    fake_result = SimpleNamespace(
        success=True,
        new_model="only-fresh-model",
        target_provider="openrouter",
        api_key="k",
        base_url="",
        api_mode="chat_completions",
    )

    with patch.dict(
        CLI_CONFIG.setdefault("agent", {}),
        {"reasoning_effort": "medium", "service_tier": ""},
    ), patch.dict(
        CLI_CONFIG,
        {"model": {"default": "stale-model", "provider": "openrouter"}},
    ), patch(
        "cli.load_cli_config",
        return_value=fresh_config,
    ), patch(
        "hermes_cli.model_switch.switch_model",
        return_value=fake_result,
    ) as switch_model:
        HermesCLI.new_session(stub, silent=True)

    assert stub.model == "only-fresh-model"
    assert switch_model.call_args.kwargs["raw_input"] == "only-fresh-model"
    agent.reset_session_state.assert_called_once()


def test_new_session_tolerates_thin_fresh_config_without_agent_key():
    from cli import HermesCLI

    stub, agent = _make_stub(model="keep-me")
    with patch("cli.load_cli_config", return_value={"model": {}}), patch(
        "hermes_cli.model_switch.switch_model"
    ) as switch_model:
        HermesCLI.new_session(stub, silent=True)

    switch_model.assert_not_called()
    assert stub.model == "keep-me"
    agent.reset_session_state.assert_called_once()


def test_new_session_tolerates_non_dict_agent_section():
    from cli import HermesCLI

    stub, agent = _make_stub(model="keep-me")
    with patch(
        "cli.load_cli_config",
        return_value={"agent": "broken", "model": {}},
    ):
        HermesCLI.new_session(stub, silent=True)
    agent.reset_session_state.assert_called_once()
