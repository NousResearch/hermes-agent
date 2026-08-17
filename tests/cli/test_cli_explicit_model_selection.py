"""Regression tests for CLI explicit model-selection propagation."""

from types import SimpleNamespace

from hermes_cli.cli_agent_setup_mixin import propagate_explicit_model_selection


def test_cli_model_flag_marks_new_agent_as_explicitly_selected():
    cli = SimpleNamespace(_explicit_model_override=True)
    agent = SimpleNamespace(_model_explicitly_selected=False)

    propagate_explicit_model_selection(cli, agent)

    assert agent._model_explicitly_selected is True


def test_default_cli_route_keeps_new_agent_unlocked():
    cli = SimpleNamespace(_explicit_model_override=False)
    agent = SimpleNamespace(_model_explicitly_selected=True)

    propagate_explicit_model_selection(cli, agent)

    assert agent._model_explicitly_selected is False


def test_explicit_cli_model_persists_custom_runtime_for_resume():
    cli = SimpleNamespace(_explicit_model_override=True)
    agent = SimpleNamespace(
        _model_explicitly_selected=False,
        _session_init_model_config={"max_iterations": 90},
    )
    runtime = {
        "provider": "custom",
        "requested_provider": "custom:turbohaul-local",
        "base_url": "http://127.0.0.1:11410/v1",
        "api_mode": "chat_completions",
    }

    propagate_explicit_model_selection(cli, agent, runtime=runtime)

    assert agent._session_init_model_config["gateway_runtime"] == {
        "provider": "custom:turbohaul-local",
        "base_url": "http://127.0.0.1:11410/v1",
        "api_mode": "chat_completions",
    }


def test_explicit_runtime_initialises_missing_session_config_on_test_double():
    cli = SimpleNamespace(_explicit_model_override=True)
    agent = SimpleNamespace(_model_explicitly_selected=False)

    propagate_explicit_model_selection(
        cli,
        agent,
        runtime={"requested_provider": "openrouter"},
    )

    assert agent._session_init_model_config == {
        "gateway_runtime": {
            "provider": "openrouter",
            "base_url": None,
            "api_mode": None,
        }
    }