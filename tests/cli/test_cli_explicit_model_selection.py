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