"""Tests for end-to-end default-off behavior of the engine."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from agent.executive.flag import resolve_v2_enabled
from agent.executive.objective_engine import ObjectiveEngine, PermissionError_


def test_cli_handler_dryrun_method_exists():
    """The /objective CLI handler method must exist in cli_commands_mixin."""
    from hermes_cli.cli_commands_mixin import CLICommandsMixin
    assert hasattr(CLICommandsMixin, "_handle_executive_v2_dryrun")
    assert callable(
        getattr(CLICommandsMixin, "_handle_executive_v2_dryrun")
    )


def test_default_off_no_submit(clean_env_executive):
    """Engine is disabled by default: submit() raises PermissionError_."""
    e = ObjectiveEngine(user_id="u", enabled=False)
    with pytest.raises(PermissionError_):
        e.submit("text")


def test_default_off_no_normalize(clean_env_executive):
    e = ObjectiveEngine(user_id="u", enabled=False)
    with pytest.raises(PermissionError_):
        e.normalize("oid")


def test_default_off_no_classify(clean_env_executive):
    e = ObjectiveEngine(user_id="u", enabled=False)
    with pytest.raises(PermissionError_):
        e.classify("oid")


def test_default_off_no_discover(clean_env_executive):
    e = ObjectiveEngine(user_id="u", enabled=False)
    with pytest.raises(PermissionError_):
        e.discover("oid")


def test_default_off_no_generate_contract(clean_env_executive):
    e = ObjectiveEngine(user_id="u", enabled=False)
    with pytest.raises(PermissionError_):
        e.generate_contract("oid")


def test_default_off_no_persist(clean_env_executive):
    e = ObjectiveEngine(user_id="u", enabled=False)
    with pytest.raises(PermissionError_):
        e.persist("oid")


def test_default_off_no_run_pipeline(clean_env_executive):
    e = ObjectiveEngine(user_id="u", enabled=False)
    with pytest.raises(PermissionError_):
        e.run_pipeline("text")


def test_enabled_via_env_var(clean_env_executive, monkeypatch):
    """Env var enables the engine."""
    monkeypatch.setenv("HERMES_EXECUTIVE_V2_ENABLED", "1")
    e = ObjectiveEngine(user_id="u", enabled=None)
    assert e.enabled is True
    # submit() works.
    oid = e.submit("text")
    assert oid


def test_enabled_via_constructor(clean_env_executive):
    """Constructor arg enables the engine."""
    e = ObjectiveEngine(user_id="u", enabled=True)
    assert e.enabled is True
    oid = e.submit("text")
    assert oid


def test_config_default_false():
    """DEFAULT_CONFIG exposes Executive v2 as a default-off agent setting."""
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["agent"]["executive_v2_enabled"] is False


def test_objective_engine_accepts_explicit_config_value(clean_env_executive, monkeypatch):
    """Integration boundaries can pass explicit raw config without hermes_cli imports."""
    monkeypatch.setenv("HERMES_EXECUTIVE_V2_ENABLED", "1")
    disabled = ObjectiveEngine(
        user_id="u",
        executive_v2_config_value=False,
    )
    assert disabled.enabled is False

    enabled = ObjectiveEngine(
        user_id="u",
        executive_v2_config_value=True,
    )
    assert enabled.enabled is True


def test_objective_disabled_message_mentions_config_not_env_or_agent(clean_env_executive):
    engine = ObjectiveEngine(user_id="u", enabled=False)

    with pytest.raises(PermissionError_) as excinfo:
        engine.submit("text")

    message = str(excinfo.value)
    assert "hermes config set agent.executive_v2_enabled true" in message
    assert "HERMES_EXECUTIVE_V2_ENABLED" not in message
    assert "_executive_v2_enabled" not in message


def test_objective_cli_disabled_message_mentions_config_not_env_or_agent(
    clean_env_executive, capsys
):
    """/objective disabled guidance is user-facing config only."""
    from hermes_cli.cli_commands_mixin import CLICommandsMixin

    cli = CLICommandsMixin()
    with patch("hermes_cli.config.read_raw_config", return_value={}):
        cli._handle_executive_v2_dryrun("/objective ship it")

    out = capsys.readouterr().out
    assert "hermes config set agent.executive_v2_enabled true" in out
    assert "HERMES_EXECUTIVE_V2_ENABLED" not in out
    assert "_executive_v2_enabled" not in out


def test_objective_cli_without_agent_uses_explicit_config_true(
    clean_env_executive, capsys
):
    """/objective without an agent uses canonical config.yaml enablement."""
    from hermes_cli.cli_commands_mixin import CLICommandsMixin

    cli = CLICommandsMixin()
    assert not hasattr(cli, "agent")
    raw_config = {"agent": {"executive_v2_enabled": True}}

    with patch("hermes_cli.config.read_raw_config", return_value=raw_config):
        cli._handle_executive_v2_dryrun("/objective ship it")

    out = capsys.readouterr().out
    assert "Executive v2 is disabled" not in out
    assert "persist/cancel are not supported by /objective dry-run" in out
