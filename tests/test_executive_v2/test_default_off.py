"""Tests for end-to-end default-off behavior of the engine."""

from __future__ import annotations

import pytest

from agent.executive.flag import resolve_v2_enabled
from agent.executive.objective_engine import ObjectiveEngine, PermissionError_


@pytest.mark.xfail(
    reason="depends on 4d8b9d309 HOLD_TANDA_5_5",
    strict=False,
)
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
