"""Materializing NOVA agents into Hermes profiles."""

from __future__ import annotations

import json

import pytest
import yaml

from nova.audit import new_correlation_id
from nova.errors import RuntimeAdapterError
from nova.runtime.hermes import materialize as M
from nova.runtime.hermes.paths import HermesPaths, resolve_home
from nova.spec import AgentSpec


def _spec(**overrides):
    data = {"id": "support", "name": "Support", "instructions_text": "You are Support."}
    data.update(overrides)
    return AgentSpec.parse(data)


# -- home resolution ---------------------------------------------------------


def test_nova_home_wins(monkeypatch, tmp_path):
    monkeypatch.setenv("NOVA_HOME", str(tmp_path / "n"))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "h"))
    assert resolve_home() == tmp_path / "n"


def test_hermes_home_still_honoured(monkeypatch, tmp_path):
    """An existing Hermes installation must keep working untouched."""
    monkeypatch.delenv("NOVA_HOME", raising=False)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "h"))
    assert resolve_home() == tmp_path / "h"


def test_existing_hermes_directory_is_adopted(tmp_path):
    hermes = tmp_path / ".hermes"
    hermes.mkdir()
    (hermes / "config.yaml").write_text("{}", encoding="utf-8")
    assert resolve_home({"HOME": str(tmp_path)}) == hermes


def test_fresh_install_prefers_nova(tmp_path):
    assert resolve_home({"HOME": str(tmp_path)}) == tmp_path / ".nova"


# -- config compilation ------------------------------------------------------


def test_unset_fields_are_omitted_so_runtime_defaults_apply():
    config = M.build_config(_spec())
    assert config == {}


def test_limits_compile_to_runtime_keys():
    config = M.build_config(_spec(limits={"max_turns": 60, "max_concurrent_tasks": 3}))
    assert config["agent"]["max_turns"] == 60
    assert config["kanban"]["max_in_progress_per_profile"] == 3


def test_tool_denials_compile_to_unconditional_deny():
    config = M.build_config(_spec(tools={"deny": ["terminal"]}))
    assert config["approvals"]["deny"] == ["terminal*"]


def test_never_emits_a_config_key_the_runtime_does_not_read():
    """Regression: 'enabled_toolsets' is not a key the runtime reads.

    It resolves toolsets from ``platform_toolsets[<surface>]``. Emitting the wrong key
    would leave a customer's restriction looking applied while doing nothing.
    """
    config = M.build_config(_spec(tools={"toolsets": ["web"], "deny": ["terminal"]}))
    assert "enabled_toolsets" not in config


def test_positive_tool_scoping_is_recorded_and_warned_about(home, audit, runtime):
    """Not enforcing it is acceptable; pretending to enforce it is not."""
    result = runtime.materialize_agent(
        _spec(tools={"toolsets": ["web"]}), audit=audit, correlation_id=new_correlation_id()
    )
    assert any("not yet enforced" in w for w in result.warnings)
    config = M.build_config(_spec(tools={"toolsets": ["web"]}))
    assert config["nova"]["toolsets"] == ["web"]


def test_declared_but_unenforced_intent_is_preserved():
    """Customer intent must survive to the phase that enforces it, not be dropped."""
    config = M.build_config(
        _spec(permissions=["read_customers"], approval={"required_for": ["refund"]})
    )
    assert config["nova"]["permissions"] == ["read_customers"]
    assert config["nova"]["approval_required_for"] == ["refund"]


def test_persona_uses_branded_display_name():
    from nova.spec import IdentitySpec

    identity = IdentitySpec.parse(
        {"product_name": "Acme", "agents": {"support": {"display_name": "Acme Helper"}}}
    )
    persona = M.build_persona(_spec(instructions_text=""), identity)
    assert "Acme Helper" in persona


# -- materialization ---------------------------------------------------------


def test_creates_profile_files(home, audit, runtime):
    result = runtime.materialize_agent(_spec(), audit=audit, correlation_id=new_correlation_id())
    paths = HermesPaths(home=home)
    assert result.created is True
    assert paths.config_path("support").is_file()
    assert paths.persona_path("support").is_file()
    assert paths.provenance_path("support").is_file()


def test_is_idempotent(home, audit, runtime):
    spec = _spec()
    runtime.materialize_agent(spec, audit=audit, correlation_id=new_correlation_id())
    second = runtime.materialize_agent(spec, audit=audit, correlation_id=new_correlation_id())
    assert second.unchanged is True
    assert second.paths_written == ()


def test_changed_spec_rewrites(home, audit, runtime):
    runtime.materialize_agent(_spec(), audit=audit, correlation_id=new_correlation_id())
    result = runtime.materialize_agent(
        _spec(limits={"max_turns": 99}), audit=audit, correlation_id=new_correlation_id()
    )
    assert result.changed is True
    config = yaml.safe_load(HermesPaths(home=home).config_path("support").read_text(encoding="utf-8"))
    assert config["agent"]["max_turns"] == 99


def test_deleted_file_is_restored_despite_current_provenance(home, audit, runtime):
    """A stale marker must not report an agent as healthy when its files are gone."""
    runtime.materialize_agent(_spec(), audit=audit, correlation_id=new_correlation_id())
    HermesPaths(home=home).persona_path("support").unlink()
    result = runtime.materialize_agent(_spec(), audit=audit, correlation_id=new_correlation_id())
    assert HermesPaths(home=home).persona_path("support").is_file()
    assert any("missing" in w for w in result.warnings)


def test_refuses_to_overwrite_a_profile_nova_does_not_own(home, audit, runtime):
    """A hand-built profile must never be silently replaced."""
    handmade = HermesPaths(home=home).profile_dir("support")
    handmade.mkdir(parents=True)
    (handmade / "config.yaml").write_text("model: {model: handmade}\n", encoding="utf-8")
    with pytest.raises(RuntimeAdapterError, match="not created by NOVA"):
        runtime.materialize_agent(_spec(), audit=audit, correlation_id=new_correlation_id())
    assert "handmade" in (handmade / "config.yaml").read_text(encoding="utf-8")


def test_never_writes_customer_owned_files(home):
    paths = HermesPaths(home=home)
    paths.profile_dir("support").mkdir(parents=True)
    for name in ("auth.json", ".env", "state.db"):
        with pytest.raises(RuntimeAdapterError, match="does not own"):
            M.atomic_write(paths.profile_dir("support") / name, "nope")


def test_dry_run_writes_nothing_but_reports_the_plan(home, audit, runtime):
    result = runtime.materialize_agent(
        _spec(), audit=audit, correlation_id=new_correlation_id(), dry_run=True
    )
    assert result.created is True
    assert len(result.paths_written) == 3
    assert not HermesPaths(home=home).profile_dir("support").exists()


def test_dry_run_plan_matches_the_real_run(home, audit, runtime):
    planned = runtime.materialize_agent(
        _spec(), audit=audit, correlation_id=new_correlation_id(), dry_run=True
    ).paths_written
    actual = runtime.materialize_agent(
        _spec(), audit=audit, correlation_id=new_correlation_id()
    ).paths_written
    assert planned == actual


def test_disabled_agent_warns(home, audit, runtime):
    result = runtime.materialize_agent(
        _spec(enabled=False), audit=audit, correlation_id=new_correlation_id()
    )
    assert any("disabled" in w for w in result.warnings)


def test_provenance_records_the_spec_digest(home, audit, runtime):
    spec = _spec()
    runtime.materialize_agent(spec, audit=audit, correlation_id=new_correlation_id())
    data = json.loads(HermesPaths(home=home).provenance_path("support").read_text(encoding="utf-8"))
    assert data["digest"] == spec.digest()
    assert data["agent_id"] == "support"


# -- listing and removal -----------------------------------------------------


def test_list_agents_distinguishes_managed_from_foreign(home, audit, runtime):
    runtime.materialize_agent(_spec(), audit=audit, correlation_id=new_correlation_id())
    (HermesPaths(home=home).profile_dir("handmade")).mkdir(parents=True)
    listed = {a.agent_id: a.managed_by_nova for a in runtime.list_agents()}
    assert listed == {"support": True, "handmade": False}


def test_list_agents_on_empty_home(runtime):
    assert runtime.list_agents() == []


def test_remove_agent(home, audit, runtime):
    runtime.materialize_agent(_spec(), audit=audit, correlation_id=new_correlation_id())
    assert runtime.remove_agent("support", audit=audit, correlation_id=new_correlation_id()) is True
    assert not HermesPaths(home=home).profile_dir("support").exists()


def test_remove_missing_agent_returns_false(runtime, audit):
    assert runtime.remove_agent("ghost", audit=audit, correlation_id=new_correlation_id()) is False


def test_remove_refuses_foreign_profile(home, runtime, audit):
    (HermesPaths(home=home).profile_dir("handmade")).mkdir(parents=True)
    with pytest.raises(RuntimeAdapterError, match="not created by NOVA"):
        runtime.remove_agent("handmade", audit=audit, correlation_id=new_correlation_id())


def test_remove_refuses_when_customer_data_is_present(home, audit, runtime):
    """Deleting an agent must never delete its credentials or history."""
    runtime.materialize_agent(_spec(), audit=audit, correlation_id=new_correlation_id())
    (HermesPaths(home=home).profile_dir("support") / "auth.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeAdapterError, match="customer state"):
        runtime.remove_agent("support", audit=audit, correlation_id=new_correlation_id())
    assert HermesPaths(home=home).profile_dir("support").exists()


# -- audit integration -------------------------------------------------------


def test_materialization_is_logged_as_model_visible(home, audit, runtime):
    runtime.materialize_agent(_spec(), audit=audit, correlation_id=new_correlation_id())
    events = [e for e in audit.read() if e.kind == "agent.materialized"]
    assert [e.phase for e in events] == ["intent", "committed"]
    assert all(e.model_visible for e in events)


def test_failed_materialization_leaves_a_failed_record(home, audit, runtime):
    handmade = HermesPaths(home=home).profile_dir("support")
    handmade.mkdir(parents=True)
    with pytest.raises(RuntimeAdapterError):
        runtime.materialize_agent(_spec(), audit=audit, correlation_id=new_correlation_id())
    assert [e.phase for e in audit.read() if e.kind == "agent.materialized"] == ["intent", "failed"]


def test_dry_run_is_not_recorded_as_model_visible(home, audit, runtime):
    runtime.materialize_agent(
        _spec(), audit=audit, correlation_id=new_correlation_id(), dry_run=True
    )
    assert all(not e.model_visible for e in audit.read())
