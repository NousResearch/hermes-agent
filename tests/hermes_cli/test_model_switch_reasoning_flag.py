"""``/model <name> --reasoning <level>`` — one request carries a model pick AND its effort.

The parser is the single owner (hermes_cli.model_switch.parse_model_switch_args); the CLI and
TUI-gateway commit steps apply the effort AFTER the agent swap, because ``agent.switch_model``
re-resolves ``reasoning_config`` from config.yaml and would clobber an earlier write.
"""

from types import SimpleNamespace

import hermes_yaml as yaml
import pytest

from hermes_cli.model_switch import (
    MODEL_SWITCH_ERR_BAD_REASONING,
    ModelSwitchResult,
    parse_model_switch_args,
)


def test_reasoning_flag_rides_with_the_pick_and_validates():
    req = parse_model_switch_args("sonnet --provider anthropic --reasoning high --session")
    assert req.target == "sonnet"
    assert req.explicit_provider == "anthropic"
    assert req.reasoning_effort == "high"
    assert req.scope == "session"
    assert req.errors == ()

    bad = parse_model_switch_args("sonnet --reasoning turbo")
    assert MODEL_SWITCH_ERR_BAD_REASONING in bad.errors
    # Unicode dash normalization (Telegram/iOS) covers the new flag too.
    assert parse_model_switch_args("sonnet \u2014reasoning low").reasoning_effort == "low"


def test_cli_commit_applies_effort_after_the_agent_swap(tmp_path, monkeypatch):
    """The agent's switch_model resets reasoning_config from config; the ride-along effort must
    win over that reset, on both the CLI and the live agent."""
    import cli as cli_mod
    from hermes_cli import cli_model_switch_mixin as mixin

    class _Agent:
        reasoning_config = {"enabled": True, "effort": "medium"}

        def switch_model(self, **_kw):
            # Mirrors agent_runtime_helpers._switch_model: re-resolve from config.yaml.
            self.reasoning_config = {"enabled": True, "effort": "medium"}

    agent = _Agent()
    cli = SimpleNamespace(
        model="old", provider="nous", requested_provider="nous", _explicit_api_key="", _explicit_base_url="",
        api_key="", base_url="", api_mode="", agent=agent, reasoning_config=None,
        _pending_one_turn_model_restore=None, _pending_model_switch_note="",
        _snapshot_model_runtime=lambda: {}, _persist_model_switch_to_session=lambda *_a: None)
    cli._stage_and_swap_model = lambda result, old: cli_mod.HermesCLI._stage_and_swap_model(cli, result, old)
    monkeypatch.setattr(mixin, "_print_switch_summary", lambda *_a, **_k: None)
    monkeypatch.setattr(cli_mod.HermesCLI, "_persist_model_switch_to_session", lambda *_a: None)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    path = tmp_path / "config.yaml"
    before = {"model": {"default": "old", "provider": "nous"}, "agent": {"reasoning_effort": "low"}}
    path.write_text(yaml.safe_dump(before))

    result = ModelSwitchResult(success=True, new_model="new", target_provider="nous")
    mixin._commit_model_switch(cli, result, persist_global=False, reasoning_effort="high")

    assert agent.reasoning_config == {"enabled": True, "effort": "high"}
    assert cli.reasoning_config == {"enabled": True, "effort": "high"}
    assert yaml.safe_load(path.read_text()) == before

    mixin._commit_model_switch(cli, result, persist_global=True, reasoning_effort="none")
    assert yaml.safe_load(path.read_text())["agent"]["reasoning_effort"] == "none"
    assert agent.reasoning_config == {"enabled": False}


@pytest.mark.parametrize("write_fails", [False, True])
def test_cli_global_model_and_effort_use_one_atomic_persist(tmp_path, monkeypatch, write_fails):
    """A global model pick and its effort must share one config write."""
    import cli as cli_mod
    from hermes_cli import cli_model_switch_mixin as mixin

    cli = SimpleNamespace(
        model="old", provider="nous", agent=None, _pending_one_turn_model_restore=None,
        _snapshot_model_runtime=lambda: {}, _persist_model_switch_to_session=lambda *_a: None,
        _stage_and_swap_model=lambda *_a: True,
    )
    monkeypatch.setattr(mixin, "_print_switch_summary", lambda *_a, **_k: None)
    monkeypatch.setattr(cli_mod, "_cprint", lambda *_a, **_k: None)
    monkeypatch.setattr(cli_mod, "save_config_value", lambda *_a, **_k: (_ for _ in ()).throw(
        AssertionError("reasoning must be part of the model write")))
    import utils
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    path = tmp_path / "config.yaml"
    before = {"model": {"default": "old", "provider": "nous"}, "agent": {"reasoning_effort": "low"}}
    path.write_text(yaml.safe_dump(before))
    calls = []
    original = utils.atomic_roundtrip_yaml_update_multi
    def write(target, updates, **kwargs):
        calls.append(dict(updates))
        assert updates["model.default"] == "new"
        assert updates["agent.reasoning_effort"] == "ultra"
        if write_fails:
            raise OSError("synthetic disk failure")
        original(target, updates, **kwargs)
    monkeypatch.setattr(utils, "atomic_roundtrip_yaml_update_multi", write)
    result = ModelSwitchResult(success=True, new_model="new", target_provider="nous")

    mixin._commit_model_switch(cli, result, persist_global=True, reasoning_effort="ultra")

    assert len(calls) == 1
    assert cli.reasoning_config == {"enabled": True, "effort": "ultra"}
    saved = yaml.safe_load(path.read_text())
    if write_fails:
        assert saved == before
    else:
        assert saved["model"]["default"] == "new"
        assert saved["agent"]["reasoning_effort"] == "ultra"

