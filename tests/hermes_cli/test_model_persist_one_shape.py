"""One config.yaml shape for a persisted model selection, whichever surface wrote it.

``/model --global`` reaches config.yaml from four places (CLI mixin, gateway slash command, TUI
gateway, dashboard main slot). Each used to hand-roll its own write; the TUI never touched
``api_mode`` (stale wire protocol after a switch) and the dashboard wrote ``base_url: ""``. All
four now go through ``hermes_cli.model_switch.persist_model_selection`` /
``apply_model_selection``, so the same ``ModelSwitchResult`` must land as the same ``model.*``
keys on disk — including the api_mode clear and the route-changed context_length clear.
"""

from __future__ import annotations

import asyncio

import pytest
import yaml

from hermes_cli.model_switch import ModelSwitchResult

_SEED = (
    "model:\n"
    "  default: local-model\n"
    "  provider: custom\n"
    "  base_url: http://localhost:1234/v1\n"
    "  api_mode: anthropic_messages\n"
    "  api_key: sk-stale\n"
    "  context_length: 32000\n"
    "  model_slots:\n"
    "    fast: gpt-5-mini\n"
    "agent:\n"
    "  system_prompt: keepme\n"
)

_RESULT = ModelSwitchResult(
    success=True, new_model="claude-haiku", target_provider="anthropic", provider_changed=True,
    api_key="sk-new", base_url="", api_mode="", is_global=True,
)


@pytest.fixture
def seeded_home(tmp_path, monkeypatch):
    import cli
    (tmp_path / "config.yaml").write_text(_SEED, encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(cli, "_hermes_home", tmp_path)
    return tmp_path


def _model_block(home) -> dict:
    return yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8"))["model"]


def _via_cli(home):
    from hermes_cli import cli_model_switch_mixin as mixin
    stub = type("Stub", (), {
        "agent": None, "model": "local-model", "_pending_one_turn_model_restore": None,
        "_stage_and_swap_model": lambda self, r, o: True})()
    mixin._commit_model_switch(stub, _RESULT, persist_global=True)


def _via_gateway(home):
    from gateway.slash_commands_model import _persist_model_switch_to_config
    asyncio.run(_persist_model_switch_to_config(_RESULT, home / "config.yaml"))


def _via_tui(home):
    from tui_gateway import server
    session = {"agent": None}
    server._apply_model_switch(
        "sid", session, "claude-haiku --provider anthropic --global", confirm_expensive_model=True)


def _via_dashboard(home):
    from hermes_cli.web_server_config import _apply_model_assignment_sync
    _apply_model_assignment_sync("main", "anthropic", "claude-haiku", "", "")


@pytest.mark.parametrize("surface", [_via_cli, _via_gateway, _via_tui, _via_dashboard],
                         ids=["cli", "gateway", "tui", "dashboard"])
def test_every_persist_surface_writes_the_same_model_block(seeded_home, monkeypatch, surface):
    monkeypatch.setattr("hermes_cli.model_switch.switch_model", lambda **_kw: _RESULT)
    monkeypatch.setattr("cli.HermesCLI._persist_model_switch_to_session", lambda *a, **k: None)
    monkeypatch.setattr("hermes_cli.cli_model_switch_mixin._print_switch_summary", lambda *a, **k: None)
    monkeypatch.setattr("hermes_cli.model_selection_guards.combined_selection_warning",
                        lambda *a, **k: None, raising=False)
    monkeypatch.setattr("cli._cprint", lambda *a, **k: None, raising=False)

    surface(seeded_home)

    block = _model_block(seeded_home)
    assert (block["default"], block["provider"]) == ("claude-haiku", "anthropic")
    # The target route has no endpoint/wire override: the OLD custom ones must not linger.
    assert not block.get("base_url") and not block.get("api_mode")
    # Route identity changed → the pinned window belongs to the old route. Cleared keys are
    # REMOVED, not left as ``key: null`` litter.
    assert "context_length" not in block and "base_url" not in block and "api_mode" not in block
    # Non-custom providers never read an inline key; the stale secret is gone.
    assert "api_key" not in block
    # Sibling keys under ``model:`` survive (targeted writes, no block rewrite).
    assert block["model_slots"] == {"fast": "gpt-5-mini"}
    assert yaml.safe_load((seeded_home / "config.yaml").read_text())["agent"]["system_prompt"] == "keepme"


def test_same_route_repick_keeps_the_context_pin(seeded_home):
    """A model re-pick on the SAME route keeps ``context_length`` (only the owner change drops it)."""
    from hermes_cli.model_switch import persist_model_selection
    same_route = ModelSwitchResult(
        success=True, new_model="local-model", target_provider="custom",
        base_url="http://localhost:1234/v1", api_mode="anthropic_messages", is_global=True)
    persist_model_selection(same_route)
    block = _model_block(seeded_home)
    assert block["context_length"] == 32000
    assert block["api_key"] == "sk-stale"  # custom targets keep their inline key


def test_custom_to_other_custom_endpoint_drops_the_inline_key(seeded_home):
    """``custom`` -> ``custom:other-box``: endpoint A's inline ``api_key`` must not become endpoint B's
    credential (the pointer-not-secret rule, #88990). Same provider *string* but a different
    ``base_url`` drops it too — the key belongs to one endpoint, not to the word "custom"."""
    from hermes_cli.model_switch import persist_model_selection
    other_provider = ModelSwitchResult(
        success=True, new_model="other-model", target_provider="custom:other-box",
        base_url="http://other-box:8000/v1", api_mode="chat_completions", is_global=True)
    persist_model_selection(other_provider)
    assert "api_key" not in _model_block(seeded_home)

    (seeded_home / "config.yaml").write_text(_SEED, encoding="utf-8")
    same_provider_other_host = ModelSwitchResult(
        success=True, new_model="local-model", target_provider="custom",
        base_url="http://10.0.0.9:1234/v1", api_mode="anthropic_messages", is_global=True)
    persist_model_selection(same_provider_other_host)
    assert "api_key" not in _model_block(seeded_home)


def test_gateway_persists_to_the_profile_config_it_was_given(tmp_path, monkeypatch):
    """Multiplexed gateway: the write lands in the routed profile's config.yaml, never the
    process-level HERMES_HOME."""
    from gateway.slash_commands_model import _persist_model_switch_to_config
    process_home, profile_home = tmp_path / "default", tmp_path / "profiles" / "named"
    for home in (process_home, profile_home):
        home.mkdir(parents=True)
        (home / "config.yaml").write_text("model:\n  default: old\n  provider: openai-codex\n")
    monkeypatch.setenv("HERMES_HOME", str(process_home))
    asyncio.run(_persist_model_switch_to_config(_RESULT, profile_home / "config.yaml"))
    assert _model_block(profile_home)["default"] == "claude-haiku"
    assert _model_block(process_home)["default"] == "old"
