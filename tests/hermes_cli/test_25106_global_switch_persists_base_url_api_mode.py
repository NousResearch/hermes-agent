"""Regression tests for #25106: CLI `/model <name> --global` never persisted
``model.base_url``/``model.api_mode`` to config.yaml, so a global provider
switch left the PREVIOUS provider's endpoint/wire-protocol on disk. The next
`hermes` launch re-read the stale base_url and routed the new model at the
old host.

Both ``_handle_model_switch`` (typed ``/model <name>``) and
``_apply_model_switch_result`` (interactive picker) shared the same gap: the
persistence block wrote ``model.default``/``model.provider`` but never
touched ``base_url``/``api_mode`` at all. Fix: sync both on every global
switch, clearing to ``None`` when the resolved result doesn't need them —
mirroring the already-correct ``tui_gateway/server.py:_persist_model_switch``
pattern (fixed for #48305).
"""

from unittest.mock import MagicMock, patch

import pytest

from hermes_cli.model_switch import ModelSwitchResult


def _make_result(*, base_url="https://api.minimax.io/v1", api_mode="chat_completions", provider_changed=True):
    return ModelSwitchResult(
        success=True,
        new_model="MiniMax-M3",
        target_provider="custom:minimax",
        provider_changed=provider_changed,
        api_key="sk-minimax",
        base_url=base_url,
        api_mode=api_mode,
        warning_message="",
        provider_label="MiniMax (custom)",
        resolved_via_alias=False,
        capabilities=None,
        model_info=None,
        is_global=True,
    )


class _StubCLI:
    """Minimum attrs/methods `_handle_model_switch` reads or calls on self."""

    agent = None
    model = "old-model"
    provider = "copilot"
    requested_provider = "copilot"
    api_key = "sk-old"
    base_url = "https://api.githubcopilot.com"
    api_mode = "chat_completions"
    _explicit_api_key = ""
    _explicit_base_url = ""
    conversation_history = []
    _pending_model_switch_note = ""

    def _confirm_expensive_model_switch(self, result) -> bool:
        return True

    def _open_model_picker(self, *a, **k):
        raise AssertionError("picker should not open when a model name is given")


def _run_switch(monkeypatch, result, cmd="/model MiniMax-M3 --global"):
    import cli as cli_mod

    monkeypatch.setattr(cli_mod, "_cprint", lambda *a, **k: None)
    saved: dict[str, object] = {}

    def _fake_save(key, value):
        saved[key] = value

    monkeypatch.setattr(cli_mod, "save_config_value", _fake_save)
    monkeypatch.setattr("hermes_cli.model_switch.switch_model", lambda **kw: result)
    monkeypatch.setattr(
        "hermes_cli.inventory.load_picker_context",
        lambda: (_ for _ in ()).throw(RuntimeError("no picker context in test")),
    )
    cli_mod.HermesCLI._handle_model_switch(_StubCLI(), cmd)
    return saved


def test_global_switch_persists_base_url_and_api_mode(monkeypatch):
    """The core #25106 fix: a --global switch to a new provider/endpoint must
    write the newly resolved base_url and api_mode, not just default/provider."""
    saved = _run_switch(monkeypatch, _make_result())

    assert saved["model.default"] == "MiniMax-M3"
    assert saved["model.provider"] == "custom:minimax"
    assert saved["model.base_url"] == "https://api.minimax.io/v1"
    assert saved["model.api_mode"] == "chat_completions"


def test_global_switch_clears_base_url_and_api_mode_when_unresolved(monkeypatch):
    """When the resolver returns no base_url/api_mode for the new provider
    (e.g. a named provider needing neither), any previous value must be
    cleared (None) rather than silently left in config.yaml."""
    result = _make_result(base_url="", api_mode="")
    saved = _run_switch(monkeypatch, result)

    assert saved["model.base_url"] is None
    assert saved["model.api_mode"] is None


def test_session_only_switch_does_not_touch_config(monkeypatch):
    """--session must not call save_config_value at all — persistence stays
    entirely in-memory."""
    import cli as cli_mod

    monkeypatch.setattr(cli_mod, "_cprint", lambda *a, **k: None)
    save_calls = []
    monkeypatch.setattr(cli_mod, "save_config_value", lambda *a, **k: save_calls.append(a))
    monkeypatch.setattr("hermes_cli.model_switch.switch_model", lambda **kw: _make_result())
    monkeypatch.setattr(
        "hermes_cli.inventory.load_picker_context",
        lambda: (_ for _ in ()).throw(RuntimeError("no picker context in test")),
    )

    cli_mod.HermesCLI._handle_model_switch(_StubCLI(), "/model MiniMax-M3 --session")

    assert save_calls == []
