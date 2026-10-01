"""Regression tests for #25106: CLI `/model <name> --global` never persisted
``model.base_url``/``model.api_mode`` to config.yaml, so a global provider
switch left the PREVIOUS provider's endpoint/wire-protocol on disk. The next
`hermes` launch re-read the stale base_url and routed the new model at the
old host.

Both ``_handle_model_switch`` (typed ``/model <name>``) and
``_apply_model_switch_result`` (interactive picker) shared the same gap: the
persistence block wrote ``model.default``/``model.provider`` but never
touched ``base_url``/``api_mode`` at all. Fix: sync both on every global
switch, clearing to ``None`` when the resolved result doesn't need them — now the
canonical ``hermes_cli.model_switch.persist_model_selection`` shape shared by every
surface.
"""

import hermes_yaml as yaml
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
    def _stage_and_swap_model(self, result, old_model):
        # Staging + in-place swap lives in a helper; run the real one on this stub.
        import cli as _cli_mod
        return _cli_mod.HermesCLI._stage_and_swap_model(self, result, old_model)


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

    def _confirm_and_apply_cli_model_switch(
        self, result, persist_global, one_turn, custom_provs=None
    ):
        import cli as cli_mod

        return cli_mod.HermesCLI._confirm_and_apply_cli_model_switch(
            self, result, persist_global, one_turn, custom_provs
        )

    def _open_model_picker(self, *a, **k):
        raise AssertionError("picker should not open when a model name is given")


def _run_switch(monkeypatch, result, cmd="/model MiniMax-M3 --global"):
    import cli as cli_mod

    monkeypatch.setattr(cli_mod, "_cprint", lambda *a, **k: None)
    monkeypatch.setattr("hermes_cli.model_switch.switch_model", lambda **kw: result)
    monkeypatch.setattr(
        "hermes_cli.inventory.load_picker_context",
        lambda: (_ for _ in ()).throw(RuntimeError("no picker context in test")),
    )
    cli_mod.HermesCLI._handle_model_switch(_StubCLI(), cmd)


@pytest.mark.parametrize("picker", [False, True])
def test_global_switch_persists_base_url_and_api_mode(tmp_path, monkeypatch, picker):
    """The core #25106 fix: a --global switch to a new provider/endpoint must
    write the newly resolved base_url and api_mode, not just default/provider."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    path = tmp_path / "config.yaml"
    before = {"model": {"default": "old-model", "provider": "copilot", "base_url": "https://old.example/v1"},
              "marker": "keep", "providers": {"fixture": {"api_key": "${FIXTURE_KEY}"}}}
    path.write_text(yaml.safe_dump(before))
    if picker:
        _run_apply(monkeypatch, _make_result())
    else:
        _run_switch(monkeypatch, _make_result())
    saved = yaml.safe_load(path.read_text())
    assert saved["model"]["default"] == "MiniMax-M3"
    assert saved["model"]["provider"] == "custom:minimax"
    assert saved["model"]["base_url"] == "https://api.minimax.io/v1"
    assert saved["model"]["api_mode"] == "chat_completions"
    assert saved["marker"] == before["marker"]
    assert saved["providers"] == before["providers"]


def test_session_only_switch_does_not_touch_config(tmp_path, monkeypatch):
    """--session must not write config.yaml at all — persistence stays
    entirely in-memory."""
    import cli as cli_mod

    monkeypatch.setattr(cli_mod, "_cprint", lambda *a, **k: None)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    path = tmp_path / "config.yaml"
    before = "# preserved exactly\nmodel: old-model\n"
    path.write_text(before)
    monkeypatch.setattr("hermes_cli.model_switch.switch_model", lambda **kw: _make_result())
    monkeypatch.setattr(
        "hermes_cli.inventory.load_picker_context",
        lambda: (_ for _ in ()).throw(RuntimeError("no picker context in test")),
    )

    cli_mod.HermesCLI._handle_model_switch(_StubCLI(), "/model MiniMax-M3 --session")

    assert path.read_text() == before


def _run_apply(monkeypatch, result, persist_global=True):
    """Drives `_apply_model_switch_result` directly — the interactive-picker
    sibling of `_handle_model_switch`. Unlike the tests above, and unlike
    `test_apply_model_switch_result_context.py` (which only ever calls this
    method with `persist_global=False`), this exercises the `persist_global=True`
    branch that actually writes to config.yaml."""
    import cli as cli_mod

    monkeypatch.setattr(cli_mod, "_cprint", lambda *a, **k: None)
    cli_mod.HermesCLI._apply_model_switch_result(_StubCLI(), result, persist_global)




