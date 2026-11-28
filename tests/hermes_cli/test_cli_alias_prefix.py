"""Classic CLI ``+<alias> rest``: the turn's prompt is ``rest`` and the alias runs for that turn only."""

from types import SimpleNamespace

from hermes_cli.model_switch import ModelSwitchResult


class _FakeAgent:
    model = "old/model"
    provider = "openrouter"

    def switch_model(self, **kwargs):
        self.model, self.provider = kwargs["new_model"], kwargs["new_provider"]


def _stub(prefix_enabled):
    import cli as cli_mod

    stub = SimpleNamespace(
        model="old/model", provider="openrouter", requested_provider="openrouter", api_key="sk-old",
        _explicit_api_key="sk-old", base_url="https://openrouter.ai/api/v1",
        _explicit_base_url="https://openrouter.ai/api/v1", api_mode="chat_completions",
        agent=_FakeAgent(), _pending_model_switch_note=None, _pending_one_turn_model_restore=None,
        config={"display": {"model_alias_prefix": prefix_enabled},
                "model_aliases": {"fast": {"model": "fast/model", "provider": "openrouter"}}},
        _confirm_expensive_model_switch=lambda result: True)
    for name in ("_alias_prefix_for", "_take_alias_prefix", "_handle_model_switch",
                 "_confirm_and_apply_cli_model_switch", "_stage_and_swap_model", "_snapshot_model_runtime",
                 "_expand_paste_references"):
        setattr(stub, name, getattr(cli_mod.HermesCLI, name).__get__(stub))
    return stub


def _no_io(monkeypatch):
    import cli as cli_mod
    monkeypatch.setattr(cli_mod, "_cprint", lambda *a, **k: None)
    monkeypatch.setattr("hermes_cli.inventory.load_picker_context", lambda: SimpleNamespace(
        with_overrides=lambda **_: SimpleNamespace(user_providers=None, custom_providers=None)))
    monkeypatch.setattr("hermes_cli.model_switch.switch_model", lambda **kw: ModelSwitchResult(
        success=True, new_model="fast/model", target_provider="openrouter",
        base_url="https://openrouter.ai/api/v1", api_mode="chat_completions", provider_label="OpenRouter"))
    monkeypatch.setattr("hermes_cli.model_switch.resolve_display_context_length", lambda *a, **k: None)


def test_alias_prefix_switches_for_one_turn_and_sends_the_rest(monkeypatch):
    _no_io(monkeypatch)
    stub = _stub(prefix_enabled=True)

    assert stub._take_alias_prefix("+fast hi there") == "hi there"
    assert stub.model == "fast/model"
    assert stub._pending_one_turn_model_restore["model"] == "old/model"  # chat() restores after the turn


def test_other_plus_text_and_the_default_leave_the_message_alone(monkeypatch):
    _no_io(monkeypatch)
    enabled, disabled = _stub(prefix_enabled=True), _stub(prefix_enabled=False)

    assert enabled._take_alias_prefix("+1 agreed") == "+1 agreed"
    assert enabled._take_alias_prefix("+unknownword hi") == "+unknownword hi"
    assert disabled._take_alias_prefix("+fast hi there") == "+fast hi there"
    assert (enabled.model, disabled.model) == ("old/model", "old/model")
    assert enabled._pending_one_turn_model_restore is None


def test_a_collapsed_paste_is_classified_by_what_was_pasted(monkeypatch, tmp_path):
    _no_io(monkeypatch)
    stub = _stub(prefix_enabled=True)
    pasted = tmp_path / "paste.txt"
    pasted.write_text("+fast summarize this long log", encoding="utf-8")

    assert stub._take_alias_prefix(f"[Pasted text #1: 30 lines \u2192 {pasted}]") == "summarize this long log"
    assert stub.model == "fast/model"
