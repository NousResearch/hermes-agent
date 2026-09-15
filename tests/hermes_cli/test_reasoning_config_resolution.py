"""reasoning_config must be re-resolved per model at the /model and /new boundaries.

Regression coverage for the staleness bug a /model switch had: the CLI wrapper's
``reasoning_config`` was only ever computed at CLI startup (and, incompletely, at
``/new`` from the raw global ``agent.reasoning_effort``), so a switch kept the PREVIOUS
model's reasoning config attached - which hard-fails (HTTP 400) against a target model
that has a ``none`` override in ``agent.reasoning_overrides`` (e.g. a local Ollama model
that rejects any thinking/reasoning parameter).

Both tests fail against the pre-fix code: neither boundary called
``hermes_constants.resolve_reasoning_config`` at all.
"""

from __future__ import annotations

from types import SimpleNamespace

from hermes_cli.model_switch import ModelSwitchResult


class _FakeAgent:
    def __init__(self):
        self.calls: list[dict] = []
        self.model = "old/model"
        self.provider = "openrouter"
        self.reasoning_config = {"enabled": True, "effort": "high"}

    def switch_model(self, **kwargs):
        self.calls.append(kwargs)
        self.model = kwargs["new_model"]
        self.provider = kwargs["new_provider"]


class _StubCLI:
    """Minimal stand-in driving the real HermesCLI model-switch methods.

    Same pattern as ``tests/hermes_cli/test_cli_model_once.py``.
    """

    model = "old/model"
    provider = "openrouter"
    requested_provider = "openrouter"
    api_key = "sk-old"
    _explicit_api_key = "sk-old"
    base_url = "https://openrouter.ai/api/v1"
    _explicit_base_url = "https://openrouter.ai/api/v1"
    api_mode = "chat_completions"
    agent = None
    # Stale: this is the previous model's resolved value.
    reasoning_config = {"enabled": True, "effort": "high"}
    _pending_model_switch_note = None
    _pending_one_turn_model_restore = None

    def _stage_and_swap_model(self, result, old_model):
        import cli as cli_mod

        return cli_mod.HermesCLI._stage_and_swap_model(self, result, old_model)

    def _confirm_expensive_model_switch(self, result):
        return True

    def _confirm_and_apply_cli_model_switch(
        self, result, persist_global, one_turn, custom_provs=None
    ):
        import cli as cli_mod

        return cli_mod.HermesCLI._confirm_and_apply_cli_model_switch(
            self, result, persist_global, one_turn, custom_provs
        )


def _record_resolve(monkeypatch, result_value):
    """Spy on the shared chokepoint; returns the dict it records the call into."""
    import hermes_constants

    seen: dict = {}

    def _fake_resolve(cfg, model=""):
        seen["model"] = model
        seen["cfg"] = cfg
        return result_value

    monkeypatch.setattr(hermes_constants, "resolve_reasoning_config", _fake_resolve)
    return seen


def _patch_switch_plumbing(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.inventory.load_picker_context",
        lambda: SimpleNamespace(
            user_providers=None,
            custom_providers=None,
            with_overrides=lambda **_: SimpleNamespace(
                user_providers=None, custom_providers=None
            ),
        ),
    )
    monkeypatch.setattr(
        "hermes_cli.model_switch.resolve_display_context_length", lambda *a, **k: None
    )


def test_model_switch_re_resolves_reasoning_config_for_the_new_model(monkeypatch):
    import cli as cli_mod

    seen = _record_resolve(monkeypatch, {"enabled": False})
    monkeypatch.setattr(cli_mod, "_cprint", lambda *a, **k: None)
    _patch_switch_plumbing(monkeypatch)
    monkeypatch.setattr(
        "hermes_cli.model_switch.switch_model",
        lambda **_: ModelSwitchResult(
            success=True,
            new_model="local/nothink",
            target_provider="ollama",
            api_key="",
            base_url="http://127.0.0.1:11434/v1",
            api_mode="chat_completions",
            provider_label="Ollama",
        ),
    )

    stub = _StubCLI()
    stub.agent = _FakeAgent()

    cli_mod.HermesCLI._handle_model_switch(stub, "/model local/nothink --provider ollama")

    assert stub.model == "local/nothink"
    # Resolved for the model that is now active, not the one that was active before ...
    assert seen.get("model") == "local/nothink"
    # ... and the stale previous-model value is gone (this is what 400s against a
    # target model whose override is `none`).
    assert stub.reasoning_config == {"enabled": False}


def test_new_session_resolves_reasoning_config_after_the_model_reset(monkeypatch, tmp_path):
    from hermes_cli import cli_session_mixin as session_mixin
    from tests.hermes_cli.test_cli_new_session import _prepare_cli_with_active_session

    cli = _prepare_cli_with_active_session(tmp_path)
    # Stale: the previous session's model, resolved the old way (raw global effort).
    cli.reasoning_config = {"enabled": True, "effort": "high"}
    cli.model = "stale/previous-model"

    # Deterministic stand-in for the config-default model reset: /new must resolve the
    # reasoning config for the model that is active AFTER this swap, not before it.
    monkeypatch.setattr(
        session_mixin,
        "_reset_model_to_config_default",
        lambda c, silent: setattr(c, "model", "reset/config-default"),
    )
    seen = _record_resolve(monkeypatch, {"enabled": False, "effort": "none"})

    cli.new_session(silent=True)

    assert seen.get("model") == "reset/config-default"
    assert cli.reasoning_config == {"enabled": False, "effort": "none"}
