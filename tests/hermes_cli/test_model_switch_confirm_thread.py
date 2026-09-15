"""The /model <name> --provider <p> path must run the expensive-model
confirm + apply sequence off the main thread.

The confirm modal blocks its calling thread on a response queue (see
``_prompt_text_input_modal``). When invoked inline from the prompt_toolkit
main thread the TUI event loop freezes, the modal never renders, and the
switch silently cancels after the 120s timeout — the user sees a frozen
terminal and "Model switch cancelled." without ever seeing the warning.

The picker path (_handle_model_picker_selection) already dispatches
confirm+apply on a worker thread; this regression test pins the command
path to the same contract: with ``_app`` present the modal runs off the
main thread, and without ``_app`` (tests / non-interactive) it stays
synchronous.
"""

import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from hermes_cli.model_switch import ModelSwitchResult


class _FakeAgent:
    def __init__(self):
        self.calls = []
        self.model = "old/model"
        self.provider = "openrouter"

    def switch_model(self, **kwargs):
        self.calls.append(kwargs)
        self.model = kwargs["new_model"]
        self.provider = kwargs["new_provider"]


class _StubCLI:
    def _stage_and_swap_model(self, result, old_model):
        # Staging + in-place swap lives in a helper; run the real one on this stub.
        import cli as _cli_mod
        return _cli_mod.HermesCLI._stage_and_swap_model(self, result, old_model)

    model = "old/model"
    provider = "openrouter"
    requested_provider = "openrouter"
    api_key = "sk-old"
    _explicit_api_key = "sk-old"
    _explicit_base_url = ""
    base_url = "https://openrouter.ai/api/v1"
    api_mode = "chat_completions"
    conversation_history = []
    agent = None
    _pending_model_switch_note = None
    _pending_one_turn_model_restore = None
    _app = None

    def _confirm_expensive_model_switch(self, result):
        return True

    def _confirm_and_apply_cli_model_switch(
        self, result, persist_global, one_turn, custom_provs=None
    ):
        import cli as cli_mod

        outcome = cli_mod.HermesCLI._confirm_and_apply_cli_model_switch(
            self, result, persist_global, one_turn, custom_provs
        )
        if hasattr(self, "apply_done"):
            self.apply_done.set()
        return outcome

    def _snapshot_model_runtime(self):
        from cli import HermesCLI
        return HermesCLI._snapshot_model_runtime(self)

    def _apply_model_switch_result(self, result, persist_global, custom_providers=None):
        from cli import HermesCLI
        return HermesCLI._apply_model_switch_result(
            self, result, persist_global, custom_providers=custom_providers
        )


def _make_result():
    return ModelSwitchResult(
        success=True,
        new_model="claude-sonnet-4.6",
        target_provider="anthropic",
        api_key="sk-ant",
        base_url="https://api.anthropic.com",
        api_mode="anthropic_messages",
        provider_label="Anthropic",
    )


def _patch_deps(monkeypatch, printed):
    import cli as cli_mod

    monkeypatch.setattr(cli_mod, "_cprint", lambda s, *a, **k: printed.append(str(s)))
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
        "hermes_cli.model_switch.switch_model", lambda **_: _make_result()
    )
    monkeypatch.setattr(
        "hermes_cli.model_switch.resolve_display_context_length", lambda *a, **k: None
    )
    return cli_mod


@pytest.fixture(autouse=True)
def _offline_storage(monkeypatch):
    config = {"model": {"default": "old/model", "provider": "openrouter"}}
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: config)
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: config)
    monkeypatch.setattr("cli.save_config_value", Mock())
    monkeypatch.setattr("cli.HermesCLI._persist_model_switch_to_session", Mock())
    monkeypatch.setattr("hermes_cli.context_switch_guard.merge_preflight_compression_warning", lambda *a, **k: None)
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})


def test_confirm_runs_off_main_thread_when_tui_present(monkeypatch):
    """With ``_app`` set, the confirm modal must not block the caller's
    thread: the switch is dispatched on a worker thread and the command
    handler returns immediately."""
    import cli as cli_mod

    stub = _StubCLI()
    stub.agent = _FakeAgent()
    stub._app = SimpleNamespace(loop=None)
    stub.apply_done = threading.Event()
    printed = []
    cli_mod = _patch_deps(monkeypatch, printed)

    called_on = {}
    ready = threading.Event()

    def _confirm(self, result):
        called_on["thread_id"] = threading.get_ident()
        called_on["is_main"] = threading.current_thread() is threading.main_thread()
        ready.set()
        return True

    # The stub's own confirm would shadow the patched one — bind the
    # recorder onto the instance so the worker thread hits it.
    monkeypatch.setattr(stub, "_confirm_expensive_model_switch", _confirm.__get__(stub))

    cli_mod.HermesCLI._handle_model_switch(
        stub, "/model claude-sonnet-4.6 --provider anthropic"
    )

    # The worker thread performs the confirm and completes the apply.
    assert ready.wait(timeout=10)
    assert stub.apply_done.wait(timeout=10)
    assert called_on["is_main"] is False

    # Apply still lands on CLI + agent state.
    assert stub.model == "claude-sonnet-4.6"
    assert stub.provider == "anthropic"
    assert stub.agent.calls[-1]["new_model"] == "claude-sonnet-4.6"


def test_confirm_stays_synchronous_without_app(monkeypatch):
    """Without a TUI app (unit tests / non-interactive) the old inline
    behaviour is preserved: confirm + apply run on the calling thread."""
    import cli as cli_mod

    stub = _StubCLI()
    stub.agent = _FakeAgent()
    printed = []
    cli_mod = _patch_deps(monkeypatch, printed)

    called_on = {}

    def _confirm(self, result):
        called_on["is_main"] = threading.current_thread() is threading.main_thread()
        return True

    monkeypatch.setattr(stub, "_confirm_expensive_model_switch", _confirm.__get__(stub))

    cli_mod.HermesCLI._handle_model_switch(
        stub, "/model claude-sonnet-4.6 --provider anthropic"
    )

    assert called_on.get("is_main") is True
    assert stub.model == "claude-sonnet-4.6"
    assert stub.provider == "anthropic"
    assert not any("PROVIDER AUTOMATICALLY CHANGED" in line for line in printed)


_PROVIDER_WARNING = (
    "PROVIDER AUTOMATICALLY CHANGED: openrouter -> anthropic. "
    "No --provider argument was supplied. This change may incur additional costs."
)


@pytest.mark.parametrize("scope", ["--session", "--global", "--once", "picker"])
def test_provider_warning_is_bold_red_after_swap_before_success(monkeypatch, scope):
    stub = _StubCLI()
    stub.agent = _FakeAgent()
    printed = []
    cli_mod = _patch_deps(monkeypatch, printed)
    result = _make_result()
    result.provider_switch_warning = _PROVIDER_WARNING
    result.warning_message = "Ordinary validation warning."
    result.runtime_capabilities = {"native_compaction": True}
    monkeypatch.setattr("hermes_cli.model_switch.switch_model", lambda **k: result)
    rendered_after = []

    def record(text, *args, **kwargs):
        printed.append(text)
        if _PROVIDER_WARNING in text:
            rendered_after.append(stub.agent.model)

    monkeypatch.setattr(cli_mod, "_cprint", record)
    if scope == "picker":
        cli_mod.HermesCLI._confirm_and_apply_model_switch_result(stub, result, False)
    else:
        cli_mod.HermesCLI._handle_model_switch(stub, f"/model claude-sonnet-4.6 {scope}")

    warning_line = f"\033[1;31m  🚨 {_PROVIDER_WARNING}\033[0m"
    assert warning_line in printed
    success = next(line for line in printed if "✓ Model switched:" in line)
    assert printed.index(warning_line) < printed.index(success)
    assert rendered_after == [result.new_model]
    assert any(result.warning_message in line for line in printed)
    assert stub.agent.calls[-1]["capabilities"] == result.runtime_capabilities
    if scope == "--once":
        assert stub._pending_one_turn_model_restore["model"] == "old/model"
        cli_mod.HermesCLI._persist_model_switch_to_session.assert_not_called()
    else:
        cli_mod.HermesCLI._persist_model_switch_to_session.assert_called_once()
    assert cli_mod.save_config_value.called == (scope == "--global")


@pytest.mark.parametrize("picker", [False, True])
@pytest.mark.parametrize("failure", ["cancel", "swap"])
def test_unapplied_cli_switch_has_no_provider_warning(monkeypatch, picker, failure):
    stub = _StubCLI()
    stub.agent = _FakeAgent()
    printed = []
    cli_mod = _patch_deps(monkeypatch, printed)
    result = _make_result()
    result.provider_switch_warning = _PROVIDER_WARNING
    monkeypatch.setattr("hermes_cli.model_switch.switch_model", lambda **k: result)
    if failure == "cancel":
        stub._confirm_expensive_model_switch = lambda result: False
    else:
        stub.agent.switch_model = Mock(side_effect=RuntimeError("offline swap failure"))

    if picker:
        cli_mod.HermesCLI._confirm_and_apply_model_switch_result(stub, result, False)
    else:
        cli_mod.HermesCLI._handle_model_switch(stub, "/model claude-sonnet-4.6 --session")

    assert any("cancelled" in line or "failed" in line for line in printed)
    assert not any(_PROVIDER_WARNING in line or "✓ Model switched:" in line for line in printed)
    assert stub.model == stub.agent.model == "old/model"
    assert stub.provider == stub.agent.provider == "openrouter"
    assert stub._pending_model_switch_note is None
    cli_mod.save_config_value.assert_not_called()
    cli_mod.HermesCLI._persist_model_switch_to_session.assert_not_called()
