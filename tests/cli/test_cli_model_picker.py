"""Tests for the canonical /model selector in the CLI."""

from __future__ import annotations

import queue
from unittest.mock import MagicMock

from prompt_toolkit.completion import Completion

from hermes_cli.model_switch import ModelSwitchResult
from hermes_cli.model_selection import ModelSwitchRequest


def _make_cli():
    from cli import HermesCLI

    cli_obj = HermesCLI.__new__(HermesCLI)
    cli_obj.model = "anthropic/claude-sonnet-4.6"
    cli_obj.provider = "openrouter"
    cli_obj.base_url = "https://openrouter.ai/api/v1"
    cli_obj.api_key = "sk-test"
    cli_obj.api_mode = "chat_completions"
    cli_obj.requested_provider = "openrouter"
    cli_obj._explicit_api_key = ""
    cli_obj._explicit_base_url = ""
    cli_obj._pending_input = queue.Queue()
    cli_obj.agent = None
    cli_obj._model_selection_controller = None
    return cli_obj


def test_handle_model_switch_no_args_shows_inline_menu_hint(monkeypatch):
    cli_obj = _make_cli()
    printed = []
    monkeypatch.setattr("cli._cprint", lambda msg: printed.append(str(msg)))

    cli_obj._handle_model_switch("/model")

    assert any("inline slash menu" in line for line in printed)


class _FakeCompletionState:
    def __init__(self, completion):
        self.current_completion = completion


class _FakeBuffer:
    def __init__(self, text: str, completion: Completion):
        self.text = text
        self.complete_state = _FakeCompletionState(completion)
        self.reset_called = False
        self.started_completion = False
        self.document_text = text

    def go_to_completion(self, index: int) -> None:
        return None

    def apply_completion(self, completion: Completion) -> None:
        self.text = "/model"

    def reset(self, append_to_history: bool = False) -> None:
        self.text = ""
        self.reset_called = append_to_history

    def set_document(self, document, bypass_readonly: bool = False) -> None:
        self.document_text = document.text
        self.text = document.text

    def start_completion(self, select_first: bool = False) -> None:
        self.started_completion = select_first


def test_maybe_accept_slash_completion_on_enter_opens_selector_in_same_surface():
    cli_obj = _make_cli()
    completion = Completion("model", start_position=-3, display="/model")
    buffer = _FakeBuffer("/mo", completion)
    opened = {"called": False, "buffer_text": None}
    cli_obj._open_model_selection = lambda buf=None: (
        opened.__setitem__("called", True),
        opened.__setitem__("buffer_text", getattr(buf, "text", None)),
    )

    handled = cli_obj._maybe_accept_slash_completion_on_enter(buffer)

    assert handled is True
    assert buffer.reset_called is False
    assert opened["called"] is True
    assert opened["buffer_text"] == "/mo"


def test_space_on_exact_model_opens_selector_in_same_surface():
    cli_obj = _make_cli()
    buffer = _FakeBuffer("/model", Completion("model", start_position=0, display="/model"))
    opened = {"called": False, "buffer_text": None}
    cli_obj._open_model_selection = lambda buf=None: (
        opened.__setitem__("called", True),
        opened.__setitem__("buffer_text", getattr(buf, "text", None)),
    )

    handled = cli_obj._maybe_accept_model_selection_on_space(buffer)

    assert handled is True
    assert opened["called"] is True
    assert opened["buffer_text"] == "/model"


class _Controller:
    def __init__(self, request):
        self._request = request
        self.moved = []
        self.back_calls = 0
        self.source_id = "oauth"
        self.provider_id = "oauth:openai"

    def current_view(self):
        class _View:
            level = "model"
            breadcrumb = "OAuth / OpenAI"
            items = ()
        return _View()

    def move_up(self):
        self.moved.append("up")

    def move_down(self):
        self.moved.append("down")

    def back(self):
        self.back_calls += 1
        return self.back_calls == 1

    def enter(self):
        return self._request


def test_activate_model_selection_executes_final_switch(monkeypatch):
    cli_obj = _make_cli()
    printed = []
    captured = {}
    monkeypatch.setattr("cli._cprint", lambda msg: printed.append(str(msg)))
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {})
    cli_obj._model_selection_controller = _Controller(
        ModelSwitchRequest(provider_slug="openai-codex", model_id="gpt-5.4")
    )

    def _fake_switch_model(**kwargs):
        captured.update(kwargs)
        return ModelSwitchResult(
            success=True,
            new_model="gpt-5.4",
            target_provider="openai-codex",
            provider_changed=True,
            api_key="oauth-token",
            base_url="https://chatgpt.com/backend-api/codex",
            api_mode="codex_responses",
            provider_label="OpenAI Codex",
        )

    monkeypatch.setattr("hermes_cli.model_switch.switch_model", _fake_switch_model)
    buffer = _FakeBuffer("/model", Completion("dummy", start_position=0))
    handled = cli_obj._activate_model_selection(buffer)

    assert handled is True
    assert cli_obj._model_selection_controller is None
    assert buffer.reset_called is True
    assert cli_obj.model == "gpt-5.4"
    assert cli_obj.provider == "openai-codex"
    assert captured.get("skip_validation", False) is False
    assert any("Model switched" in line for line in printed)


def test_space_advances_active_model_selection(monkeypatch):
    cli_obj = _make_cli()
    cli_obj._model_selection_controller = _Controller(None)
    buffer = _FakeBuffer("/model ", Completion("dummy", start_position=0))

    called = {"activated": False}
    cli_obj._activate_model_selection = lambda buf=None: called.__setitem__("activated", True) or True

    handled = cli_obj._maybe_accept_model_selection_on_space(buffer)

    assert handled is True
    assert called["activated"] is True


def test_commit_model_selection_forwards_buffer():
    cli_obj = _make_cli()
    buffer = _FakeBuffer("/model ", Completion("dummy", start_position=0))
    captured = {"buffer": None}
    cli_obj._activate_model_selection = lambda buf=None: captured.__setitem__("buffer", buf) or True

    handled = cli_obj._commit_model_selection(buffer)

    assert handled is True
    assert captured["buffer"] is buffer


def test_move_and_back_model_selection():
    cli_obj = _make_cli()
    controller = _Controller(None)
    cli_obj._model_selection_controller = controller

    buffer = _FakeBuffer("/model", Completion("dummy", start_position=0))
    assert cli_obj._move_model_selection(-1, buffer) is True
    assert cli_obj._move_model_selection(1, buffer) is True
    assert controller.moved == ["up", "down"]

    assert cli_obj._back_model_selection(buffer) is True
    assert cli_obj._model_selection_controller is controller
    assert cli_obj._back_model_selection(buffer) is True
    assert cli_obj._model_selection_controller is None


def test_sync_model_selection_buffer_writes_inline_path():
    cli_obj = _make_cli()
    from hermes_cli.model_selection import ModelSelectionController, build_model_selection_tree

    tree = build_model_selection_tree(current_provider="openrouter", current_model="openai/gpt-5.4")
    controller = ModelSelectionController(tree)
    cli_obj._model_selection_controller = controller
    buffer = _FakeBuffer("/model", Completion("dummy", start_position=0))

    cli_obj._sync_model_selection_buffer(buffer)
    assert buffer.text == "/model "

    controller.enter()  # openrouter
    cli_obj._sync_model_selection_buffer(buffer)
    assert buffer.text == "/model openrouter "

    controller.enter()  # default provider under openrouter
    cli_obj._sync_model_selection_buffer(buffer)
    assert buffer.text.startswith("/model openrouter ")
