from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from cli import HermesCLI


class _FakeBuffer:
    def __init__(self, text="", cursor_position=None):
        self.text = text
        self.cursor_position = len(text) if cursor_position is None else cursor_position

    def reset(self, append_to_history=False):
        self.text = ""
        self.cursor_position = 0


def _make_cli_stub(initial_text="/model"):
    cli = HermesCLI.__new__(HermesCLI)
    cli._app = SimpleNamespace(invalidate=MagicMock(), current_buffer=_FakeBuffer(initial_text))
    cli._invalidate = MagicMock()
    cli._model_picker_state = None
    cli._modal_input_snapshot = None
    cli.model = "glm-5.1"
    cli.provider = "zai"
    cli.requested_provider = "zai"
    cli.api_key = ""
    cli.base_url = "https://api.z.ai/api/coding/paas/v4"
    cli.api_mode = "chat_completions"
    cli.agent = None
    return cli


def test_model_picker_does_not_restore_triggering_slash_command_after_selection():
    cli = _make_cli_stub("/model")
    providers = [{
        "slug": "zai",
        "name": "Z.AI",
        "models": ["glm-5.1"],
        "total_models": 1,
        "is_current": True,
    }]
    switch_result = SimpleNamespace(
        success=True,
        new_model="glm-5.1",
        target_provider="zai",
        provider_changed=False,
        api_key="",
        base_url="https://api.z.ai/api/coding/paas/v4",
        api_mode="chat_completions",
        error_message="",
        warning_message="",
        provider_label="Z.AI",
        model_info=None,
    )

    with patch("cli._cprint"), \
         patch("hermes_cli.model_switch.list_authenticated_providers", return_value=providers), \
         patch("hermes_cli.models.provider_model_ids", return_value=[]), \
         patch("hermes_cli.model_switch.switch_model", return_value=switch_result):
        cli._handle_model_switch("/model")
        assert cli._model_picker_state is not None
        assert cli._modal_input_snapshot is None
        assert cli._app.current_buffer.text == ""

        cli._handle_model_picker_selection()
        assert cli._model_picker_state["stage"] == "model"

        cli._handle_model_picker_selection()
        assert cli._model_picker_state is None
        assert cli._app.current_buffer.text == ""
