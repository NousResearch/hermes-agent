import queue
import threading
import time
from unittest.mock import patch

import cli as cli_module
import tools.skills_tool as skills_tool_module
from cli import HermesCLI
from hermes_cli.callbacks import prompt_for_secret
from tools.skills_tool import set_secret_capture_callback


class _FakeBuffer:
    def __init__(self):
        self.reset_called = False

    def reset(self):
        self.reset_called = True


class _FakeApp:
    def __init__(self):
        self.invalidated = False
        self.current_buffer = _FakeBuffer()

    def invalidate(self):
        self.invalidated = True


def _make_cli_stub(with_app=False):
    cli = HermesCLI.__new__(HermesCLI)
    cli._app = _FakeApp() if with_app else None
    cli._last_invalidate = 0.0
    cli._secret_state = None
    cli._secret_deadline = 0
    return cli


def test_secret_capture_callback_can_be_completed_from_cli_state_machine():
    cli = _make_cli_stub(with_app=True)
    results = []

    with patch("hermes_cli.callbacks.save_env_value_secure") as save_secret:
        save_secret.return_value = {
            "success": True,
            "stored_as": "TENOR_API_KEY",
            "validated": False,
        }

        thread = threading.Thread(
            target=lambda: results.append(
                cli._secret_capture_callback("TENOR_API_KEY", "Tenor API key")
            )
        )
        thread.start()

        deadline = time.time() + 2
        while cli._secret_state is None and time.time() < deadline:
            time.sleep(0.01)

        assert cli._secret_state is not None
        cli._submit_secret_response("super-secret-value")
        thread.join(timeout=2)

    assert results[0]["success"] is True
    assert results[0]["stored_as"] == "TENOR_API_KEY"
    assert results[0]["skipped"] is False


def test_cancel_secret_capture_marks_setup_skipped():
    cli = _make_cli_stub()
    cli._secret_state = {
        "response_queue": queue.Queue(),
        "var_name": "TENOR_API_KEY",
        "prompt": "Tenor API key",
        "metadata": {},
    }
    cli._secret_deadline = 123

    cli._cancel_secret_capture()

    assert cli._secret_state is None
    assert cli._secret_deadline == 0


def test_secret_capture_uses_getpass_without_tui():
    cli = _make_cli_stub()

    with patch("hermes_cli.callbacks.getpass.getpass", return_value="secret-value"), patch(
        "hermes_cli.callbacks.save_env_value_secure"
    ) as save_secret:
        save_secret.return_value = {
            "success": True,
            "stored_as": "TENOR_API_KEY",
            "validated": False,
        }
        result = prompt_for_secret(cli, "TENOR_API_KEY", "Tenor API key")

    assert result["success"] is True
    assert result["stored_as"] == "TENOR_API_KEY"
    assert result["skipped"] is False


def test_secret_capture_timeout_clears_hidden_input_buffer():
    cli = _make_cli_stub(with_app=True)
    cleared = {"value": False}

    def clear_buffer():
        cleared["value"] = True

    cli._clear_secret_input_buffer = clear_buffer

    with patch("hermes_cli.callbacks.queue.Queue.get", side_effect=queue.Empty), patch(
        "hermes_cli.callbacks._time.monotonic",
        side_effect=[0, 121],
    ):
        result = prompt_for_secret(cli, "TENOR_API_KEY", "Tenor API key")

    assert result["success"] is True
    assert result["skipped"] is True
    assert result["reason"] == "timeout"
    assert cleared["value"] is True


def test_cli_chat_registers_secret_capture_callback():
    clean_config = {
        "model": {
            "default": "anthropic/claude-opus-4.6",
            "base_url": "https://openrouter.ai/api/v1",
            "provider": "auto",
        },
        "display": {"compact": False, "tool_progress": "all"},
        "agent": {},
        "terminal": {"env_type": "local"},
    }

    with patch("cli.get_tool_definitions", return_value=[]), patch.dict(
        "os.environ", {"LLM_MODEL": "", "HERMES_MAX_ITERATIONS": ""}, clear=False
    ), patch.dict(cli_module.__dict__, {"CLI_CONFIG": clean_config}):
        cli_obj = HermesCLI()
        with patch.object(cli_obj, "_ensure_runtime_credentials", return_value=False):
            cli_obj.chat("hello")

    try:
        assert skills_tool_module._secret_capture_callback == cli_obj._secret_capture_callback
    finally:
        set_secret_capture_callback(None)
