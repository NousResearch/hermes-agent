import threading
import time
from unittest.mock import patch

from cli import HermesCLI


class _FakeBuffer:
    def __init__(self, text="", cursor_position=None):
        self.text = text
        self.cursor_position = len(text) if cursor_position is None else cursor_position
        self.reset_called = False

    def reset(self):
        self.text = ""
        self.cursor_position = 0
        self.reset_called = True


class _FakeApp:
    def __init__(self, text="draft text"):
        self.current_buffer = _FakeBuffer(text=text)
        self.invalidated = 0

    def invalidate(self):
        self.invalidated += 1


class _FakeChatConsole:
    def __init__(self, sink):
        self._sink = sink

    def print(self, message):
        self._sink.append(message)


def _make_cli_stub(with_app=True):
    cli = HermesCLI.__new__(HermesCLI)
    cli._app = _FakeApp() if with_app else None
    cli._modal_input_snapshot = None
    cli._clarify_state = None
    cli._clarify_freetext = False
    cli._clarify_deadline = 0
    cli._last_invalidate = 0.0
    cli._invalidate = lambda *args, **kwargs: None
    return cli


def test_clarify_callback_restores_draft_and_prints_choice_to_scrollback():
    cli = _make_cli_stub(with_app=True)
    printed = []
    results = []

    with patch("cli.ChatConsole", side_effect=lambda: _FakeChatConsole(printed)), patch("builtins.print"):
        worker = threading.Thread(
            target=lambda: results.append(cli._clarify_callback("How should I proceed?", ["rebase", "merge"]))
        )
        worker.start()

        deadline = time.time() + 2
        while cli._clarify_state is None and time.time() < deadline:
            time.sleep(0.01)

        assert cli._clarify_state is not None
        assert cli._app.current_buffer.reset_called is True
        assert cli._app.current_buffer.text == ""

        cli._clarify_state["response_queue"].put("merge")
        worker.join(timeout=2)

    assert results == ["merge"]
    assert cli._app.current_buffer.text == "draft text"
    assert any("How should I proceed?" in line for line in printed)
    assert any("merge" in line for line in printed)


def test_clarify_callback_prints_open_ended_answer_to_scrollback():
    cli = _make_cli_stub(with_app=False)
    printed = []
    results = []

    with patch("cli.ChatConsole", side_effect=lambda: _FakeChatConsole(printed)), patch("builtins.print"):
        worker = threading.Thread(
            target=lambda: results.append(cli._clarify_callback("What changed?", []))
        )
        worker.start()

        deadline = time.time() + 2
        while cli._clarify_state is None and time.time() < deadline:
            time.sleep(0.01)

        assert cli._clarify_state is not None
        assert cli._clarify_freetext is True

        cli._clarify_state["response_queue"].put("I chose the first option")
        worker.join(timeout=2)

    assert results == ["I chose the first option"]
    assert any("What changed?" in line for line in printed)
    assert any("I chose the first option" in line for line in printed)


def test_clarify_timeout_restores_draft_without_printing_answer():
    cli = _make_cli_stub(with_app=True)
    printed = []

    with patch("cli.ChatConsole", side_effect=lambda: _FakeChatConsole(printed)), patch("builtins.print"), patch(
        "cli._cprint"
    ) as mock_cprint, patch("cli.queue.Queue.get", side_effect=__import__("queue").Empty), patch(
        "cli.CLI_CONFIG", {"clarify": {"timeout": 1}}
    ), patch("time.monotonic", side_effect=[0, 0, 2]):
        result = cli._clarify_callback("Still there?", ["yes", "no"])

    assert "did not provide a response" in result
    assert cli._app.current_buffer.text == "draft text"
    assert not printed
    assert mock_cprint.called
