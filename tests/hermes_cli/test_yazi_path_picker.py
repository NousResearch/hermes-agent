import asyncio

from cli import HermesCLI


class FakeDocument:
    def __init__(self, text: str, cursor_position: int):
        self.text_before_cursor = text[:cursor_position]
        self.text_after_cursor = text[cursor_position:]


class FakeBuffer:
    def __init__(self, text: str = "", cursor_position: int | None = None):
        self.text = text
        self.cursor_position = len(text) if cursor_position is None else cursor_position

    @property
    def document(self) -> FakeDocument:
        return FakeDocument(self.text, self.cursor_position)

    def insert_text(self, value: str) -> None:
        self.text = self.text[:self.cursor_position] + value + self.text[self.cursor_position:]
        self.cursor_position += len(value)


class FakeApp:
    def __init__(self, buffer: FakeBuffer):
        self.current_buffer = buffer
        self.invalidate_count = 0

    def invalidate(self) -> None:
        self.invalidate_count += 1


def make_cli(buffer: FakeBuffer) -> HermesCLI:
    cli = HermesCLI.__new__(HermesCLI)
    setattr(cli, "_app", FakeApp(buffer))
    cli._command_running = False
    cli._sudo_state = None
    cli._secret_state = None
    cli._approval_state = None
    cli._slash_confirm_state = None
    cli._clarify_state = None
    cli._status_bar_visible = True
    return cli


def test_yazi_path_picker_returns_false_when_yazi_missing(monkeypatch):
    buffer = FakeBuffer("open")
    cli = make_cli(buffer)
    monkeypatch.setattr("cli.shutil.which", lambda name: None)

    result = asyncio.run(cli._pick_path_with_yazi(buffer))

    assert result is False
    assert buffer.text == "open"


def test_yazi_path_picker_inserts_selected_path_at_cursor(monkeypatch):
    buffer = FakeBuffer("please inspect", cursor_position=len("please inspect"))
    cli = make_cli(buffer)
    monkeypatch.setattr("cli.shutil.which", lambda name: "/usr/bin/yazi")

    async def fake_run_in_terminal(func):
        return ["/tmp/example.c"]

    monkeypatch.setattr("prompt_toolkit.application.run_in_terminal", fake_run_in_terminal)

    result = asyncio.run(cli._pick_path_with_yazi(buffer))

    assert result is True
    assert buffer.text == "please inspect /tmp/example.c"


def test_yazi_path_picker_preserves_spacing_around_inserted_path(monkeypatch):
    buffer = FakeBuffer("before after", cursor_position=len("before "))
    cli = make_cli(buffer)
    monkeypatch.setattr("cli.shutil.which", lambda name: "/usr/bin/yazi")

    async def fake_run_in_terminal(func):
        return ["/tmp/example.c"]

    monkeypatch.setattr("prompt_toolkit.application.run_in_terminal", fake_run_in_terminal)

    result = asyncio.run(cli._pick_path_with_yazi(buffer))

    assert result is True
    assert buffer.text == "before /tmp/example.c after"
