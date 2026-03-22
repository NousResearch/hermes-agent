from io import StringIO

from cli import HermesCLI


def test_display_background_result_falls_back_when_console_write_fails(monkeypatch):
    cli_obj = HermesCLI.__new__(HermesCLI)
    cli_obj.bell_on_complete = False

    class BrokenConsole:
        def print(self, *args, **kwargs):
            raise ValueError("I/O operation on closed file")

    fallback = StringIO()

    monkeypatch.setattr("cli.ChatConsole", lambda: BrokenConsole())
    monkeypatch.setattr("builtins.print", lambda *args, **kwargs: None)
    monkeypatch.setattr("sys.__stdout__", fallback, raising=False)

    cli_obj._display_background_result(1, "/usage", "done")

    output = fallback.getvalue()
    assert "Background task #1 complete" in output
    assert '/usage' in output
    assert "done" in output
