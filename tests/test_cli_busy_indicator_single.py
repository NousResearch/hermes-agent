"""T2.7: busy acknowledgement must be ONE indicator, not two.

RED premise: ``_busy_command`` echoes ``⏳ status`` into scrollback AND the prompt hint row
(cli_tui_mixin) paints ``<spinner frame> <status>`` — two acks per slash command run.
"""
from hermes_cli.cli_stream_mixin import CLIStreamMixin


class _Busy(CLIStreamMixin):
    def __init__(self):
        self._command_blocks_input = False
        self._command_running = False
        self._command_status = ""
        self.prints = []

    def _invalidate(self, **kwargs):
        pass


def test_busy_command_prints_no_scrollback_echo(monkeypatch):
    import cli
    busy = _Busy()
    monkeypatch.setattr(cli, "_cprint", lambda text="": busy.prints.append(text))
    with busy._busy_command("Reloading skills…"):
        assert busy._command_running is True
        assert busy._command_status == "Reloading skills…"
    assert busy._command_running is False
    assert busy._command_status == ""
    assert busy.prints == [], f"duplicate busy ack echoed: {busy.prints}"


def test_hint_row_remains_the_single_indicator():
    import inspect
    from hermes_cli.cli_tui_mixin import CLITuiMixin
    src = inspect.getsource(CLITuiMixin)
    assert "self._command_spinner_frame()" in src and "self._command_status" in src
