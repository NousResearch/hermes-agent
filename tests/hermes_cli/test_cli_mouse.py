"""``/mouse`` flips mouse reporting on the running Application and persists ``display.cli_mouse``."""
import asyncio

import hermes_yaml as yaml
from agent.i18n import t


def _cli(enabled=False):
    from cli import HermesCLI
    cli_obj = HermesCLI.__new__(HermesCLI)
    cli_obj._mouse_enabled = enabled
    cli_obj._app = None
    cli_obj.printed = []
    cli_obj._console_print = lambda text, **kwargs: cli_obj.printed.append(text)
    return cli_obj


def test_mouse_command_turns_terminal_mouse_reporting_on_and_off_live(tmp_path, monkeypatch):
    """The real renderer sends the enable/disable sequences on the next paint after /mouse."""
    from prompt_toolkit.application import Application
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.layout import Layout
    from prompt_toolkit.output import DummyOutput
    from prompt_toolkit.widgets import TextArea
    from hermes_cli.cli_mouse import mouse_filter
    from tests.waiting import async_wait_until

    cli_obj = _cli(enabled=False)
    output = DummyOutput()
    calls = []
    output.enable_mouse_support = lambda: calls.append("on")
    output.disable_mouse_support = lambda: calls.append("off")

    async def run():
        with create_pipe_input() as pipe:
            app = Application(layout=Layout(TextArea()), input=pipe, output=output,
                              mouse_support=mouse_filter(cli_obj))
            cli_obj._app = app
            task = asyncio.create_task(app.run_async())
            try:
                await async_wait_until(lambda: app.is_running, what="app running")
                assert calls == []
                cli_obj.process_command("/mouse on")
                await async_wait_until(lambda: calls == ["on"], what="mouse reporting enabled")
                cli_obj.process_command("/mouse")  # bare toggles
                await async_wait_until(lambda: calls == ["on", "off"], what="mouse reporting disabled")
            finally:
                app.exit()
                await task
    asyncio.run(run())
    from hermes_constants import get_hermes_home
    saved = yaml.safe_load((get_hermes_home() / "config.yaml").read_text(encoding="utf-8"))
    assert saved["display"]["cli_mouse"] is False
    assert cli_obj.printed[-1].strip() == t("cli.display.mouse_off")


def test_mouse_status_reports_without_changing_and_bad_args_show_usage(tmp_path):
    cli_obj = _cli(enabled=True)
    cli_obj.process_command("/mouse status")
    assert cli_obj._mouse_enabled is True
    assert cli_obj.printed[-1].strip() == t("cli.display.mouse_state", state=t("cli.shared.label_on"))
    cli_obj.process_command("/mouse sideways")
    assert cli_obj._mouse_enabled is True
    assert cli_obj.printed[-1].strip() == t("cli.display.mouse_usage")


def test_saved_mouse_setting_is_what_the_next_cli_starts_with(monkeypatch):
    """``/mouse on`` writes display.cli_mouse; a CLI loaded from that config starts with clicks on."""
    import cli as cli_mod
    from cli import load_cli_config
    from hermes_constants import get_hermes_home
    monkeypatch.setattr(cli_mod, "_hermes_home", get_hermes_home())  # the loader's import-time home
    cli_obj = _cli(enabled=False)
    cli_obj.process_command("/mouse on")
    monkeypatch.setattr(cli_mod, "CLI_CONFIG", load_cli_config())
    fresh = cli_mod.HermesCLI(model="test-model", toolsets=[], provider="auto")
    assert fresh._mouse_enabled is True


def test_cpr_less_clicks_map_from_the_bottom_row_only_while_the_chrome_sits_there(monkeypatch):
    """Without CPR, clicks are mapped from the terminal's bottom row. That holds once startup or
    the transcript pushed the chrome there; right after a clear the chrome sits under the top rows
    with blanks below, so a click on a blank bottom row must not land on the chrome's rows."""
    import io
    from types import SimpleNamespace
    from prompt_toolkit.application import Application
    from prompt_toolkit.data_structures import Size
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import HSplit, Layout, Window
    from prompt_toolkit.layout.controls import FormattedTextControl
    from prompt_toolkit.output.vt100 import Vt100_Output
    from prompt_toolkit.widgets import TextArea
    import cli as cli_mod
    from hermes_cli import cli_render
    from hermes_cli.cli_dock_footer import bind_keys
    from tests.waiting import async_wait_until

    monkeypatch.setattr(cli_mod, "_OUTPUT_HISTORY", type(cli_mod._OUTPUT_HISTORY)())
    cli_render._set_chrome_floor(None)  # /clear: the chrome is at the top, the rows below blank
    monkeypatch.setattr(cli_render, "_CHROME_FLOOR", None)
    clicks = []
    pill = Window(FormattedTextControl(lambda: [("", "pill", lambda event: clicks.append(event.event_type))]),
                  height=1)
    output = Vt100_Output(io.StringIO(), lambda: Size(rows=40, columns=80), enable_cpr=False)

    async def run():
        with create_pipe_input() as pipe:
            kb = KeyBindings()
            bind_keys(SimpleNamespace(_subagent_monitor=None), kb)
            app = Application(layout=Layout(HSplit([TextArea(height=1), pill])), key_bindings=kb,
                              input=pipe, output=output, mouse_support=True)
            task = asyncio.create_task(app.run_async())
            try:
                await async_wait_until(lambda: app.renderer._last_screen is not None, what="first paint")
                assert not app.renderer.height_is_known  # CPR off: prompt_toolkit cannot map clicks
                pipe.send_text("\x1b[<0;2;40Mx")  # the blank bottom row, then a key to sync on
                await async_wait_until(lambda: app.layout.current_buffer.text == "x", what="click handled")
                assert clicks == []
                cli_render._pin_chrome()  # startup / a full transcript: the chrome reaches the bottom
                pipe.send_text("\x1b[<0;2;40M")  # now the bottom row is the pill
                await async_wait_until(lambda: clicks, what="pill clicked")
                assert len(clicks) == 1
            finally:
                cli_render._pin_chrome(False)
                app.exit()
                await task
    asyncio.run(run())
