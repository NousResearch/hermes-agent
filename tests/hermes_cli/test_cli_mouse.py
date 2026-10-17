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
