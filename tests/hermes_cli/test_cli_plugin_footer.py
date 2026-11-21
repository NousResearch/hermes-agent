"""Plugin footer items (``ctx.register_footer_item``) in the classic CLI live-work footer: a plugin
found by real discovery paints its label and rows, Enter / letter keys / clicks run its actions,
and a raising or hanging snapshot never breaks the footer's refresh."""
import asyncio
import textwrap
import threading
from types import SimpleNamespace

import hermes_yaml as yaml
import pytest

from agent.i18n import t

PLUGIN = '''
import threading

calls = []
state = {"mode": "live", "boom": False, "badge": "◈ live"}
release_slow = threading.Event()


def snapshot(session):
    if state["mode"] == "hidden":
        return None
    return {
        "label": "◈ demo · " + state["mode"],
        "tone": "warn" if state["mode"] == "held" else "ok",
        "badge": state["badge"],
        "rows": [{"key": "r1", "label": "demo r1", "detail": "1280x800 · 2 windows"},
                 {"key": "", "label": "dropped: no key"}],
        "actions": [{"id": "watch", "label": "watch", "key": "enter"},
                    {"id": "view", "label": "view", "key": "v"},
                    {"id": "stop", "label": "stop", "key": "x", "confirm": True},
                    {"id": "broken", "label": "bad", "key": "b"},
                    {"id": "nokey", "label": "no key", "key": "F5"}],
    }


def on_action(action_id, session, row_key):
    calls.append((action_id, row_key, dict(session)))
    if action_id == "broken":
        raise RuntimeError("guest went away")
    if action_id == "view":
        def body():
            calls.append(("body-ran", row_key, {}))
            return "viewer closed"
        return {"run_in_terminal": body}
    if action_id == "watch":
        return {"open_url": "http://127.0.0.1:9/realms/r1/view", "notice": "opening watch"}
    return "stopped " + row_key


def boom(session):
    if state["boom"]:
        raise RuntimeError("status store unreadable")
    return {"label": "boom ok", "tone": "dim"}


def slow(session):
    release_slow.wait(30)
    return {"label": "slow done", "tone": "ok"}


def register(ctx):
    ctx.register_footer_item("demo", snapshot=snapshot, on_action=on_action)
    ctx.register_footer_item("boom", snapshot=boom, order=200)
    ctx.register_footer_item("slow", snapshot=slow, order=300)
'''


@pytest.fixture
def footer_plugin(tmp_path, monkeypatch):
    """The fixture plugin installed as a user plugin in a temp HERMES_HOME and discovered."""
    from hermes_cli.plugins import discover_plugins, get_plugin_manager

    home = tmp_path / "hermes-home"
    plugin_dir = home / "plugins" / "footer-fixture"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text("name: footer-fixture\nversion: '0.1.0'\n", encoding="utf-8")
    (plugin_dir / "__init__.py").write_text(textwrap.dedent(PLUGIN), encoding="utf-8")
    (home / "config.yaml").write_text(
        yaml.safe_dump({"plugins": {"enabled": ["footer-fixture"]}}), encoding="utf-8")
    bundled = tmp_path / "bundled"
    bundled.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_BUNDLED_PLUGINS", str(bundled))
    discover_plugins(force=True)
    loaded = get_plugin_manager()._plugins["footer-fixture"]
    assert loaded.enabled and loaded.error is None, loaded.error
    module = loaded.module
    try:
        yield SimpleNamespace(module=module, home=home)
    finally:
        module.release_slow.set()


def _settled(monitor, predicate, what):
    """Refresh until ``predicate`` holds: snapshots finish on worker threads, never in refresh."""
    from tests.waiting import wait_until

    def check():
        monitor.refresh()
        return predicate()
    return wait_until(check, what=what)


def _cli(session_id="sess-1"):
    return SimpleNamespace(agent=None, session_id=session_id, _invalidate=lambda *a, **k: None, _app=None)


def test_footer_item_label_and_rows_join_the_footer_for_this_session(footer_plugin):
    from hermes_cli import cli_dock_footer as footer
    from hermes_cli.cli_subagent_monitor import install_dock

    cli = _cli()
    install_dock(cli)
    monitor = cli._subagent_monitor
    _settled(monitor, lambda: any(i["label"].startswith("◈ demo") for i in monitor.plugin_items), "demo item")
    text = monitor.dock_text(columns=120, rows=30)
    assert "◈ demo · live" in text and "demo r1" in text and "1280x800 · 2 windows" in text
    assert "dropped: no key" not in text  # a row without a key is not selectable, so it is dropped
    # Rows are selectable footer rows after the tasks; the pill keeps the plugin's tone.
    assert [r["label"] for r in monitor.roster if r["kind"] == "plugin"] == ["demo r1"]
    pill = footer.dock_lines(monitor, columns=120, rows=30)[0]
    assert ("class:live-dock.ok", "◈ demo · live") in pill
    footer_plugin.module.state["mode"] = "held"
    _settled(monitor, lambda: any(i["label"] == "◈ demo · held" for i in monitor.plugin_items), "held")
    assert ("class:live-dock.warn", "◈ demo · held") in footer.dock_lines(monitor, columns=120, rows=30)[0]
    footer_plugin.module.state["mode"] = "hidden"
    _settled(monitor, lambda: not any(i["key"].endswith(":demo") for i in monitor.plugin_items), "hidden")
    assert "demo r1" not in monitor.dock_text(columns=120, rows=30)


def test_keys_on_a_plugin_row_run_its_actions_with_the_session(footer_plugin, monkeypatch):
    from prompt_toolkit.data_structures import Size
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput
    from prompt_toolkit.widgets import TextArea
    from hermes_cli import cli_plugin_footer
    from hermes_cli.cli_subagent_monitor import install_dock
    from tests.hermes_cli.test_dock_footer import _footer_app
    from tests.waiting import async_wait_until, wait_until

    opened = []
    monkeypatch.setattr(cli_plugin_footer, "open_url", lambda url: opened.append(url) or True)
    cli = _cli()
    install_dock(cli)
    monitor = cli._subagent_monitor
    _settled(monitor, lambda: monitor.plugin_rows, "plugin row")
    monitor._last_poll = float("inf")
    calls = footer_plugin.module.calls
    editor = TextArea(height=1, multiline=False)
    output = DummyOutput()
    output.get_size = lambda: Size(rows=30, columns=120)

    async def run():
        with create_pipe_input() as pipe:
            app = _footer_app(cli, editor, pipe, output)
            cli._app = app
            painted = asyncio.Event()
            app.after_render += lambda _: painted.set()
            task = asyncio.create_task(app.run_async())
            await asyncio.wait_for(painted.wait(), 3)
            try:
                pipe.send_text("\x1b[B")  # Down: into the footer, the plugin row is the only row
                await async_wait_until(lambda: monitor.footer_active, what="footer selected")
                assert monitor.selected["kind"] == "plugin"
                assert "↵ watch" in monitor.dock_text(columns=120, rows=30, selecting=True)
                pipe.send_text("\r")  # Enter runs the Enter action and never submits the composer
                await async_wait_until(lambda: monitor.notice == "opening watch", what="watch notice")
                assert opened == ["http://127.0.0.1:9/realms/r1/view"] and editor.text == ""
                action, row_key, session = calls[0]
                assert (action, row_key) == ("watch", "r1")
                assert session["session_id"] == session["stored_session_id"] == "sess-1"
                assert session["surface"] == "cli" and session["hermes_home"] == str(footer_plugin.home)
                assert session["profile"]
                pipe.send_text("v")  # a letter action that takes over the terminal, then returns
                await async_wait_until(lambda: monitor.notice == "viewer closed", what="terminal body")
                assert [c[0] for c in calls[1:]] == ["view", "body-ran"] and not monitor.opening
                pipe.send_text("x")  # confirm: the first x only asks
                await async_wait_until(lambda: monitor.confirm_action is not None, what="confirm asked")
                assert t("cli.subagents.plugin_confirm", action="stop", letter="x") in monitor.dock_text(
                    columns=120, rows=30, selecting=True)
                assert [c[0] for c in calls] == ["watch", "view", "body-ran"]
                pipe.send_text("x")
                await async_wait_until(lambda: monitor.notice == "stopped r1", what="stopped")
                pipe.send_text("b")  # a raising action becomes a short notice, the footer stays up
                await async_wait_until(lambda: "guest went away" in monitor.notice, what="error notice")
                assert monitor.footer_active
                pipe.send_text("q")  # not one of the row's keys: the composer gets it back
                await async_wait_until(lambda: editor.text == "q", what="typed")
            finally:
                app.exit()
                await task
    asyncio.run(run())
    wait_until(lambda: not monitor.opening, what="terminal released")


def test_click_on_a_plugin_row_runs_its_enter_action(footer_plugin, monkeypatch):
    from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType
    from prompt_toolkit.data_structures import Point
    from hermes_cli import cli_dock_footer as footer
    from hermes_cli import cli_plugin_footer
    from hermes_cli.cli_subagent_monitor import install_dock
    from tests.waiting import wait_until

    opened = []
    monkeypatch.setattr(cli_plugin_footer, "open_url", lambda url: opened.append(url) or True)
    cli = _cli()
    install_dock(cli)
    monitor = cli._subagent_monitor
    _settled(monitor, lambda: monitor.plugin_rows, "plugin row")
    lines = footer.dock_lines(monitor, columns=120, rows=30, actions=footer.click_actions(cli))
    row_line = next(line for line in lines if any("demo r1" in fragment[1] for fragment in line))
    handler = row_line[0][2]
    handler(MouseEvent(Point(0, 0), MouseEventType.MOUSE_UP, MouseButton.LEFT, frozenset()))
    wait_until(lambda: opened, what="clicked row opened")
    assert footer_plugin.module.calls[0][:2] == ("watch", "r1")


def _status_bar_cli(monitor):
    """A real HermesCLI status bar reading the footer monitor that polled the plugin."""
    from datetime import datetime, timedelta
    from cli import HermesCLI

    cli_obj = HermesCLI.__new__(HermesCLI)
    cli_obj.model = "anthropic/claude-sonnet-4"
    cli_obj.session_start = datetime.now() - timedelta(minutes=2)
    cli_obj.agent = None
    cli_obj._background_tasks = {}
    cli_obj._subagent_monitor = monitor
    return cli_obj


def test_an_item_badge_is_a_read_only_status_bar_segment_while_the_item_is_visible(footer_plugin):
    from hermes_cli.cli_subagent_monitor import install_dock

    cli = _cli()
    install_dock(cli)
    monitor = cli._subagent_monitor
    _settled(monitor, lambda: any(i.get("badge") == "◈ live" for i in monitor.plugin_items), "badge")
    bar = _status_bar_cli(monitor)
    # Medium and wide tiers show it; the narrow tier drops it like the other optional counters.
    assert "◈ live" in bar._build_status_bar_text(width=140)
    assert "◈ live" in bar._build_status_bar_text(width=60)
    assert "◈ live" not in bar._build_status_bar_text(width=40)
    # Painted in the item's tone; a badge is display only, so it carries no mouse handler.
    bar._status_bar_visible = True
    bar._prompt_stash = SimpleNamespace(indicator=lambda: "")
    bar._get_tui_terminal_width = lambda: 140
    badge = [frag for frag in bar._get_status_bar_fragments() if frag[1] == "◈ live"]
    assert badge == [("class:status-bar-good", "◈ live")]
    # A hidden item takes its badge with it.
    footer_plugin.module.state["mode"] = "hidden"
    _settled(monitor, lambda: not monitor.plugin_items or all(not i.get("badge") for i in monitor.plugin_items),
             "hidden")
    assert "◈" not in bar._build_status_bar_text(width=140)


def test_badges_are_sanitised_clipped_and_follow_status_bar_fields(footer_plugin, monkeypatch):
    import cli as cli_mod
    from prompt_toolkit.utils import get_cwidth
    from hermes_cli.cli_plugin_footer import BADGE_COLUMNS, status_badges
    from hermes_cli.cli_subagent_monitor import install_dock

    footer_plugin.module.state["badge"] = "◈ live\x1b[31m\nwith a much longer tail"
    cli = _cli()
    install_dock(cli)
    monitor = cli._subagent_monitor
    _settled(monitor, lambda: any(i.get("badge") for i in monitor.plugin_items), "long badge")
    ((tone, text),) = status_badges(cli)
    assert tone == "ok" and text.startswith("◈ live") and text.endswith("…")
    assert text.isprintable() and get_cwidth(text) <= BADGE_COLUMNS
    bar = _status_bar_cli(monitor)
    assert text in bar._build_status_bar_text(width=140)
    hidden = _status_bar_cli(monitor)
    monkeypatch.setattr(cli_mod, "CLI_CONFIG", {"display": {"status_bar": {"fields": ["model", "duration"]}}})
    assert text not in hidden._build_status_bar_text(width=140)


def test_raising_or_hanging_snapshots_never_break_the_footer(footer_plugin):
    from hermes_cli.cli_subagent_monitor import install_dock

    module = footer_plugin.module
    cli = _cli()
    install_dock(cli)
    monitor = cli._subagent_monitor
    labels = lambda: {i["label"] for i in monitor.plugin_items}
    _settled(monitor, lambda: {"boom ok", "◈ demo · live"} <= labels(), "boom and demo")
    assert "slow done" not in labels()  # still running in its worker; refresh did not wait for it
    module.state["boom"] = True
    module.state["mode"] = "held"
    # The raising snapshot keeps its last good value while the others keep updating.
    _settled(monitor, lambda: "◈ demo · held" in labels(), "demo updated while boom raises")
    assert "boom ok" in labels() and "slow done" not in labels()
    module.release_slow.set()
    _settled(monitor, lambda: "slow done" in labels(), "slow snapshot arrives later")


def test_a_new_session_never_shows_the_previous_sessions_value(footer_plugin):
    from hermes_cli.cli_subagent_monitor import install_dock

    cli = _cli("sess-a")
    install_dock(cli)
    monitor = cli._subagent_monitor
    _settled(monitor, lambda: monitor.plugin_rows, "first session")
    footer_plugin.module.state["mode"] = "hidden"
    cli.session_id = "sess-b"
    monitor.refresh()
    assert not monitor.plugin_rows  # cleared at the switch, before the new snapshot lands


def test_unload_removes_the_item_and_bad_registrations_are_refused(footer_plugin):
    from hermes_cli.plugins import get_plugin_footer_items, get_plugin_manager

    manager = get_plugin_manager()
    assert {i["name"] for i in get_plugin_footer_items()} >= {"demo", "boom", "slow"}
    ctx = next(iter(manager._plugins.values()))
    manager.unload()
    assert get_plugin_footer_items() == []
    del ctx
    from hermes_cli.plugins import PluginContext
    from hermes_cli.plugins_manifest import PluginManifest
    ctx = PluginContext(PluginManifest(name="bad-footer"), manager)
    with pytest.raises(ValueError):
        ctx.register_footer_item("Bad Name!", snapshot=lambda s: None)
    with pytest.raises(ValueError):
        ctx.register_footer_item("ok", snapshot="not callable")
    handle = ctx.register_footer_item("ok", snapshot=lambda s: None)
    assert [i["key"] for i in get_plugin_footer_items()] == ["bad-footer:ok"]
    handle.dispose()
    assert get_plugin_footer_items() == []


def test_action_results_are_validated_before_they_touch_the_terminal_or_browser(monkeypatch):
    """Only http(s) URLs open; a run_in_terminal callable from the plugin host (another process,
    no terminal) or without an interactive app is refused with a notice instead of running."""
    from hermes_cli import cli_plugin_footer
    from hermes_cli.cli_subagent_monitor import SubagentMonitor

    opened, ran = [], []
    monkeypatch.setattr(cli_plugin_footer, "open_url", lambda url: opened.append(url) or True)
    cli = _cli()
    monitor = SubagentMonitor(cli)
    item = {"key": "p:x", "plugin": "p"}
    cli_plugin_footer.apply_result(cli, monitor, item, {"open_url": "file:///etc/passwd"})
    assert opened == [] and "http(s)" in monitor.notice
    cli_plugin_footer.apply_result(cli, monitor, item, {"open_url": "https://example.invalid/x"})
    assert opened == ["https://example.invalid/x"]

    def host_body():
        ran.append("host")
    host_body.__hermes_plugin_host_ref__ = 7
    cli_plugin_footer.apply_result(cli, monitor, item, {"run_in_terminal": host_body})
    assert ran == [] and "in_process" in monitor.notice
    cli_plugin_footer.apply_result(cli, monitor, item, {"run_in_terminal": lambda: ran.append("local")})
    assert ran == [] and "interactive terminal" in monitor.notice  # no running app here


def _plugin_row(*actions, key="r1"):
    return {"kind": "plugin", "key": f"plugin:p:item:{key}", "item": "p:item", "plugin": "p", "row_key": key,
            "label": f"row {key}", "detail": "", "tone": "ok",
            "actions": [{"id": ident, "label": ident, "key": letter, "confirm": confirm}
                        for ident, letter, confirm in actions]}


def test_a_terminal_callable_writes_to_the_terminal_while_it_runs_under_patch_stdout(monkeypatch):
    """The CLI runs inside ``patch_stdout()``: a viewer or prompt printed by a ``run_in_terminal``
    callable must reach the terminal while the callable runs, not after it returns."""
    import io
    import time
    from prompt_toolkit.application import Application
    from prompt_toolkit.application.current import get_app_session
    from prompt_toolkit.data_structures import Size
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.layout import Layout, Window
    from prompt_toolkit.layout.controls import FormattedTextControl
    from prompt_toolkit.output.vt100 import Vt100_Output
    from prompt_toolkit.patch_stdout import patch_stdout
    from hermes_cli import cli_plugin_footer
    from hermes_cli.cli_subagent_monitor import SubagentMonitor
    from tests.waiting import async_wait_until

    class Terminal(io.StringIO):
        def isatty(self):
            return True

    terminal = Terminal()
    output = Vt100_Output(terminal, lambda: Size(rows=24, columns=80), term="xterm")
    monkeypatch.setattr(get_app_session(), "_output", output)  # the session the real CLI runs in
    seen = []

    def body():
        print("viewer frame 1", flush=True)
        deadline = time.monotonic() + 3
        while "viewer frame 1" not in terminal.getvalue() and time.monotonic() < deadline:
            time.sleep(0.01)
        seen.append("viewer frame 1" in terminal.getvalue())
        return "viewer closed"

    cli = _cli()
    monitor = SubagentMonitor(cli)

    async def run():
        with create_pipe_input() as pipe, patch_stdout():
            app = Application(layout=Layout(Window(FormattedTextControl("composer"))), input=pipe, output=output)
            cli._app = app
            painted = asyncio.Event()
            app.after_render += lambda _: painted.set()
            task = asyncio.create_task(app.run_async())
            await asyncio.wait_for(painted.wait(), 3)
            try:
                cli_plugin_footer.run_in_terminal(cli, monitor, {"key": "p:item", "plugin": "p"}, body)
                await async_wait_until(lambda: monitor.notice == "viewer closed", what="body finished")
                print("chat line after the viewer", flush=True)
                await async_wait_until(lambda: "chat line after the viewer" in terminal.getvalue(),
                                       what="chat output still reaches the terminal afterwards")
            finally:
                app.exit()
                await task
    asyncio.run(run())
    assert seen == [True]


def test_a_pending_action_ignores_repeats_until_it_finishes(monkeypatch):
    """Holding an action key while the backend blocks runs one action per row, not a thread per
    key press; once it finishes the key works again."""
    from hermes_cli import cli_plugin_footer, plugins
    from hermes_cli.cli_subagent_monitor import SubagentMonitor
    from tests.waiting import wait_until

    release, calls = threading.Event(), []

    def on_action(action_id, session, row_key):
        calls.append((action_id, row_key))
        release.wait(10)
        return f"done {row_key}"

    item = {"key": "p:item", "plugin": "p", "on_action": on_action}
    monkeypatch.setattr(plugins, "get_plugin_footer_items", lambda: [item])
    cli = _cli()
    monitor = SubagentMonitor(cli)
    row, other = _plugin_row(("go", "g", False)), _plugin_row(("go", "g", False), key="r2")
    action = row["actions"][0]
    first = cli_plugin_footer.run_action(cli, monitor, row, action)
    wait_until(lambda: calls, what="first action running")
    assert [cli_plugin_footer.run_action(cli, monitor, row, action) for _ in range(5)] == [None] * 5
    assert "still running" in monitor.notice
    second = cli_plugin_footer.run_action(cli, monitor, other, action)  # another row is independent
    wait_until(lambda: len(calls) == 2, what="other row's action running")
    release.set()
    first.join(5)
    second.join(5)
    assert calls == [("go", "r1"), ("go", "r2")]
    again = cli_plugin_footer.run_action(cli, monitor, row, action)
    assert again is not None
    again.join(5)
    assert calls[-1] == ("go", "r1")


def test_a_click_on_a_plugin_row_shows_its_confirmation_before_running_it(monkeypatch):
    from prompt_toolkit.data_structures import Point
    from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType
    from hermes_cli import cli_dock_footer as footer
    from hermes_cli import cli_plugin_footer
    from hermes_cli.cli_subagent_monitor import SubagentMonitor

    ran = []
    monkeypatch.setattr(cli_plugin_footer, "run_action", lambda cli, monitor, row, action: ran.append(action["id"]))
    cli = _cli()
    monitor = cli._subagent_monitor = SubagentMonitor(cli)
    monitor.plugin_rows = [_plugin_row(("stop", "enter", True))]
    monitor.plugin_items = [{"key": "p:item", "plugin": "p", "label": "", "tone": "ok", "badge": ""}]
    monitor._last_poll = float("inf")
    key = monitor.plugin_rows[0]["key"]
    click = MouseEvent(Point(0, 0), MouseEventType.MOUSE_UP, MouseButton.LEFT, frozenset())
    footer.click_actions(cli)["row"](click, key)
    assert ran == []
    # The clicked row is selected, so the footer paints the confirmation the next click answers.
    assert monitor.footer_active and monitor.selected_id == key
    assert t("cli.subagents.plugin_confirm", action="stop", letter="enter") in monitor.dock_text(
        columns=120, rows=30, selecting=True)
    footer.click_actions(cli)["row"](click, key)
    assert ran == ["stop"]


def test_a_replaced_registration_gets_its_own_snapshot_and_drops_the_old_result():
    from hermes_cli.cli_plugin_footer import FooterItemPoller

    release = threading.Event()

    def stale(session):
        release.wait(10)
        return {"label": "old plugin", "tone": "ok"}

    poller = FooterItemPoller()
    session = {"session_id": "s"}
    old = {"key": "p:item", "plugin": "p", "snapshot": stale}
    new = {"key": "p:item", "plugin": "p", "snapshot": lambda s: {"label": "new plugin", "tone": "ok"}}
    assert poller.poll(session, items=[old]) == []  # the old snapshot hangs in its worker
    assert [i["label"] for i in poller.poll(session, items=[new], wait=5)] == ["new plugin"]
    release.set()
    for _ in range(50):
        labels = [i["label"] for i in poller.poll(session, items=[new], wait=0.05)]
        assert labels == ["new plugin"]


def test_task_keys_never_act_on_a_plugin_row_that_does_not_advertise_them():
    """``s`` (steer) and ``x`` (stop) belong to task rows; on a plugin row without those actions
    they are ordinary typing, like any other unadvertised letter."""
    from prompt_toolkit.data_structures import Size
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput
    from prompt_toolkit.widgets import TextArea
    from hermes_cli.cli_subagent_monitor import install_dock
    from tests.hermes_cli.test_dock_footer import _footer_app
    from tests.waiting import async_wait_until

    cli = _cli()
    install_dock(cli)
    monitor = cli._subagent_monitor
    monitor.plugin_rows = [_plugin_row(("open", "enter", False))]
    monitor._last_poll = float("inf")
    editor = TextArea(height=1, multiline=False)
    output = DummyOutput()
    output.get_size = lambda: Size(rows=30, columns=120)

    async def run():
        with create_pipe_input() as pipe:
            app = _footer_app(cli, editor, pipe, output)
            painted = asyncio.Event()
            app.after_render += lambda _: painted.set()
            task = asyncio.create_task(app.run_async())
            await asyncio.wait_for(painted.wait(), 3)
            try:
                for letter in "sx":
                    editor.text = ""
                    pipe.send_text("\x1b[B")
                    await async_wait_until(lambda: monitor.footer_active, what="plugin row selected")
                    pipe.send_text(letter)
                    await async_wait_until(lambda: editor.text == letter, what=f"{letter} typed")
                    assert monitor.steer_target is None and monitor.confirm_key is None
            finally:
                app.exit()
                await task
    asyncio.run(run())


def test_plugin_text_is_one_line_wherever_the_footer_paints_it(monkeypatch):
    from hermes_cli import cli_dock_footer as footer
    from hermes_cli import cli_plugin_footer
    from hermes_cli.cli_subagent_monitor import SubagentMonitor

    item = cli_plugin_footer.normalize({
        "label": "◈ two\nlines\x1b[31m", "tone": "ok",
        "rows": [{"key": "r1", "label": "row\none", "detail": "a\r\nb"}],
        "actions": [{"id": "open", "label": "op\nen", "key": "enter"}]})
    cli = _cli()
    monitor = cli._subagent_monitor = SubagentMonitor(cli)
    monitor.plugin_items = [{**item, "key": "p:item", "plugin": "p"}]
    monitor.plugin_rows = cli_plugin_footer.footer_rows(monitor.plugin_items)
    monitor._last_poll = float("inf")
    monitor.footer_active, monitor.selected_id = True, monitor.plugin_rows[0]["key"]

    def painted():
        lines = footer.dock_lines(monitor, columns=120, rows=30, selecting=True)
        return [text for line in lines for _style, text, *_ in line]
    texts = painted()
    assert all(text.isprintable() for text in texts), texts
    assert any(text.startswith("◈ two lines") for text in texts) and any("op en" in text for text in texts)
    cli_plugin_footer.apply_result(cli, monitor, monitor.plugin_items[0], "closed\nviewer\x07")
    assert monitor.notice == "closed viewer" and all(text.isprintable() for text in painted())


def test_row_keys_reach_on_action_exactly_or_the_row_is_skipped():
    from hermes_cli.cli_plugin_footer import MAX_ROW_KEY, normalize

    exact = "k" * MAX_ROW_KEY
    rows = normalize({"label": "x", "rows": [{"key": exact + "a"}, {"key": exact + "b"}, {"key": exact},
                                             {"key": "short"}]})["rows"]
    assert [row["key"] for row in rows] == [exact, "short"]
