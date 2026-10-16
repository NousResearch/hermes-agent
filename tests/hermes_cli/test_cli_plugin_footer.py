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
state = {"mode": "live", "boom": False}
release_slow = threading.Event()


def snapshot(session):
    if state["mode"] == "hidden":
        return None
    return {
        "label": "◈ demo · " + state["mode"],
        "tone": "warn" if state["mode"] == "held" else "ok",
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
