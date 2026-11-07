"""The classic CLI's full-screen subagent viewer: read any subagent's run like a chat, switch
between subagents and the master view, and send to the one on screen."""
import asyncio
import json
import queue
from types import SimpleNamespace

import pytest


def _call(call_id, name, args):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


@pytest.fixture
def store(tmp_path):
    from hermes_state import SessionDB

    db = SessionDB(tmp_path / "state.db")
    db.create_session("master", "cli")
    db.append_message("master", "user", "Split the work")
    db.append_message("master", "assistant", None, tool_calls=[_call("d1", "delegate_task", {"tasks": [
        {"goal": "Write the docs"}, {"goal": "Run the tests"}]})])
    for child, goal in (("docs-child", "Write the docs"), ("tests-child", "Run the tests")):
        db.create_session(child, source="subagent", parent_session_id="master",
                          model_config={"_delegate_from": "master"})
        db.append_message(child, "user", goal)
    db.append_message("docs-child", "assistant", "Editing README.", tool_calls=[
        _call("t1", "write_file", {"path": "README.md", "content": "hi"})])
    db.append_message("docs-child", "tool", json.dumps({"bytes_written": 2}), tool_call_id="t1", tool_name="write_file")
    db.append_message("docs-child", "assistant", "README updated.")
    db.end_session("docs-child", "agent_close")
    yield db
    db.close()


@pytest.fixture
def setup(store, monkeypatch):
    from hermes_cli.cli_subagent_monitor import SubagentMonitor
    from tools import delegate_tool_registry as registry

    monkeypatch.setattr(registry, "_active_subagents", {})
    steered = []
    child_agent = SimpleNamespace(session_id="tests-child", steer=lambda text: steered.append(text) or True,
                                  interrupt=lambda *a, **kw: steered.append("STOP") or True)
    # depth 0 = a direct child, as delegate_tool_child_run registers it
    registry._register_subagent(dict(subagent_id="sa-tests", goal="Run the tests", owner_agent_session_id="master",
                                     started_at=1, status="running", agent=child_agent, depth=0))
    busy = []
    cli = SimpleNamespace(agent=SimpleNamespace(session_id="master"), session_id="master", _session_db=store,
                          _pending_input=queue.Queue(), _agent_running=False,
                          _tui_enter_while_busy=lambda text, images, payload: busy.append(text))
    monitor = SubagentMonitor(cli)
    monitor.refresh(history=True)
    return SimpleNamespace(monitor=monitor, cli=cli, steered=steered, busy=busy, db=store)


def _viewer(setup, key):
    from hermes_cli.cli_subagent_viewer import SubagentViewer
    return SubagentViewer(setup.monitor, key)


def _screen(viewer, width=80, height=40):
    return "\n".join("".join(text for _, text in line) for line in viewer.visible_lines(width, height))


def test_a_finished_subagent_reads_as_brief_steps_and_result(setup):
    viewer = _viewer(setup, "docs-child")

    screen = _screen(viewer)

    assert "Write the docs" in screen
    assert "Editing README." in screen
    assert "write_file" in screen
    assert "README updated." in screen


def test_views_run_master_first_then_each_subagent_in_list_order(setup):
    viewer = _viewer(setup, "docs-child")

    assert viewer.view_keys == ["master", "tests-child", "docs-child"]
    viewer.switch(1)
    assert viewer.current == "master"
    viewer.switch(1)
    assert viewer.current == "tests-child"
    viewer.switch(-1)
    assert viewer.current == "master"


def test_each_view_remembers_its_own_selected_entry(setup):
    viewer = _viewer(setup, "docs-child")
    viewer.move(-10)  # the first entry
    assert viewer.state.cursor == 0

    viewer.switch(1)  # master: untouched, so still on its newest entry
    assert viewer.current == "master"
    assert viewer.state.cursor == len(viewer.blocks) - 1 > 0
    viewer.switch(-1)

    assert viewer.current == "docs-child"
    assert viewer.state.cursor == 0


def test_enter_on_a_tool_entry_shows_its_result(setup):
    viewer = _viewer(setup, "docs-child")
    tool = next(i for i, b in enumerate(viewer.blocks) if b.kind == "tool")
    viewer.state.cursor = tool

    assert "bytes_written" not in _screen(viewer)
    viewer.activate()
    assert "bytes_written" in _screen(viewer)


def test_a_delegation_entry_in_the_master_view_opens_that_subagent(setup):
    viewer = _viewer(setup, "master")
    tasks = [i for i, b in enumerate(viewer.blocks) if b.kind == "delegation"]
    assert [viewer.blocks[i].link for i in tasks] == ["docs-child", "tests-child"]

    viewer.state.cursor = tasks[0]
    viewer.activate()

    assert viewer.current == "docs-child"


def test_a_background_task_shows_done_once_its_subagent_has_finished(setup):
    viewer = _viewer(setup, "master")

    outcomes = {b.link: b.outcome for b in viewer.blocks if b.kind == "delegation"}

    assert outcomes == {"docs-child": "completed", "tests-child": "running"}


def test_sending_in_a_live_subagent_view_steers_that_subagent(setup):
    viewer = _viewer(setup, "tests-child")

    assert "Run the tests" in viewer.target_label()
    result = viewer.send("Focus on the API tests")

    assert setup.steered == ["Focus on the API tests"]
    assert "error" not in result


def test_sending_in_a_finished_subagent_view_is_refused(setup):
    viewer = _viewer(setup, "docs-child")

    result = viewer.send("more please")

    assert "error" in result
    assert setup.steered == []


def test_sending_in_the_master_view_goes_to_the_master(setup):
    viewer = _viewer(setup, "master")

    viewer.send("Thanks, now summarise")
    setup.cli._agent_running = True
    viewer.send("Also check CI")

    assert setup.cli._pending_input.get_nowait() == "Thanks, now summarise"
    assert setup.busy == ["Also check CI"]


def test_a_live_view_picks_up_new_steps(setup):
    viewer = _viewer(setup, "tests-child")
    assert "pytest" not in _screen(viewer)

    setup.db.append_message("tests-child", "assistant", "Starting pytest now.")

    assert viewer.poll() is True
    assert "Starting pytest now." in _screen(viewer)
    assert viewer.poll() is False


def test_the_view_follows_new_output_while_at_the_bottom(setup):
    viewer = _viewer(setup, "tests-child")
    setup.db.append_message("tests-child", "assistant", "\n".join(f"line {n}" for n in range(30)))
    viewer.poll()

    screen = _screen(viewer, height=10)

    assert "line 29" in screen and "line 0\n" not in screen


def test_moving_up_stops_following(setup):
    viewer = _viewer(setup, "tests-child")
    setup.db.append_message("tests-child", "assistant", "first reply")
    viewer.poll()
    viewer.move(-1)

    setup.db.append_message("tests-child", "assistant", "second reply")
    viewer.poll()

    assert viewer.blocks[viewer.state.cursor].kind == "brief"


def test_keys_switch_views_send_and_return_to_the_list(setup):
    from hermes_cli.cli_subagent_monitor import build_monitor_application
    from prompt_toolkit.data_structures import Size
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput
    from tests.waiting import async_wait_until

    monitor = setup.monitor

    async def run():
        with create_pipe_input() as pipe:
            output = DummyOutput()
            output.get_size = lambda: Size(rows=30, columns=100)
            app = build_monitor_application(monitor, input=pipe, output=output)
            rendered = asyncio.Event()
            app.after_render += lambda app: rendered.set()
            task = asyncio.create_task(app.run_async())
            await asyncio.wait_for(rendered.wait(), 3)

            async def send(text):
                rendered.clear()
                pipe.send_text(text)
                await asyncio.wait_for(rendered.wait(), 3)

            await send("\r")  # open the first row (the live subagent)
            assert monitor.viewer is not None and monitor.viewer.current == "tests-child"
            await send("Use the fixtures\r")
            # The steer runs off the UI loop.
            await async_wait_until(lambda: setup.steered == ["Use the fixtures"], timeout=3)
            await send("\t")  # next view
            assert monitor.viewer.current == "docs-child"
            await send("\x1b[Z")  # Shift+Tab: back
            assert monitor.viewer.current == "tests-child"
            await send("\x1b[1;3C")  # Alt+Right (xterm): next view, not the Esc binding
            assert monitor.viewer is not None and monitor.viewer.current == "docs-child"
            await send("\x1b[1;3D")  # Alt+Left: back
            assert monitor.viewer is not None and monitor.viewer.current == "tests-child"
            await send("\x1b")  # Esc: back to the list
            assert monitor.viewer is None
            pipe.send_text("\x14")  # Ctrl+T closes
            await asyncio.wait_for(task, 3)

    asyncio.run(run())
    assert setup.steered == ["Use the fixtures"]
