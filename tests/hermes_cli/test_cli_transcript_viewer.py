"""Ctrl+O transcript viewer: a bounded per-turn log rendered into a full-screen, searchable
prompt_toolkit Application that exits on q/Esc without touching scrollback."""
import asyncio
import json

from agent.i18n import t
from hermes_cli.cli_transcript_viewer import (
    MAX_TEXT_CHARS, TranscriptLog, build_viewer_application, document)


def _log():
    log = TranscriptLog(maxlen=50)
    log.begin_turn()
    log.add_tool("terminal", {"command": "old"}, json.dumps({"output": "first turn", "exit_code": 0}),
                 duration=0.1, failed=False)
    log.begin_turn()
    log.add_reasoning("**Plan**\nthink hard", 3.0)
    log.add_tool("terminal", {"command": "make"}, json.dumps({"output": "line1\nneedle\nline3", "exit_code": 2}),
                 duration=1.5, failed=True)
    log.attach_diff("", "terminal", "+added\n-removed")
    return log


def test_last_turn_view_holds_full_output_exit_code_reasoning_and_diff():
    text = document(_log().select(all_turns=False))
    assert "needle" in text and "line3" in text
    assert t("cli.transcript.exit_code", code=2) in text
    assert "think hard" in text and "+added" in text
    assert "first turn" not in text
    assert "first turn" in document(_log().select(all_turns=True))


def test_log_is_bounded_in_entries_and_characters():
    log = TranscriptLog(maxlen=3)
    for index in range(10):
        log.add_tool("terminal", {}, f"out{index}", duration=0.0, failed=False)
    assert [e.result for e in log.entries] == ["out7", "out8", "out9"]
    log.add_tool("terminal", {}, "x" * (MAX_TEXT_CHARS * 2), duration=0.0, failed=False)
    assert len(log.entries[-1].result) < MAX_TEXT_CHARS + 200


def test_secret_values_in_args_are_not_shown(monkeypatch):
    monkeypatch.setattr("agent.redact._REDACT_ENABLED", True)
    log = TranscriptLog()
    log.add_tool("terminal", {"command": "curl -H 'Authorization: Bearer sk-abcdefghijklmnopqrstuvwxyz0123'"},
                 "ok", duration=0.0, failed=False)
    assert "sk-abcdefghijklmnopqrstuvwxyz0123" not in document(log.select(all_turns=True))


def test_empty_log_says_so():
    assert document([]) == t("cli.transcript.empty")


def test_viewer_scopes_searches_and_exits_on_escape():
    from prompt_toolkit.data_structures import Size
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    log = _log()

    async def run():
        with create_pipe_input() as pipe:
            output = DummyOutput()
            output.get_size = lambda: Size(rows=20, columns=80)
            app = build_viewer_application(log, input=pipe, output=output)
            area = app.layout.current_buffer  # the focused, read-only transcript
            assert "first turn" not in area.text
            rendered = asyncio.Event()
            app.after_render += lambda app: rendered.set()
            task = asyncio.create_task(app.run_async())
            await asyncio.wait_for(rendered.wait(), 3)

            async def send(keys):
                rendered.clear()
                pipe.send_text(keys)
                await asyncio.wait_for(rendered.wait(), 3)

            await send("/needle\r")
            assert area.document.current_line.strip() == "needle"
            await send("a")
            assert "first turn" in area.text
            pipe.send_text("\x1b")
            await asyncio.wait_for(task, 3)

    asyncio.run(run())


def test_stored_arguments_are_bounded_including_nested_values():
    from hermes_cli.cli_transcript_viewer import MAX_ARG_CHARS
    log = TranscriptLog()
    huge = {"path": "notes.md", "content": "x" * 1_000_000,
            "meta": {"rows": [{"blob": "y" * 50_000}] * 500}}
    entry = log.add_tool("write_file", huge, "ok", duration=0.0, failed=False)
    assert len(json.dumps(entry.args)) < MAX_ARG_CHARS * 2
    assert entry.args["path"] == "notes.md"
    assert "notes.md" in document(log.select(all_turns=True))
    small = {"command": "ls", "opts": [1, {"a": None}]}
    assert log.add_tool("terminal", small, "ok", duration=0.0, failed=False).args == small


def test_a_modal_prompt_arriving_while_the_viewer_is_open_closes_it():
    """An approval raised on a worker thread mid-view must not stay hidden behind the viewer."""
    import threading
    from types import SimpleNamespace
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput
    from cli import HermesCLI
    from hermes_cli.cli_subagent_monitor import modal_prompt_active

    cli = SimpleNamespace(_approval_state=None, _app=None, _terminal_io_broken=False)

    async def run():
        with create_pipe_input() as pipe:
            app = build_viewer_application(_log(), should_close=lambda: modal_prompt_active(cli),
                                           input=pipe, output=DummyOutput())
            cli._transcript_viewer_app = app
            rendered = asyncio.Event()
            app.after_render += lambda app: rendered.set()
            task = asyncio.create_task(app.run_async())
            await asyncio.wait_for(rendered.wait(), 3)
            assert not task.done()

            def approval_arrives():  # what the modal mixin does from the agent thread
                cli._approval_state = {"command": "rm -rf build"}
                HermesCLI._paint_now(cli)

            threading.Thread(target=approval_arrives).start()
            await asyncio.wait_for(task, 3)

    asyncio.run(run())


def test_ctrl_o_is_left_to_vi_in_insert_and_replace_mode(monkeypatch):
    from types import SimpleNamespace
    from prompt_toolkit.application import Application
    from prompt_toolkit.enums import EditingMode
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.key_binding.vi_state import InputMode
    from prompt_toolkit.layout import Layout
    from prompt_toolkit.output import DummyOutput
    from prompt_toolkit.widgets import TextArea
    from hermes_cli import cli_transcript_viewer as viewer
    from tests.waiting import async_wait_until

    opened = []
    monkeypatch.setattr(viewer, "open_viewer", lambda cli: opened.append(cli))
    cli = SimpleNamespace()
    editor = TextArea(text="abc", multiline=False)

    async def run():
        with create_pipe_input() as pipe:
            kb = KeyBindings()
            viewer.bind_keys(cli, kb)
            app = Application(layout=Layout(editor), key_bindings=kb, editing_mode=EditingMode.VI,
                              input=pipe, output=DummyOutput())
            task = asyncio.create_task(app.run_async())
            try:
                await async_wait_until(lambda: app.is_running, what="app running")
                for enter in ("i", "R"):  # from navigation mode into insert / replace
                    pipe.send_text("\x1b" + enter)
                    await async_wait_until(lambda: app.vi_state.input_mode != InputMode.NAVIGATION,
                                           what=f"left navigation via {enter}")
                    pipe.send_text("\x0f")  # Ctrl+O: vi's one-shot normal mode, not the viewer
                    await async_wait_until(lambda: app.vi_state.temporary_navigation_mode,
                                           what="vi took Ctrl+O")
                    pipe.send_text("\x1b")
                    await async_wait_until(lambda: not app.vi_state.temporary_navigation_mode,
                                           what="one-shot normal mode consumed")
                assert opened == []
                pipe.send_text("\x1b")
                await async_wait_until(lambda: app.vi_state.input_mode == InputMode.NAVIGATION,
                                       what="navigation mode")
                pipe.send_text("\x0f")
                await async_wait_until(lambda: opened, what="viewer opened from navigation mode")
            finally:
                app.exit()
                await task

    asyncio.run(run())
