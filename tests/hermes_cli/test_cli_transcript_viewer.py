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
