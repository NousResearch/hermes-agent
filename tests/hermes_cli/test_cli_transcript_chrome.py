"""Classic CLI transcript chrome: the user-message band, turn receipt, interrupt marker,
spinner hint/verb, and queued-prompt recall. Contracts, not snapshots: colours are checked
for contrast against each other, labels against the catalog and the inputs they summarise."""
import asyncio
import queue
from types import SimpleNamespace

import pytest

import cli as cli_mod
from agent.i18n import t
from cli import HermesCLI
from hermes_cli import cli_transcript as ct
from hermes_cli.skin_engine import _BUILTIN_SKINS, load_skin, set_active_skin
from tools.ansi_strip import strip_ansi


@pytest.fixture(autouse=True)
def _restore_skin_and_light_mode():
    saved = cli_mod._LIGHT_MODE_CACHE
    yield
    cli_mod._LIGHT_MODE_CACHE = saved
    set_active_skin("default")


def _captured_echo(monkeypatch, text, **attrs):
    printed = []
    monkeypatch.setattr(cli_mod, "_cprint", printed.append)
    stub = object.__new__(HermesCLI)
    stub.user_message_preview_first_lines, stub.user_message_preview_last_lines = 2, 2
    for key, value in attrs.items():
        setattr(stub, key, value)
    stub._print_user_message_preview(text)
    return printed


# ── 1. user-message band ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("skin", sorted(_BUILTIN_SKINS))
@pytest.mark.parametrize("light", [False, True])
def test_band_text_contrasts_with_its_own_background(skin, light):
    colors = ct.user_band_colors(dict(load_skin(skin).colors), light)
    fg, bg = ct.luminance(colors["fg"]), ct.luminance(colors["bg"])
    assert abs(fg - bg) >= 0.45
    # The band is a subtle surface: it stays on the terminal's side of mid-grey.
    assert (bg > 0.5) is light


def test_skin_band_override_wins_and_text_stays_readable():
    colors = ct.user_band_colors({"user_message_bg": "#102030", "banner_text": "#101010"}, light=False)
    assert colors["bg"] == "#102030"
    assert abs(ct.luminance(colors["fg"]) - ct.luminance("#102030")) >= 0.45


def test_light_authored_skin_gets_a_light_band_on_an_undetected_terminal():
    cli_mod._LIGHT_MODE_CACHE = False
    set_active_skin("daylight")
    assert ct.luminance(ct.active_band_colors()["bg"]) > 0.5
    set_active_skin("default")
    assert ct.luminance(ct.active_band_colors()["bg"]) < 0.5
    cli_mod._LIGHT_MODE_CACHE = True
    assert ct.luminance(ct.active_band_colors()["bg"]) > 0.5


def test_echo_uses_the_skin_prompt_symbol_across_a_full_width_band(monkeypatch):
    from hermes_cli.skin_engine import get_active_prompt_symbol
    cli_mod._LIGHT_MODE_CACHE = False
    monkeypatch.setattr("shutil.get_terminal_size", lambda *a, **k: SimpleNamespace(columns=50, lines=20))
    rows = _captured_echo(monkeypatch, "fix the flaky test")
    plain = [strip_ansi(row) for row in rows]
    assert plain[0].startswith(get_active_prompt_symbol() + "fix the flaky test")
    assert "●" not in "".join(plain)
    bg = ct.active_band_colors()["bg"].lstrip("#")
    sgr = ";".join(str(int(bg[i:i + 2], 16)) for i in (0, 2, 4))
    assert all(len(row) == 50 for row in plain) and all(f"48;2;{sgr}" in row for row in rows)


def test_multiline_echo_keeps_head_and_tail_and_counts_the_hidden_middle(monkeypatch):
    rows = _captured_echo(monkeypatch, "\n".join(f"line{i}" for i in range(1, 8)))
    text = "\n".join(strip_ansi(row) for row in rows)
    for kept in ("line1", "line2", "line6", "line7"):
        assert kept in text
    assert "line4" not in text and t("cli.stream.preview_more_lines_other", count=3) in text


# ── 2. turn receipt ───────────────────────────────────────────────────────────────────────


def test_receipt_leads_with_worked_for_the_turn_duration():
    from agent.turn_summary import TurnTally, format_elapsed, format_turn_summary
    line = format_turn_summary(125.0, TurnTally(verbs={"read": {"files": 3}}))
    assert t("cli.stream.turn_worked_for", duration=format_elapsed(125.0)) in line
    assert line.endswith("read 3 files")


def test_receipt_extras_report_this_turns_tokens_and_estimated_cost_only():
    agent = SimpleNamespace(session_output_tokens=1500, session_estimated_cost_usd=0.25,
                            session_cost_status="estimated")
    extras = ct.receipt_extras(agent, token_baseline=300, cost_baseline=0.20)
    from agent.turn_summary import format_token_flow
    assert extras[0] == format_token_flow(1200)
    assert extras[1].endswith("0.05")
    for status in ("included", "unknown"):
        agent.session_cost_status = status
        assert ct.receipt_extras(agent, 300, 0.20) == [format_token_flow(1200)]
    assert ct.receipt_extras(None, 0, None) == []


def test_receipt_emit_appends_extras_to_the_tool_tally(monkeypatch):
    printed = []
    monkeypatch.setattr(cli_mod, "_cprint", printed.append)
    agent = SimpleNamespace(quiet_mode=False, session_output_tokens=0, session_estimated_cost_usd=0.0,
                            session_cost_status="estimated")
    stub = SimpleNamespace(_turn_summary_enabled=True, tool_progress_mode="all", _interactive_turn=True,
                           agent=agent, _turn_summary_collector=None)
    for name in ("_turn_summary_is_active", "_turn_summary_begin", "_turn_summary_record", "_turn_summary_emit"):
        setattr(stub, name, getattr(HermesCLI, name).__get__(stub))
    stub._turn_summary_begin()
    stub._turn_summary_record("read_file", "x", False)
    agent.session_output_tokens, agent.session_estimated_cost_usd = 2400, 0.5
    stub._turn_summary_emit()
    assert len(printed) == 1 and "read 1 file" in printed[0] and "↓ 2.4k tok" in printed[0] and "0.50" in printed[0]


# ── 3. interrupted marker ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("pending", [None, "do this instead"])
@pytest.mark.parametrize("response", ["", "partial answer"])
def test_every_interrupted_turn_shows_the_marker(pending, response):
    stub = object.__new__(HermesCLI)
    turn = SimpleNamespace(result={"interrupted": True, "interrupt_message": pending})
    alive = SimpleNamespace(is_alive=lambda: False)
    message, show = stub._chat_resolve_interrupt(turn, alive, None, response)
    assert show and message == pending and stub._last_turn_interrupted


def test_completed_turn_shows_no_marker():
    stub = object.__new__(HermesCLI)
    turn = SimpleNamespace(result={"completed": True})
    assert stub._chat_resolve_interrupt(turn, SimpleNamespace(is_alive=lambda: False), None, "ok") == (None, False)


# ── 4. spinner: turn elapsed, interrupt hint, one verb per turn ───────────────────────────


def _spinner_stub(**attrs):
    import time as _time
    stub = SimpleNamespace(_spinner_text="(◔_◔) pondering...", _tool_start_time=0, _agent_running=True,
                           _prompt_start_time=_time.time() - 10, agent=SimpleNamespace(), _spinner_token_flow=lambda: "",
                           _get_tui_terminal_width=lambda: 120,
                           _status_bar_display_width=HermesCLI._status_bar_display_width)
    stub.__dict__.update(attrs)
    stub._render_spinner_text = HermesCLI._render_spinner_text.__get__(stub)
    return stub


def test_spinner_shows_turn_elapsed_and_the_interrupt_key_after_a_few_seconds():
    line = _spinner_stub()._render_spinner_text()
    assert t("cli.stream.spinner_interrupt_hint") in line and ct.spinner_elapsed_label(10.0)[:3] in line


def test_spinner_hint_waits_for_the_threshold_and_never_wraps_the_line():
    import time as _time
    fresh = _spinner_stub(_prompt_start_time=_time.time() - (ct.SPINNER_HINT_AFTER_S / 4))
    assert t("cli.stream.spinner_interrupt_hint") not in fresh._render_spinner_text()
    narrow = _spinner_stub(_get_tui_terminal_width=lambda: 30)
    line = narrow._render_spinner_text()
    assert t("cli.stream.spinner_interrupt_hint") not in line and "pondering" in line


def test_spinner_hint_absent_when_no_turn_is_running():
    assert t("cli.stream.spinner_interrupt_hint") not in _spinner_stub(_agent_running=False)._render_spinner_text()


def test_thinking_verb_is_picked_once_per_turn():
    from agent.display import KawaiiSpinner
    faces, verbs = KawaiiSpinner.get_thinking_faces(), KawaiiSpinner.get_thinking_verbs()
    first, second = f"{faces[0]} {verbs[0]}...", f"{faces[1]} {verbs[1]}..."
    stub = SimpleNamespace()
    assert ct.stable_thinking_label(stub, first) == first
    assert ct.stable_thinking_label(stub, second) == first
    assert ct.stable_thinking_label(stub, "⏳ retrying in 5s") == "⏳ retrying in 5s"
    stub._turn_thinking_label = None  # what _turn_summary_begin does for the next turn
    assert ct.stable_thinking_label(stub, second) == second


# ── 5. queued prompts: ↳ rows and Alt+Up recall ───────────────────────────────────────────


def test_queue_rows_mark_each_prompt_and_advertise_recall():
    from hermes_cli.cli_subagent_monitor import SubagentMonitor
    pending = queue.Queue()
    for item in ("first prompt", "second prompt"):
        pending.put(item)
    dock = SubagentMonitor(SimpleNamespace(agent=None, session_id="sid-q", _goal_manager=None,
                                           _pending_input=pending))
    assert dock.refresh() and dock.has_rows
    text = dock.dock_text(columns=100, rows=30)
    assert "↳ first prompt" in text and "↳ second prompt" in text
    assert t("cli.subagents.queue_recall_hint").strip(" ·") in text


class _Composer(SimpleNamespace):
    pass


def _queue_cli(items, images=None):
    stub = SimpleNamespace(_pending_input=queue.Queue(), _attached_images=list(images or []))
    for item in items:
        stub._pending_input.put(item)
    stub._mutate_pending_input = HermesCLI._mutate_pending_input.__get__(stub)
    return stub


def test_recall_moves_the_newest_typed_prompt_into_an_empty_composer():
    from tools.process_registry_notifications import TimelineNotification
    note = TimelineNotification("raw", "Process finished", "process")
    stub = _queue_cli(["older", ("with image", ["/tmp/a.png"]), note])
    composer = _Composer(text="", cursor_position=0)
    assert ct.recall_last_queued(stub, composer)
    assert composer.text == "with image" and composer.cursor_position == len("with image")
    assert stub._attached_images == ["/tmp/a.png"]
    assert list(stub._pending_input.queue) == ["older", note]


def test_recall_never_overwrites_a_draft_or_lifts_system_turns():
    from tools.process_registry_notifications import TimelineNotification
    stub = _queue_cli(["queued"])
    draft = _Composer(text="half typed", cursor_position=3)
    assert not ct.recall_last_queued(stub, draft) and draft.text == "half typed"
    only_notes = _queue_cli([TimelineNotification("raw", "done", "process")])
    assert not ct.recall_last_queued(only_notes, _Composer(text="", cursor_position=0))
    assert only_notes._pending_input.qsize() == 1


def test_alt_up_key_recalls_through_prompt_toolkit():
    from prompt_toolkit.application import Application
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.key_binding import KeyBindings, merge_key_bindings
    from prompt_toolkit.layout import Layout
    from prompt_toolkit.output import DummyOutput
    from prompt_toolkit.widgets import TextArea
    from tests.waiting import async_wait_until

    stub = _queue_cli(["queued one", "queued two"])
    editor = TextArea(height=1, multiline=False)

    async def run():
        with create_pipe_input() as pipe:
            kb = KeyBindings()
            app = Application(layout=Layout(editor), input=pipe, output=DummyOutput(),
                              key_bindings=merge_key_bindings([editor.control.key_bindings or KeyBindings(), kb]))
            stub._app = app
            ct.bind_queue_recall(stub, kb)
            task = asyncio.create_task(app.run_async())
            try:
                await async_wait_until(lambda: app.is_running, what="app running")
                pipe.send_text("\x1b[1;3A")  # Alt+Up
                await async_wait_until(lambda: editor.text == "queued two", what="recalled")
                assert list(stub._pending_input.queue) == ["queued one"]
                pipe.send_text("\x1b[1;3A")  # composer no longer empty: no-op
                pipe.send_text("!")
                await async_wait_until(lambda: editor.text.endswith("!"), what="typed")
                assert editor.text == "queued two!" and stub._pending_input.qsize() == 1
            finally:
                app.exit()
                await task
    asyncio.run(run())
