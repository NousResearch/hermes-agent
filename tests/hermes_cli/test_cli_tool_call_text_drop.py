"""Regression: assistant text sharing a message with a tool call must reach rendered output."""
from types import SimpleNamespace


def _cli_stub(monkeypatch):
    from cli import HermesCLI
    import cli as climod

    cli = HermesCLI.__new__(HermesCLI)
    cli.show_reasoning = False
    cli.final_response_markdown = "raw"
    cli.show_timestamps = False
    cli._modern_layout = False
    cli._reset_stream_state()
    cli._streamed_text_this_turn = ""
    cli._last_turn_interrupted = False
    emitted = []
    printed = []
    monkeypatch.setattr(climod, "_cprint", lambda s: emitted.append(s))
    monkeypatch.setattr(climod, "_terminal_width_for_streaming", lambda: 74)
    monkeypatch.setattr(HermesCLI, "_scrollback_box_width", lambda self: 74)
    monkeypatch.setattr(climod, "ChatConsole", lambda: SimpleNamespace(print=printed.append))
    return cli, emitted, printed


def test_tool_call_turn_content_reaches_rendered_output(monkeypatch):
    cli, emitted, printed = _cli_stub(monkeypatch)
    # Simulate a suppressed tool-call turn whose content was recorded but never
    # delivered as the final response: only the footer line is final_response.
    cli._streamed_text_this_turn = "ANSWER TEXT HERE"
    turn = SimpleNamespace(use_streaming_tts=False, box_opened=False,
                           result={"completed": True, "final_response": "footer line"})
    cli._chat_print_response_panel(turn, "footer line")
    out = " ".join(str(x) for x in printed + emitted)
    # The rendered Panel wraps the answer text in its renderable.
    for p in printed:
        r = getattr(p, "renderable", None)
        if r is not None:
            out += " " + str(r)
    assert "ANSWER TEXT HERE" in out, (printed, emitted)


def test_suffix_only_still_prints(monkeypatch):
    cli, _emitted, _printed = _cli_stub(monkeypatch)
    cli._streamed_text_this_turn = "footer line"
    turn = SimpleNamespace(use_streaming_tts=False, box_opened=False,
                           result={"completed": True, "final_response": "footer line"})
    cli._chat_print_response_panel(turn, "footer line")
    panels = [str(getattr(p, "renderable", "")) for p in _printed]
    assert len(panels) == 1, _printed
    assert panels[0].count("footer line") == 1, panels[0]


def test_disjoint_final_response_is_not_dropped(monkeypatch):
    """A later final answer must survive earlier tool-call-turn text."""
    cli, _emitted, printed = _cli_stub(monkeypatch)
    cli._streamed_text_this_turn = "The tool finished processing and saved the requested file."
    turn = SimpleNamespace(use_streaming_tts=False, box_opened=False,
                           result={"completed": True, "final_response": "Done."})
    cli._chat_print_response_panel(turn, "Done.")
    panels = [getattr(p, "renderable", None) for p in printed]
    assert len(panels) == 1, printed
    assert "The tool finished processing" in str(panels[0])
    assert "Done." in str(panels[0]), panels[0]


def test_short_tool_call_text_before_longer_final_is_kept(monkeypatch):
    """Tool-call text shorter than the final answer must not be dropped."""
    cli, _emitted, printed = _cli_stub(monkeypatch)
    cli._streamed_text_this_turn = "Saved the report to disk."
    final = "Here is a much longer final explanation of everything that happened in this turn."
    turn = SimpleNamespace(use_streaming_tts=False, box_opened=False,
                           result={"completed": True, "final_response": final})
    cli._chat_print_response_panel(turn, final)
    panels = [str(getattr(p, "renderable", "")) for p in printed]
    assert len(panels) == 1, printed
    assert "Saved the report to disk." in panels[0], panels[0]
    assert "much longer final explanation" in panels[0], panels[0]


def test_final_that_contains_the_streamed_text_is_rendered_once(monkeypatch):
    cli, _emitted, printed = _cli_stub(monkeypatch)
    cli._streamed_text_this_turn = "Checking the file."
    final = "Checking the file. It has 3 errors on lines 4, 9 and 12."
    turn = SimpleNamespace(use_streaming_tts=False, box_opened=False,
                           result={"completed": True, "final_response": final})
    cli._chat_print_response_panel(turn, final)
    panels = [str(getattr(p, "renderable", "")) for p in printed]
    assert len(panels) == 1, printed
    assert panels[0].count("Checking the file.") == 1, panels[0]
