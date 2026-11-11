"""The classic CLI's refusal behind a lost turn and its ``/discard`` reply say only what is true."""
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("mode", ["strip", "render"])
def test_refused_notice_prints_the_users_text_and_discard_line_as_written(monkeypatch, mode):
    """The notice is Hermes's copy, not a model answer: the markdown display modes must neither
    rewrite the quoted prompt nor merge the ``/discard`` line into the paragraph around it."""
    import cli as cli_mod
    from rich.console import Console
    from hermes_cli.gateway_chat_view import pending_lines

    text = "fix the *bug* in __init__.py and `run` it"
    notice = "Your message was not sent.\n" + "\n".join(pending_lines([
        {"admission_id": "a1", "status": "unknown", "text": text},
        {"admission_id": "q1", "status": "queued", "text": "next"}]))
    console = Console(record=True, width=200)
    monkeypatch.setattr(cli_mod, "ChatConsole", lambda: console)
    stub = SimpleNamespace(final_response_markdown=mode, _last_turn_interrupted=False, _streamed_text_this_turn="",
                           _stream_started=False, _stream_box_opened=False, _modern_layout=False,
                           _scrollback_box_width=lambda: 200)
    turn = SimpleNamespace(use_streaming_tts=False, box_opened=False, result={
        "failed": True, "failure_reason": "gateway_connection", "error": notice, "final_response": ""})

    cli_mod.HermesCLI._chat_print_response_panel(stub, turn, notice)

    lines = [line.strip() for line in console.export_text().splitlines()]
    assert f'"{text}"' in lines and "/discard a1" in lines, lines


def test_discard_says_when_another_lost_turn_still_blocks_the_chat(monkeypatch):
    import cli as cli_mod

    class Connection:
        pending = [{"admission_id": a, "status": "unknown", "execution_generation": 4} for a in ("a1", "a2")]

        def reattach(self, _session_id):
            return {"pending": list(self.pending)}

        def call(self, _method, **params):
            self.pending = [row for row in self.pending if row["admission_id"] != params["admission_id"]]

    printed = []
    monkeypatch.setattr(cli_mod, "_cprint", lambda line: printed.append(line))
    stub = SimpleNamespace(session_id="s")
    connection = Connection()
    monkeypatch.setattr("hermes_cli.gateway_side.connection_for", lambda _cli: connection)

    cli_mod.HermesCLI._handle_discard_command(stub, "/discard a1")
    cli_mod.HermesCLI._handle_discard_command(stub, "/discard a2")

    assert "send messages again" not in printed[0] and "/discard" in printed[0], printed
    assert "send messages again" in printed[1], printed
