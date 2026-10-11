"""Exercise production input consumers, not paste-helper/source assertions."""
import json
from queue import Queue
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from cli import HermesCLI


def paste(tmp_path, text):
    path = tmp_path / "paste.txt"
    path.write_text(text, encoding="utf-8")
    return f"[Pasted text #1: {text.count(chr(10)) + 1} lines → {path}]"


def runtime(monkeypatch):
    cli = HermesCLI.__new__(HermesCLI)
    cli._tui_unwrap_input = lambda value: (value, False, False)
    cli._typed_voice_stop = lambda _: False
    cli._pending_resume_sessions = None
    cli._app = MagicMock(is_running=False)
    cli._turn_summary_begin = MagicMock()
    cli._tui_after_turn = MagicMock()
    cli._print_user_message_preview = MagicMock()
    cli._console_print = MagicMock()
    cli.chat = MagicMock()
    monkeypatch.setattr("cli._cprint", MagicMock())
    monkeypatch.setattr("hermes_cli.bang_shell.bang_shell_enabled", lambda: True)
    return cli


@pytest.mark.parametrize("text", ["!Ref MyBucket\nResources:\n  Bucket: AWS::S3::Bucket", "!important.png\n*.log\nbuild/", "!literal\n*.tmp"])
def test_pasted_bang_data_reaches_chat_not_shell(tmp_path, monkeypatch, text):
    cli = runtime(monkeypatch)
    shell = MagicMock(return_value=0)
    monkeypatch.setattr("hermes_cli.bang_shell.run_bang_command", shell)
    cli._tui_process_one_input(paste(tmp_path, text))
    shell.assert_not_called()
    cli.chat.assert_called_once_with(text, images=None, voice_input=False)


def test_pasted_existing_path_is_text_but_typed_path_is_attachment(tmp_path, monkeypatch):
    cli = runtime(monkeypatch)
    report = tmp_path / "report.md"
    report.write_text("report")
    text = str(report) + "\n"
    cli._tui_process_one_input(paste(tmp_path, text))
    cli.chat.assert_called_once_with(text, images=None, voice_input=False)
    cli.chat.reset_mock()
    cli._tui_process_one_input(str(report))
    assert cli.chat.call_args.args[0] == f"[User attached file: {report}]"


def test_typed_bang_still_reaches_shell(monkeypatch):
    cli = runtime(monkeypatch)
    shell = MagicMock(return_value=0)
    monkeypatch.setattr("hermes_cli.bang_shell.run_bang_command", shell)
    cli._tui_process_one_input("!printf hello")
    assert shell.call_args.args[0] == "printf hello"
    cli.chat.assert_not_called()


@pytest.mark.parametrize("via_tui", [False, True])
def test_slash_handler_receives_inline_paste(tmp_path, monkeypatch, via_tui):
    cli = runtime(monkeypatch)
    cli._slash_metrics_surface = None
    cli._pending_agent_seed = None
    cli.session_id = "test"
    handler = MagicMock(return_value=True)
    cli._slash_handler = lambda _: ("test_handler", True)
    cli.test_handler = handler
    text = "!literal\nKeep Case"
    command = "/title " + paste(tmp_path, text)
    if via_tui:
        cli._tui_process_one_input(command)
    else:
        assert cli.process_command(command)
    handler.assert_called_once_with("/title " + text)
    cli.chat.assert_not_called()


@pytest.mark.parametrize("multi", [False, True])
def test_clarify_freetext_expands_answer_and_revisit_prefill(tmp_path, multi):
    cli = HermesCLI.__new__(HermesCLI)
    cli._paint_now = MagicMock()
    cli._persist_prompt_summary = MagicMock()
    cli._clarify_prefill = ""
    cli._clarify_multi_base = ["a"] if multi else None
    questions = [{"qid": "q0", "question": "first", "choices": ["a"], "multi_select": multi},
                 {"qid": "q1", "question": "second", "choices": ["b"], "multi_select": False}]
    state = {"questions": questions, "active": 0, "answers": {}, "response_queue": Queue()}
    cli._clarify_state = state
    text = 'first\n"quoted"\\path'
    buffer = MagicMock(text=paste(tmp_path, text))
    event = SimpleNamespace(app=SimpleNamespace(current_buffer=buffer, invalidate=MagicMock()))
    cli._tui_enter_clarify_freetext(event)
    answer = state["answers"]["q0"]
    assert (json.loads(answer) if multi else answer) == (["a", text] if multi else text)
    assert state["answer_meta"]["q0"]["other_text"] == text
    cli._clarify_batch_set_active(state, 0)
    cli._tui_enter_clarify_choice(event)
    assert buffer.text == text
    cli._clarify_batch_lock(state, text, meta={"kind": "other", "other_text": paste(tmp_path, text)})
    assert state["answer_meta"]["q0"]["other_text"] == text
    cli._clarify_batch_lock(state, "b")
    assert state["response_queue"].get_nowait()["q0"] == text
