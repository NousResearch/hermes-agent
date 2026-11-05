"""Regression for #129969: a collapsed long-paste placeholder typed into a clarify
answer must reach the agent as content, not as a host-local path.

Drives the real keybinding handler `_tui_enter_clarify_freetext`, so the test
pins *where* the expansion happens rather than that it happens somewhere. On the
multi-select "Other" path the handler serializes with `json.dumps(base + [text])`,
so expanding the already-serialized string injects raw newlines into a JSON
string; `json.loads` then raises inside `tools/clarify_tool.py::_clean_answer`,
whose `except JSONDecodeError` degrades the whole answer to one element.
"""

import json
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from cli import HermesCLI
from hermes_cli.cli_stream_mixin import CLIStreamMixin
from tools.clarify_tool import _clean_answer


def _make_cli_stub():
    cli = HermesCLI.__new__(HermesCLI)
    cli._clarify_state = None
    cli._clarify_freetext = False
    cli._clarify_multi_base = None
    cli._clarify_prefill = ""
    cli._clarify_deadline = None
    cli._paint_now = MagicMock()
    cli._persist_prompt_summary = MagicMock()
    return cli


def _q(index, question, choices=None, multi_select=False):
    return {
        "qid": f"q{index}",
        "question": question,
        "choices": list(choices) if choices else None,
        "choices_offered": list(choices) if choices else None,
        "multi_select": bool(multi_select) and bool(choices),
    }


def _start_batch(cli, questions):
    result = {}
    thread = threading.Thread(
        target=lambda: result.setdefault("value", cli._clarify_callback(questions)),
        daemon=True,
    )
    thread.start()
    deadline = time.time() + 2
    while cli._clarify_state is None and time.time() < deadline:
        time.sleep(0.01)
    assert cli._clarify_state is not None
    return thread, result


def _paste(tmp_path, content, name="paste_1_101010.txt"):
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


def _placeholder(path, lines):
    return f"[Pasted text #1: {lines} lines → {path}]"


class _Buf:
    """Just the prompt_toolkit Buffer surface the handler touches."""

    def __init__(self, text):
        self.text = text
        self.cursor_position = len(text)
        self.resets = 0

    def reset(self, append_to_history=False):
        self.text = ""
        self.cursor_position = 0
        self.resets += 1


def _event(buf):
    app = SimpleNamespace(current_buffer=buf, invalidate=MagicMock())
    return SimpleNamespace(app=app, current_buffer=buf)


def _submit_freetext(cli, typed, *, multi_base=None):
    """Type `typed` into a freetext clarify answer and press Enter, for real."""
    cli._clarify_freetext = True
    if multi_base is not None:
        cli._clarify_multi_base = list(multi_base)
    cli._tui_enter_clarify_freetext(_event(_Buf(typed)))
    return cli._clarify_state["answers"]["q0"] if cli._clarify_state else None


def _submitted_answer(cli, typed, *, multi_base=None, multi=False):
    """Run the batch panel, submit, and return what the agent would receive."""
    questions = [_q(0, "Which?", ["alpha", "beta"], multi_select=multi)]
    thread, result = _start_batch(cli, questions)
    stored = _submit_freetext(cli, typed, multi_base=multi_base)
    thread.join(timeout=2)
    reply = result.get("value") or {}
    assert reply.get("outcome") == "submitted"
    return stored if stored is not None else reply["answers"].get("q0")


def _assert_answer(stored, expected):
    """Assert a stored multi-select answer equals *expected*.

    Every element but the last must match exactly. The last is compared on
    `.strip()`: whether the handler strips the typed text before or after
    expanding the paste is an implementation detail, but the stored answer must
    parse as a list and keep every answer the user gave.
    """
    parsed = json.loads(stored)
    assert len(parsed) == len(expected), "answer list collapsed: %r" % (parsed,)
    for got, want in zip(parsed[:-1], expected[:-1]):
        assert got == want
    assert parsed[-1].strip() == expected[-1].strip()


class TestClarifyPasteExpansion:
    def test_single_other_answer_reaches_agent_as_content(self, tmp_path):
        """The reported case: a free-text clarify answer holding a collapsed paste."""
        cli = _make_cli_stub()
        path = _paste(tmp_path, "first pasted line\nsecond pasted line\n")

        stored = _submitted_answer(cli, _placeholder(path, 2))

        assert "Pasted text #" not in stored
        # the handler strips what was typed; whether that strip runs before or
        # after the expansion is an implementation detail, so compare the content
        assert stored.strip() == "first pasted line\nsecond pasted line"
        assert _clean_answer(stored, multi=False) == "first pasted line\nsecond pasted line"

    def test_multi_other_answer_stays_a_parseable_json_list(self, tmp_path):
        """Multi-select + "Other" is stored as a JSON array *string*.

        Expanding that string after serialization leaves raw newlines inside it,
        `json.loads` raises in `_clean_answer`, and the swallowed error turns the
        checked choices plus the pasted text into a single string.
        """
        cli = _make_cli_stub()
        path = _paste(tmp_path, "first pasted line\nsecond pasted line\n")

        stored = _submitted_answer(cli, _placeholder(path, 2),
                                   multi_base=["alpha", "beta"], multi=True)

        _assert_answer(stored, [
            "alpha", "beta", "first pasted line\nsecond pasted line",
        ])
        assert _clean_answer(stored, multi=True) == [
            "alpha", "beta", "first pasted line\nsecond pasted line",
        ]

    @pytest.mark.parametrize(
        "content",
        [
            'he said "hello" and left\nsecond line\n',
            "path C:\\Users\\x\nnext\n",
            "control \x07 char here\nnext\n",
        ],
        ids=["double-quote", "backslash", "control-char"],
    )
    def test_multi_other_survives_content_needing_json_escaping(self, tmp_path, content):
        cli = _make_cli_stub()
        path = _paste(tmp_path, content)

        stored = _submitted_answer(cli, _placeholder(path, 2),
                                   multi_base=["alpha"], multi=True)

        _assert_answer(stored, ["alpha", content])
        assert _clean_answer(stored, multi=True) == ["alpha", content.strip()]

    def test_expanding_the_serialized_string_breaks_the_answer(self, tmp_path):
        """Pins the mechanism, so the ordering fix cannot silently regress.

        Run the real expander over the serialized array — the natural-looking
        place to add the call — and show the result stops parsing and degrades
        to a single element at the consumer.
        """
        path = _paste(tmp_path, "first pasted line\nsecond pasted line\n")
        stub = SimpleNamespace()
        expanded = CLIStreamMixin._expand_paste_references(
            stub, json.dumps(["alpha", "beta", _placeholder(path, 2)], ensure_ascii=False))

        with pytest.raises(json.JSONDecodeError):
            json.loads(expanded)
        assert len(_clean_answer(expanded, multi=True)) == 1

    def test_missing_paste_file_leaves_the_placeholder_intact(self, tmp_path):
        """An unreadable paste file must not swallow the answer (see #17666)."""
        cli = _make_cli_stub()
        stored = _submitted_answer(cli, _placeholder(tmp_path / "gone.txt", 2))
        assert "Pasted text #" in stored

    def test_plain_freetext_answer_is_untouched(self):
        cli = _make_cli_stub()
        assert _submitted_answer(cli, "just a normal answer") == "just a normal answer"

    def test_empty_freetext_still_skips_the_question(self):
        cli = _make_cli_stub()
        questions = [_q(0, "Which?", ["alpha", "beta"])]
        thread, result = _start_batch(cli, questions)
        cli._clarify_freetext = True
        cli._tui_enter_clarify_freetext(_event(_Buf("   ")))
        thread.join(timeout=2)
        assert (result.get("value") or {}).get("answers") == {"q0": None}
