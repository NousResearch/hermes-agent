"""The idle example placeholder yields to a parked next-prompt suggestion."""

from prompt_toolkit.auto_suggest import Suggestion
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.document import Document
from prompt_toolkit.layout.controls import BufferControl
from prompt_toolkit.layout.processors import TransformationInput

from hermes_cli.cli_tui_placeholder import _PlaceholderProcessor

PROMPT = [("", "❯ ")]


def _apply(text="", suggestion=None, lineno=0):
    buffer = Buffer()
    buffer.document = Document(text, len(text))
    buffer.suggestion = Suggestion(suggestion) if suggestion is not None else None
    ti = TransformationInput(
        BufferControl(buffer), buffer.document, lineno, lambda i: i, list(PROMPT), 80, 1)
    return _PlaceholderProcessor(lambda: "Draft a reply").apply_transformation(ti).fragments


def test_suggestion_hides_placeholder():
    assert _apply(suggestion="Show me the README.") == PROMPT


def test_empty_suggestion_text_keeps_placeholder():
    assert _apply(suggestion="") == PROMPT + [("class:placeholder", "Draft a reply")]


def test_no_suggestion_appends_placeholder():
    assert _apply() == PROMPT + [("class:placeholder", "Draft a reply")]


def test_typed_text_has_no_placeholder():
    assert _apply(text="hi") == PROMPT
