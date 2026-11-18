"""``::followup{...}`` is a desktop-UI directive with no Feishu renderer, so it
reached Lark as literal trailing noise on every reply. ``format_message`` strips
it on the way out; these tests pin that, and pin that ordinary text with a
``::`` or a brace is left alone."""

import pytest

from plugins.platforms.feishu.adapter import FeishuAdapter


@pytest.fixture
def fmt():
    return FeishuAdapter.format_message.__get__(object.__new__(FeishuAdapter))


def test_strips_directive_on_its_own_trailing_line(fmt):
    assert fmt('Done.\n\n::followup{p1="Run the tests" p2="Open a PR"}') == "Done."


def test_strips_directive_with_unicode_payload(fmt):
    """Prompts are normally in the user's language; non-ASCII must not break the match."""
    out = fmt('Xong.\n\n::followup{p1="Chạy lại toàn bộ test" p2="Mở PR cho nhánh này"}')
    assert out == "Xong."


def test_strips_leading_indented_directive(fmt):
    assert fmt('Body.\n\n   ::followup{p1="a"}   ') == "Body."


def test_keeps_body_when_directive_is_the_whole_message(fmt):
    assert fmt('::followup{p1="a"}') == ""


def test_leaves_ordinary_prose_untouched(fmt):
    text = "Use ::followup inline like this and {braces} stay put."
    assert fmt(text) == text


def test_does_not_strip_other_directives(fmt):
    """Only followup is desktop-only noise here; MEDIA/preview are handled elsewhere."""
    text = '::preview{file="chart.html"}'
    assert fmt(text) == text


def test_multiple_directives_all_removed(fmt):
    out = fmt('A\n\n::followup{p1="x"}\n::followup{p2="y"}')
    assert out == "A"
