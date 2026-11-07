"""Literal code remains intact through the complete Matrix renderer."""

import sys

import pytest

from tests.gateway.test_matrix import _make_adapter


@pytest.mark.parametrize(("source", "expected", "fallback"), [
    ('`<script>alert(1)</script>`', '<script>alert(1)</script>', False),
    ('`<script>alert(1)</script>`', '<script>alert(1)</script>', True),
    ('```html\n<style>.demo { color: red }</style>\n```', '<style>.demo { color: red }</style>', False),
    ('```html\n<style>.demo { color: red }</style>\n```', '<style>.demo { color: red }</style>', True),
    ('`<a href="javascript:demo" onclick="demo()">example</a>`',
     '<a href="javascript:demo" onclick="demo()">example</a>', False),
    ('`<a href="javascript:demo" onclick="demo()">example</a>`',
     '<a href="javascript:demo" onclick="demo()">example</a>', True),
    ('~~~html\n<script>alert(1)</script>\n~~~', '<script>alert(1)</script>', False),
    ('    <script>alert(1)</script>', '<script>alert(1)</script>', False),
    ('``<script>with `backtick`</script>``', '<script>with `backtick`</script>', False),
    ('before <script>alert(1)</script> after', 'before  after', False),
    ('before <script>alert(1)</script> after', 'before  after', True),
])
def test_complete_markdown_pipeline_preserves_code_and_removes_unsafe_raw_html(
        monkeypatch, source, expected, fallback):
    from html.parser import HTMLParser

    class RenderedText(HTMLParser):
        def __init__(self):
            super().__init__()
            self.parts = []
            self.active_tags = []

        def handle_data(self, text):
            self.parts.append(text)

        def handle_starttag(self, tag, attrs):
            if tag in {"script", "style"}:
                self.active_tags.append(tag)

    if fallback:
        monkeypatch.setitem(sys.modules, "markdown", None)
    parser = RenderedText()
    parser.feed(_make_adapter()._markdown_to_html(source))
    assert ("".join(parser.parts).strip(), parser.active_tags) == (expected, [])
