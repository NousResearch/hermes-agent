"""Tests for the HTML session export renderer."""

from hermes_cli.session_export_html import (
    generate_html_export,
)








def test_single_session_untitled_coalesces_none_title_and_model():
    """An un-named session (title/model still ``None`` until async title
    generation completes) is the default state, so the single-session export
    must fall back to human-readable defaults for the browser-tab ``<title>``
    and the ``Model:`` meta line — matching the on-page ``<h1>`` — instead of
    leaking the literal string ``None``."""
    session = {"id": "abc", "title": None, "model": None, "messages": []}
    html = generate_html_export(session)

    assert "<title>None</title>" not in html
    assert "<strong>Model:</strong> None" not in html
