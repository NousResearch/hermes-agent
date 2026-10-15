"""A malformed publication date must not abort an otherwise readable feed."""

import importlib.util
import json
from pathlib import Path
from xml.sax.saxutils import escape

import pytest

SCRIPT = (Path(__file__).resolve().parents[2] / "optional-skills" / "research"
          / "rss-feeds" / "scripts" / "feed.py")


@pytest.fixture
def feed():
    spec = importlib.util.spec_from_file_location("rss_date_contract", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _document(kind, malformed_date):
    entries = [
        ("Recent", "2026-09-03T12:00:00+00:00"),
        ("Boundary", "2026-09-01T00:00:00+00:00"),
        ("Old", "2026-08-31T23:59:59+00:00"),
        ("Malformed", malformed_date),
        ("Undated", None),
    ]
    if kind == "json":
        doc = {"version": "https://jsonfeed.org/version/1.1", "title": "Test", "items": [
            {"id": title, "title": title, **({"date_published": date} if date else {})}
            for title, date in entries
        ]}
        return json.dumps(doc).encode(), "application/feed+json"
    if kind == "atom":
        items = "".join(
            f"<entry><title>{title}</title>"
            + (f"<updated>{escape(date)}</updated>" if date else "") + "</entry>"
            for title, date in entries
        )
        return (f'<feed xmlns="http://www.w3.org/2005/Atom"><title>Test</title>{items}</feed>'
                .encode(), "application/atom+xml")
    items = "".join(
        f"<item><title>{title}</title>"
        + (f"<pubDate>{escape(date)}</pubDate>" if date else "") + "</item>"
        for title, date in entries
    )
    return f"<rss><channel><title>Test</title>{items}</channel></rss>".encode(), "application/rss+xml"


@pytest.mark.parametrize("kind", ["rss", "atom", "json"])
@pytest.mark.parametrize("malformed_date", ["not-a-date", "2026-02-30", "2026-09-01T25:00:00"])
def test_since_skips_bad_dates_without_losing_valid_entries(feed, monkeypatch, capsys, kind, malformed_date):
    response = _document(kind, malformed_date)
    monkeypatch.setattr(feed, "fetch", lambda url: response)
    assert feed.main(["read", "https://example.test/feed", "--since", "2026-09-01", "--json"]) == 0
    output = capsys.readouterr()
    assert not output.err
    result = json.loads(output.out)
    assert [entry["title"] for entry in result["entries"]] == ["Recent", "Boundary"]
    # The normal unfiltered read still preserves the feed's original malformed value.
    unfiltered = feed.read("https://example.test/feed")
    assert next(e for e in unfiltered["entries"] if e["title"] == "Malformed")["published"] == malformed_date


@pytest.mark.parametrize("kind", ["rss", "atom", "json"])
def test_an_invalid_since_argument_still_reports_a_cli_error(feed, monkeypatch, capsys, kind):
    monkeypatch.setattr(feed, "fetch", lambda url: _document(kind, "not-a-date"))
    assert feed.main(["read", "https://example.test/feed", "--since", "not-a-cutoff", "--json"]) == 2
    output = capsys.readouterr()
    assert not output.out
    assert "error:" in output.err and "not-a-cutoff" in output.err
