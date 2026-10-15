import importlib.util
from pathlib import Path

import pytest


def _load_watch_rss():
    script_path = (
        Path(__file__).resolve().parents[2]
        / "optional-skills"
        / "devops"
        / "watchers"
        / "scripts"
        / "watch_rss.py"
    )
    spec = importlib.util.spec_from_file_location("watch_rss", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


watch_rss = _load_watch_rss()


def test_parse_feed_accepts_basic_rss():
    entries = watch_rss._parse_feed(
        b"""<?xml version="1.0"?>
<rss>
  <channel>
    <item>
      <guid>item-1</guid>
      <title>First item</title>
      <link>https://example.test/first</link>
      <description>Summary</description>
    </item>
  </channel>
</rss>
"""
    )

    assert entries == [
        {
            "id": "item-1",
            "title": "First item",
            "url": "https://example.test/first",
            "summary": "Summary",
        }
    ]


def test_parse_feed_rejects_doctype_after_long_comment(capsys):
    payload = (
        b"<?xml version=\"1.0\"?>"
        + b"<!--"
        + (b"x" * 2100)
        + b"-->"
        + b"<!DOCTYPE rss [<!ENTITY boom \"boom\">]>"
        + b"<rss><channel><item><guid>1</guid><title>&boom;</title></item></channel></rss>"
    )

    with pytest.raises(SystemExit) as exc:
        watch_rss._parse_feed(payload)

    assert exc.value.code == 2
    assert "feed rejected" in capsys.readouterr().err


def test_parse_feed_rejects_oversized_payload(monkeypatch, capsys):
    monkeypatch.setattr(watch_rss, "_MAX_FEED_BYTES", 8)

    with pytest.raises(SystemExit) as exc:
        watch_rss._parse_feed(b"<rss></rss>")

    assert exc.value.code == 2
    assert "feed too large" in capsys.readouterr().err
