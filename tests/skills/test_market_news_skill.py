from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import patch


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "optional-skills"
    / "finance"
    / "market-news"
    / "scripts"
    / "market_news_watch.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("market_news_skill", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_feed_supports_rss_items():
    mod = load_module()
    xml_text = """<?xml version="1.0"?>
    <rss><channel>
      <item>
        <title>SEC delays spot ETF decision</title>
        <link>https://example.com/sec-etf</link>
        <guid>sec-etf-1</guid>
        <description>Bitcoin markets brace for volatility.</description>
        <pubDate>Mon, 26 May 2026 12:00:00 GMT</pubDate>
      </item>
    </channel></rss>
    """

    rows = mod.parse_feed("https://example.com/feed.xml", xml_text)

    assert len(rows) == 1
    assert rows[0].identity == "sec-etf-1"
    assert rows[0].title == "SEC delays spot ETF decision"
    assert rows[0].source == "example.com"


def test_score_item_promotes_crypto_regulation_to_cooldown():
    mod = load_module()
    item = mod.NewsItem(
        identity="one",
        title="SEC starts new stablecoin regulation review",
        link="https://example.com/1",
        summary="Bitcoin and exchange liquidity may react.",
        published=None,
        source="example.com",
    )

    scored = mod.score_item("crypto", item)

    assert scored.level == "high"
    assert scored.action == "cooldown"
    assert "sec" in scored.tags


def test_main_json_emits_action_and_items(tmp_path, monkeypatch, capsys):
    mod = load_module()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setenv("WATCHER_STATE_DIR", str(tmp_path / ".hermes" / "watcher-state"))

    xml_text = """<?xml version="1.0"?>
    <rss><channel>
      <item>
        <title>Fed signals rate cut path</title>
        <link>https://example.com/fed</link>
        <guid>fed-1</guid>
        <description>Macro traders expect a volatility jump.</description>
        <pubDate>Mon, 26 May 2026 12:00:00 GMT</pubDate>
      </item>
    </channel></rss>
    """

    with patch.object(mod, "fetch_url", return_value=xml_text):
        exit_code = mod.main(["--profile", "forex", "--emit-initial", "--json", "--url", "https://example.com/rss"])

    rendered = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert rendered["action"] == "cooldown"
    assert rendered["count"] == 1
    assert rendered["items"][0]["level"] == "high"


def test_main_baselines_first_run_without_emit_initial(tmp_path, monkeypatch, capsys):
    mod = load_module()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setenv("WATCHER_STATE_DIR", str(tmp_path / ".hermes" / "watcher-state"))

    xml_text = """<?xml version="1.0"?>
    <rss><channel>
      <item>
        <title>Company earnings beat expectations</title>
        <link>https://example.com/earnings</link>
        <guid>earnings-1</guid>
        <description>Guidance still matters.</description>
      </item>
    </channel></rss>
    """

    with patch.object(mod, "fetch_url", return_value=xml_text):
        exit_code = mod.main(["--profile", "stocks", "--url", "https://example.com/rss"])

    assert exit_code == 0
    assert capsys.readouterr().out == ""