"""Offline behavioural witnesses; generated snowflakes are test data, not corpus."""
import sys
from pathlib import Path
from datetime import datetime, timezone
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import x_ingest as ingest
import x_voice_gate as voice

NOW = 1_789_200_000.0

def row(age=1, **changes):
    tid = str((int((NOW - age * 3600) * 1000) - ingest._SNOWFLAKE_EPOCH) << 22)
    value = dict(id=tid, text="A source post", url=f"https://x.com/test/status/{tid}", source="mention")
    value.update(changes)
    return value

@pytest.mark.parametrize("bad", [row(id="unknown"), row(-0.01), row(6.001), row(30), row(created_at="yesterday"), row(url="https://evil.test/status/1"), row(origin=["mention"])])
def test_freshness_fails_closed(bad, monkeypatch):
    monkeypatch.setattr(ingest.time, "time", lambda: NOW)
    assert ingest.apply_freshness([bad], 72) == []

def test_freshness_outputs_complete_provenance_without_mutation(monkeypatch):
    monkeypatch.setattr(ingest.time, "time", lambda: NOW)
    original = row()
    result = ingest.apply_freshness([original], 6)[0]
    assert result["origin"] == "mention"
    assert datetime.fromisoformat(result["created_at"]).utcoffset().total_seconds() == 0
    assert result["age_hours"] == 1
    assert "age_hours" not in original

@pytest.mark.parametrize("draft", ["Great point", "", "Try this #agents", "RT if you agree"])
def test_voice_rejects_each_issue_without_budget(draft):
    assert not voice.voice_gate_pass(draft, max_issues=99)

def test_personal_experience_requires_evidence():
    assert any("experience" in issue for issue in voice.voice_gate_issues("I shipped the new grocery app yesterday."))

class Tab:
    def __init__(self, selected): self.selected, self.clicked = selected, False
    def click(self, **kwargs): self.clicked = True
    def get_attribute(self, key): return "true" if self.selected else "false"

class EmptyFeed:
    def __init__(self, selected): self.tab = Tab(selected)
    def goto(self, *args, **kwargs): pass
    def get_by_role(self, *args, **kwargs): return self.tab
    def wait_for_selector(self, *args, **kwargs): raise RuntimeError("offline empty feed")

def test_selected_for_you_is_not_clicked_again():
    page = EmptyFeed(True)
    assert ingest._scrape_feed(page) == []
    assert not page.tab.clicked


def test_home_explicitly_selects_and_verifies_for_you():
    page = EmptyFeed(False)
    assert ingest._scrape_feed(page) == []
    assert page.tab.clicked


def test_following_pages_cache_and_bounds(tmp_path):
    calls = []
    def fetch(cursor):
        calls.append(cursor)
        return ({"handles": ["alice"], "next_cursor": "next", "complete": False}
                if cursor is None else {"handles": ["bob", "alice"], "next_cursor": None, "complete": True})
    cache = tmp_path / "registry.json"
    result = ingest.load_following_registry(fetch, cache_path=cache, account="owner", now=NOW)
    assert result["handles"] == ["alice", "bob"]
    assert result["complete"] and calls == [None, "next"]
    assert ingest.load_following_registry(fetch, cache_path=cache, account="owner", now=NOW)["cached"]
    assert calls == [None, "next"]
    partial = ingest.load_following_registry(fetch, cache_path=tmp_path / "partial.json", account="owner", max_pages=1, now=NOW)
    assert not partial["complete"] and partial["partial_reason"] == "page_limit"
    assert partial["handles"] == ["alice"]


def test_extras_are_not_following(monkeypatch):
    monkeypatch.setattr(ingest, "_scrape_feed", lambda *args, **kwargs: [{**row(), "author": "alice", "source": kwargs["source"], "origin": kwargs["source"]}])
    assert ingest._scrape_account(None, "alice")[0]["source"] == "extra"


def test_blog_references_require_grounding(monkeypatch):
    idea = {"id": "local", "title": "sqlite atomic transaction recovery", "concept": "sqlite atomic transaction recovery"}
    monkeypatch.setattr(voice, "_load_blog_ideas", lambda: [idea])
    assert voice.blog_cross_reference("sqlite atomic transaction recovery") == []
    idea.update(url="https://example.org/research", evidence="Atomic transaction recovery in sqlite journals")
    result = voice.blog_cross_reference("sqlite atomic transaction recovery")
    assert result[0]["url"] == idea["url"]
    assert result[0]["evidence"] == idea["evidence"]


def test_corpus_approval_identity_and_integrity(tmp_path):
    import json, hashlib
    # Deliberately labelled synthetic test export; never installed as a real corpus.
    record = row(20, source="own")
    record.update(author="test", text="Synthetic fixture only", created_at=ingest.snowflake_created_at(record["id"]).isoformat())
    path = tmp_path / "synthetic.json"
    path.write_text(json.dumps([record]))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert voice.load_voice_corpus(path, account="test", now=NOW) == []
    loaded = voice.load_voice_corpus(path, account="test", approved_sha256=digest, now=NOW)
    assert loaded[0]["text"] == record["text"]
    assert loaded[0]["approved"] and loaded[0]["provenance"]["sha256"] == digest
    assert voice.load_voice_corpus(path, account="other", approved_sha256=digest, now=NOW) == []
    path.write_text("[]")
    assert voice.load_voice_corpus(path, account="test", approved_sha256=digest, now=NOW) == []


def test_real_browser_feed_selection_mentions_and_article_identity(monkeypatch):
    from playwright.sync_api import sync_playwright
    tid = row()["id"]
    other = row(2)["id"]
    timestamp = ingest.snowflake_created_at(tid).isoformat()
    html = f'''<button role="tab" aria-selected="false" onclick="this.setAttribute('aria-selected','true')">For you</button>
        <article data-testid="tweet"><a href="/quoted/status/{other}">quoted preview</a>
        <a href="/test/status/{tid}"><time datetime="{timestamp}">1h</time></a>
        <div data-testid="tweetText">Actual primary post</div></article>'''
    monkeypatch.setattr(ingest.time, "time", lambda: NOW)
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True, args=["--no-sandbox"])
        page = browser.new_page()
        # Every URL is fulfilled locally; no network or account session.
        page.route("**/*", lambda route: route.fulfill(status=200, content_type="text/html", body=html))
        try:
            rows = ingest._scrape_feed(page, limit=1)
            assert rows[0]["id"] == tid
            assert rows[0]["source"] == "for_you"
            assert page.get_by_role("tab").get_attribute("aria-selected") == "true"
            mentions = ingest._scrape_feed(page, "https://x.com/notifications/mentions", source="mention", limit=1)
            assert mentions[0]["origin"] == "mention"
            assert page.get_by_role("tab").get_attribute("aria-selected") == "false"
            extras = ingest._scrape_account(page, "test", limit=1)
            assert extras[0]["id"] == tid
            assert extras[0]["origin"] == "extra"
            assert ingest._scrape_account(page, "someone_else", limit=1) == []
        finally:
            browser.close()


def test_voice_checks_every_personal_claim():
    evidence = [{"text": "I tested the parser", "approved": True, "url": "https://x.com/test/status/1", "provenance": "synthetic fixture"}]
    assert any("experience" in issue for issue in voice.voice_gate_issues("I tested the parser. I shipped a banking app.", evidence=evidence))


@pytest.mark.parametrize("draft", ["I've shipped a grocery app.", "Built an agent last night.", "My deployment saved 100 hours."])
def test_personal_claim_variants_fail_closed(draft):
    assert any("experience" in issue for issue in voice.voice_gate_issues(draft))


def test_browser_failure_is_diagnostic_not_a_crash(monkeypatch):
    import types
    def fail(): raise RuntimeError("offline launch failure")
    monkeypatch.setitem(sys.modules, "engagement_x_poster", types.SimpleNamespace(_make_browser=fail, _ensure_logged_in=lambda page: True))
    diagnostics = {}
    assert ingest.ingest(registry={}, diagnostics=diagnostics) == []
    assert diagnostics["session"]["status"] == "unavailable"


def test_real_browser_following_virtual_pages(tmp_path):
    from playwright.sync_api import sync_playwright
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True, args=["--no-sandbox"])
        page = browser.new_page()
        profile = '<a href="/owner/following">2 Following</a>'
        grid = '''<a href="/navigation">not a followed account</a><div data-testid="UserCell"><a href="/alice">Alice</a></div>
        <script>window.scrollBy = () => { document.querySelector('[data-testid="UserCell"]').innerHTML='<a href="/bob">Bob</a>'; };</script>'''
        page.route("**/*", lambda route: route.fulfill(status=200, content_type="text/html", body=grid if route.request.url.endswith("/following") else profile))
        try:
            result = ingest.load_following_registry(ingest._following_page_reader(page, "owner"), account="owner", cache_path=tmp_path / "following.json", now=NOW)
            assert result["complete"] and result["handles"] == ["alice", "bob"]
            assert result["pages"] == 2
        finally:
            browser.close()


def test_duplicate_retains_separately_observed_mention_provenance():
    source = row(source="for_you", origin="for_you")
    mention = {**source, "source": "mention", "origin": "mention"}
    result = ingest.dedupe_by_id([source, mention])[0]
    assert result["source"] == "mention"
    assert result["origins"] == ["for_you", "mention"]
    assert source["source"] == "for_you"
    assert ingest.dedupe_by_id([result])[0]["origins"] == ["for_you", "mention"]


@pytest.mark.parametrize("changes", [{"url": "https://[invalid/status/1"}, {"created_at": datetime.fromtimestamp(NOW + .5, timezone.utc).isoformat()}])
def test_malformed_url_and_future_supplied_timestamp_fail_closed(changes, monkeypatch):
    monkeypatch.setattr(ingest.time, "time", lambda: NOW)
    assert ingest.apply_freshness([row(0, **changes)], 6) == []
