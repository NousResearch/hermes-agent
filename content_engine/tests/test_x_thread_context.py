"""Synthetic DOM fixtures, real headless Chromium, no live requests or approval."""
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'content_engine'))
sys.path.insert(0, str(ROOT / 'scripts/content_engine'))
import x_ingest


def post(age=1, author='alice'):
    tid = str((int((time.time() - age * 3600) * 1000) - x_ingest._SNOWFLAKE_EPOCH) << 22)
    return dict(id=tid, author=author, text='Synthetic source about bounded context collection and careful attribution.',
                url=f'https://x.com/{author}/status/{tid}', source='mention', origin='mention',
                created_at=x_ingest.snowflake_created_at(tid).isoformat())


def card(row, extra='', tag='article', attrs='data-testid="tweet"'):
    return (f'<{tag} {attrs}><a href="/{row["author"]}/status/{row["id"]}">'
            f'<time datetime="{row["created_at"]}">now</time></a>'
            f'<div data-testid="tweetText">{row["text"]}</div>{extra}</{tag}>')


@pytest.fixture
def browser_page():
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page = browser.new_page()
        yield page
        browser.close()


def ingestion(monkeypatch, page, routes, registry=None):
    import engagement_x_poster
    calls = []
    def serve(route):
        calls.append((route.request.method, route.request.url))
        body, status = routes.get(route.request.url, ('<main>unavailable</main>', 404))
        route.fulfill(status=status, content_type='text/html', body=body)
    page.route('**/*', serve)
    import x_thread_context
    monkeypatch.setattr(x_thread_context, 'make_readonly_browser', lambda: (
        SimpleNamespace(stop=lambda: None), SimpleNamespace(close=lambda: None), None, page))
    monkeypatch.setattr(engagement_x_poster, '_ensure_logged_in', lambda p: True)
    monkeypatch.setattr(x_ingest.time, 'sleep', lambda seconds: None)
    diagnostics = {}
    result = x_ingest.ingest(include_home=False, registry=registry or {}, diagnostics=diagnostics)
    return result, calls, diagnostics


def test_real_dom_ingestion_fetches_historical_context_without_promoting_it(browser_page, monkeypatch):
    source, ancestor, quoted, other = post(), post(72, 'parent'), post(48, 'quoted'), post(1, 'other')
    quote = card(quoted, tag='div', attrs='role="link" data-testid="quoteTweet"')
    routes = {
        'https://x.com/notifications/mentions': (card(source), 200),
        f'https://x.com/i/web/status/{source["id"]}': (card(ancestor) + card(source, quote) + card(other), 200),
    }
    rows, calls, diagnostics = ingestion(monkeypatch, browser_page, routes)
    assert len(rows) == 1
    row = rows[0]
    assert {c['id'] for c in row.get('thread_context', [])} == {ancestor['id'], quoted['id']}
    assert row['origin'] == 'mention' and row['id'] == source['id']
    assert row['context_status']['state'] == 'partial'
    assert row['context_status']['complete'] is False
    assert row['context_status']['source_id'] == source['id']
    assert all(c['source_id'] == source['id'] and c['origin'] == 'conversation' for c in row['thread_context'])
    assert all(c['observed_url'].endswith(source['id']) for c in row['thread_context'])
    assert len(calls) == 2 and all(method == 'GET' for method, _ in calls)
    assert x_ingest.apply_freshness([row])
    assert not x_ingest.apply_freshness(row['thread_context'])


def test_block_stops_remaining_context_requests(browser_page, monkeypatch):
    one, two = post(1), post(2)
    routes = {'https://x.com/notifications/mentions': (card(one) + card(two), 200),
              f'https://x.com/i/web/status/{one["id"]}': ('<main>Rate limit exceeded</main>', 429)}
    rows, calls, diag = ingestion(monkeypatch, browser_page, routes)
    assert len(rows) == 2
    assert len(calls) == 2
    assert rows[0].get('context_status', {}).get('reason') == 'http_429'
    assert rows[1]['context_status']['reason'] == 'stopped:http_429'
    assert all(not row['thread_context'] for row in rows)


def test_context_budget_and_stale_source(browser_page, monkeypatch):
    fresh, stale = post(1), post(7)
    rows, calls, diag = ingestion(monkeypatch, browser_page,
        {'https://x.com/notifications/mentions': (card(fresh) + card(stale), 200)},
        registry={'context_max_sources': 0})
    assert len(rows) == 1 and len(calls) == 1
    assert rows[0].get('context_status', {}).get('reason') == 'source_budget'


def test_repost_is_not_selected_or_context(browser_page):
    row = post()
    browser_page.set_content(card(row, '<div data-testid="socialContext">bob reposted</div>'))
    assert x_ingest._extract_article(browser_page.locator('article')) is None



def test_context_survives_scout_stage_reload_and_review(browser_page, monkeypatch, tmp_path):
    import hashlib
    import json
    import x_quote_scout as scout
    import x_manager as xm
    import x_manager_report as report
    from x_delivery import prepare_delivery
    source, ancestor = post(), post(72, 'parent')
    rows, calls, diag = ingestion(monkeypatch, browser_page, {
        'https://x.com/notifications/mentions': (card(source), 200),
        f'https://x.com/i/web/status/{source["id"]}': (card(ancestor) + card(source), 200)})
    home = tmp_path/'home'
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setattr(xm, 'DB_PATH', home/'stage.db')
    monkeypatch.setattr(scout, 'VERDICT_FILE', home/'verdicts.json')
    monkeypatch.setattr(report, 'REPORT_DIR', home/'reports')
    corpus = home/'research/x-voice/sahil-public-corpus-2026.json'
    corpus.parent.mkdir(parents=True)
    own = post(48, 'Sahil_Saghir')
    corpus.write_text(json.dumps([{**own, 'text': 'Small queues keep failures visible.'}]))
    Path(str(corpus)+'.approval.json').write_text(json.dumps({
        'approved': True, 'sha256': hashlib.sha256(corpus.read_bytes()).hexdigest()}))
    skill = home / 'skills/voice/SKILL.md'
    skill.parent.mkdir(parents=True)
    skill.write_text('Synthetic voice guidance')
    (skill.parent / 'references').mkdir()
    (skill.parent / 'references/runtime-voice.md').write_text('Synthetic voice guidance')
    (skill.parent / 'references/approved-conversational-calibration.md').write_text('Synthetic calibration')
    monkeypatch.setattr(scout, 'VOICE_SKILL', skill)
    prompts = []
    def draft(system, user, **kwargs):
        prompts.append(user)
        return json.dumps(dict(verdict='quote', reason='fixture', post='The source says: "' + rows[0]['text'] + '"'))
    monkeypatch.setattr(scout, '_call_llm_chain', draft)
    artifacts, seeds, discards = scout._candidate_artifacts(rows)
    assert len(artifacts) == 1 and not seeds
    assert ancestor['id'] in prompts[0]
    assert 'historical_grounding' in prompts[0]
    assert 'coverage' in prompts[0]
    artifact = artifacts[0]
    assert [s['id'] for s in artifact.pack.context['sources']] == [source['id']]
    xm.stage_for_approval(artifact)
    loaded = prepare_delivery(artifact.id)
    assert loaded.pack.context['thread_context'][0]['id'] == ancestor['id']
    path = report.render_report([loaded], lane='quote_scan', title='Synthetic thread test')
    html = path.read_text()
    assert ancestor['id'] in html and source['id'] in html
    assert 'possible_ancestor' in html and 'partial' in html and 'full ancestor chain' in html


def test_standalone_seed_keeps_context(browser_page, monkeypatch, tmp_path):
    import x_quote_scout as scout
    source, ancestor = post(), post(72, 'parent')
    rows, calls, diag = ingestion(monkeypatch, browser_page, {
        'https://x.com/notifications/mentions': (card(source), 200),
        f'https://x.com/i/web/status/{source["id"]}': (card(ancestor) + card(source), 200)})
    monkeypatch.setattr(scout, 'VERDICT_FILE', tmp_path/'verdicts.json')
    monkeypatch.setattr(scout, '_draft', lambda row: {'verdict': 'standalone'})
    _, seeds, _ = scout._candidate_artifacts(rows)
    assert seeds[0].get('thread_context') == rows[0]['thread_context']
    assert seeds[0]['context_status'] == rows[0]['context_status']


@pytest.mark.parametrize('body,status,reason', [
    ('<div role="alert">Try again later</div>', 200, 'blocked_or_rate_limited'),
    ('<input autocomplete="username">', 200, 'login_required'),
    ('<main>Forbidden</main>', 403, 'http_403'),
])
def test_visible_block_or_login_stops_context_batch(browser_page, monkeypatch, body, status, reason):
    one, two = post(1), post(2)
    rows, calls, diag = ingestion(monkeypatch, browser_page, {
        'https://x.com/notifications/mentions': (card(one) + card(two), 200),
        f'https://x.com/i/web/status/{one["id"]}': (body, status)})
    assert len(calls) == 2
    assert rows[0]['context_status']['reason'] == reason
    assert rows[1]['context_status']['reason'] == 'stopped:' + reason


def test_missing_target_does_not_label_unrelated_posts_ancestors(browser_page, monkeypatch):
    source, unrelated = post(), post(2, 'unrelated')
    rows, calls, diag = ingestion(monkeypatch, browser_page, {
        'https://x.com/notifications/mentions': (card(source), 200),
        f'https://x.com/i/web/status/{source["id"]}': (card(unrelated), 200)})
    assert rows[0]['thread_context'] == []
    assert rows[0]['context_status']['state'] == 'unknown'
    assert rows[0]['context_status']['reason'] == 'source_not_visible'



def test_quoted_timestamp_and_text_cannot_become_outer_source(browser_page):
    source, quote = post(), post(48, 'quoted')
    quoted = card(quote, tag='div', attrs='role="link" data-testid="quoteTweet"')
    # Deliberately put a nested timestamp before the outer timestamp.
    browser_page.set_content(card(source).replace('>', '>' + quoted, 1))
    result = x_ingest._extract_article(browser_page.locator('article'))
    assert result['id'] == source['id'] and result['text'] == source['text']
    browser_page.set_content(f'<article data-testid="tweet">{quoted}</article>')
    assert x_ingest._extract_article(browser_page.locator('article')) is None


def test_quote_provenance_names_actual_parent(browser_page, monkeypatch):
    source, ancestor, quote = post(), post(72, 'parent'), post(96, 'quoted')
    quoted = card(quote, tag='div', attrs='role="link" data-testid="quoteTweet"')
    rows, _, _ = ingestion(monkeypatch, browser_page, {
        'https://x.com/notifications/mentions': (card(source), 200),
        f'https://x.com/i/web/status/{source["id"]}': (card(ancestor, quoted) + card(source), 200)})
    context = next(c for c in rows[0]['thread_context'] if c['id'] == quote['id'])
    assert context.get('related_post_id') == ancestor['id']
    assert context['source_id'] == source['id']



def test_ingestion_uses_stock_headless_not_posting_launcher(monkeypatch, tmp_path):
    import engagement_x_poster as poster
    import playwright.sync_api as api
    launches, cookies, cleaned = [], [], []
    page = SimpleNamespace()
    context = SimpleNamespace(add_cookies=cookies.append, new_page=lambda: page)
    browser = SimpleNamespace(new_context=lambda **kw: context, close=lambda: cleaned.append('browser'))
    def launch(**kwargs):
        launches.append(kwargs)
        return browser
    pw = SimpleNamespace(chromium=SimpleNamespace(launch=launch), stop=lambda: cleaned.append('pw'))
    monkeypatch.setattr(api, 'sync_playwright', lambda: SimpleNamespace(start=lambda: pw))
    monkeypatch.setattr(poster, '_load_cookies', lambda: [])
    def forbidden():
        raise RuntimeError('posting launcher forbidden')
    monkeypatch.setattr(poster, '_make_browser', forbidden)
    monkeypatch.setattr(poster, '_ensure_logged_in', lambda p: True)
    diag = {}
    assert x_ingest.ingest(include_home=False, include_mentions=False, registry={}, diagnostics=diag) == []
    assert diag['session']['status'] == 'authenticated'
    assert launches == [{'headless': True, 'timeout': 10000}]
    assert cleaned == ['browser', 'pw']



def test_partial_feed_session_still_marks_context_missing(browser_page, monkeypatch):
    source = post()
    def broken(*args, **kwargs):
        raise RuntimeError('fixture feed ended')
    monkeypatch.setattr(x_ingest, '_scrape_account', broken)
    rows, calls, diag = ingestion(monkeypatch, browser_page, {
        'https://x.com/notifications/mentions': (card(source), 200)},
        registry={'extra_accounts': ['extra']})
    assert len(rows) == 1
    assert rows[0].get('context_status', {}).get('state') == 'unknown'
    assert rows[0]['context_status']['reason'] == 'session_partial'
    assert rows[0]['thread_context'] == []


def test_source_request_budget_has_hard_ceiling(browser_page, monkeypatch):
    sources = [post(1 + i / 10) for i in range(9)]
    routes = {'https://x.com/notifications/mentions': (''.join(card(s) for s in sources), 200)}
    routes.update({f'https://x.com/i/web/status/{s["id"]}': (card(s), 200) for s in sources})
    rows, calls, diag = ingestion(monkeypatch, browser_page, routes,
                                  registry={'context_max_sources': 999})
    assert len(rows) == 9 and len(calls) == 9
    assert rows[-1]['context_status']['reason'] == 'source_budget'
    assert all(r['context_status']['complete'] is False for r in rows)
