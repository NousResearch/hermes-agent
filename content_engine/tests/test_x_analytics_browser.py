"""Contract and real headless DOM tests; never contact X in tests."""
import x_analytics_browser as collector


def test_normalize_own_identity_metrics_dedupe_and_bounds():
    row = dict(url='https://x.com/Owner/status/1234567890123456789', author='Owner',
               created_at='2026-09-01T10:00:00Z', text='A post', action=None,
               metrics={'likes': '12', 'replies': '0', 'reposts': '', 'views': '1.2K'})
    foreign = dict(row, author='Other')
    snapshot = dict(source='https://x.com/Owner', posts=[foreign, row, row], followers='1.2K')
    result = collector.normalize_snapshot('Owner', snapshot, '2026-09-12T10:00:00Z', max_posts=1)
    assert len(result['posts']) == 1
    post = result['posts'][0]
    assert post['metrics'] == dict(likes=12, replies=0, reposts=None, views=None)
    assert post['topic'] is None and post['action'] is None
    assert post['provenance']['kind'] == 'browser'
    assert len(post['provenance']['sha256']) == 64
    assert result['followers'][0]['count'] is None
    assert result['capabilities']['account_analytics']['status'] == 'not_checked'



import hashlib
import json
from contextlib import contextmanager

import pytest
from playwright.sync_api import sync_playwright


HTML = """<html><body>
<a data-testid="AppTabBar_Profile_Link" href="/Owner">Profile</a>
<main><a href="/Other/followers">999 Followers</a>
<a href="/Owner/verified_followers">123 Followers</a>
<div style="display:none">Rate limit exceeded</div>
<article data-testid="tweet">
<div data-testid="User-Name"><a href="/Owner">Owner</a>
<a href="https://x.com/Owner/status/1234567890123456789"><time datetime="2026-09-01T10:00:00Z">Sep 1</time></a></div>
<div data-testid="tweetText">A post about rate limit exceeded</div>
<div role="link"><a href="https://x.com/Other/status/999"><time datetime="2020-01-01T00:00:00Z">quote</time></a></div>
<button data-testid="like" aria-label="12 Likes. Like">12</button>
<button data-testid="reply" aria-label="0 Replies. Reply"></button>
<button data-testid="retweet" style="display:none" aria-label="99 Reposts"></button>
<a href="/Owner/status/1234567890123456789/analytics" aria-label="1,203 Views. View post analytics">1,203</a>
</article></main></body></html>"""


@pytest.mark.parametrize('scenario', ['ok', 'rate', 'block', 'wrong_identity', 'paywall', 'entitled'])
def test_real_headless_collector_offline_routes_and_stop(monkeypatch, tmp_path, scenario):
    """Exercise real Chromium + real collector; every request is locally fulfilled."""
    import playwright.sync_api
    requests = []
    @contextmanager
    def local_playwright():
        with sync_playwright() as pw:
            original_launch = pw.chromium.launch
            def launch(**kwargs):
                assert kwargs['headless'] is True
                browser = original_launch(**kwargs)
                original_context = browser.new_context
                def new_context(**opts):
                    context = original_context(**opts)
                    def respond(route):
                        requests.append((route.request.method, route.request.url))
                        body = HTML
                        status = 200
                        if scenario == 'rate':
                            status = 429
                        if scenario == 'block':
                            body = '<main>Verify you are human</main>'
                        if scenario == 'wrong_identity':
                            body = HTML.replace('href="/Owner">Profile', 'href="/Other">Profile')
                        if route.request.url.endswith('/i/account_analytics'):
                            body = ('<a data-testid="AppTabBar_Profile_Link" href="/Owner">Profile</a><main>Account analytics<br>Last 28 days<br>Impressions<br>1,203<br>Engagements<br>42</main>' if scenario == 'entitled' else '<main>Subscribe to unlock analytics</main>')
                        route.fulfill(status=status, content_type='text/html', body=body)
                    context.route('**/*', respond)
                    return context
                monkeypatch.setattr(browser, 'new_context', new_context)
                return browser
            monkeypatch.setattr(pw.chromium, 'launch', launch)
            yield pw
    monkeypatch.setattr(playwright.sync_api, 'sync_playwright', local_playwright)
    cookies = tmp_path / 'cookies.json'
    cookies.write_text('[]')
    result = collector.collect_own_analytics('Owner', cookies_path=cookies,
        max_posts=3, max_scrolls=0, timeout_seconds=30, check_account_analytics=scenario in ('paywall','entitled'))
    assert cookies.read_text() == '[]'
    assert all(method == 'GET' for method, _ in requests)
    if scenario in ('rate', 'block', 'wrong_identity'):
        assert result['posts'] == []
        assert result['capabilities']['collection']['status'] == 'blocked'
        assert requests == [('GET', 'https://x.com/Owner')]
    else:
        assert len(result['posts']) == 1
        post = result['posts'][0]
        assert post['id'] == '1234567890123456789'  # never the quote ID
        assert post['metrics'] == dict(likes=12, replies=0, reposts=None, views=1203)
        assert result['followers'][0]['count'] == 123
        prov = post['provenance']
        raw = json.dumps(prov['observation'], sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()
        assert prov['sha256'] == hashlib.sha256(raw).hexdigest()
        if scenario == 'entitled':
            capability=result['capabilities']['account_analytics']
            assert capability['status']=='available_observed'
            assert capability['metrics']['impressions']==1203
            assert capability['period']=='Last 28 days'
        if scenario == 'paywall':
            assert result['capabilities']['account_analytics']['status'] == 'unavailable'
            assert len(requests) == 2


@pytest.mark.parametrize('kwargs', [dict(account='@Owner'), dict(max_posts=0),
    dict(max_posts=101), dict(max_scrolls=6), dict(timeout_seconds=float('nan'))])
def test_invalid_bounds_before_browser(tmp_path, kwargs):
    args = dict(account='Owner', cookies_path=tmp_path / 'absent.json')
    args.update(kwargs)
    with pytest.raises(ValueError):
        collector.collect_own_analytics(**args)


def test_reject_foreign_future_missing_and_malformed_identity():
    valid = dict(url='https://x.com/Owner/status/1234567890123456789', author='Owner',
        created_at='2026-09-01T00:00:00Z', text='own', metrics={})
    invalid = [dict(valid, author='Other'), dict(valid, url=valid['url']+'?bad=1'),
        dict(valid, created_at='2030-01-01T00:00:00Z'), dict(valid, created_at=None),
        dict(valid, url='https://x.com/Other/status/1234567890123456789')]
    result = collector.normalize_snapshot('Owner', dict(source='https://x.com/Owner',
        posts=invalid+[valid]), '2026-09-12T10:00:00Z')
    assert len(result['posts']) == 1
    assert all(x is None for x in result['posts'][0]['metrics'].values())
