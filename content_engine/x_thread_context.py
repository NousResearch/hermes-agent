"""Bounded read-only conversation observations, never action-source candidates.

DOM order is not proof of a reply edge. Preceding posts are labelled as possible
ancestors, quote cards as quotes; no snapshot claims complete thread coverage.
"""
from datetime import datetime, timezone
import re
import time
from urllib.parse import urlsplit


_SNAPSHOT = r"""({limit}) => {
  const quoteSelector = '[data-testid="quoteTweet"], [role="link"]';
  const visible = n => !!(n && n.getClientRects().length);
  const read = (root, quoted=false) => {
    const own = n => quoted || !n.closest(quoteSelector);
    const link = [...root.querySelectorAll('a[href*="/status/"]')]
      .find(a => a.querySelector('time') && own(a));
    const text = [...root.querySelectorAll('[data-testid="tweetText"]')].find(own);
    if (!link || !text || !visible(root)) return null;
    return {href: link.getAttribute('href'), text: text.innerText.slice(0, 4000),
      created_at: link.querySelector('time').getAttribute('datetime'),
      repost: [...root.querySelectorAll('[data-testid="socialContext"]')]
        .some(n => /reposted|retweeted/i.test(n.innerText))};
  };
  const articles = [...document.querySelectorAll('article[data-testid="tweet"]')]
    .filter(n => !n.parentElement.closest('article[data-testid="tweet"]'));
  const posts = articles.slice(0, limit).map(root => ({post: read(root),
    quotes: [...root.querySelectorAll(quoteSelector)]
      .filter(n => !n.parentElement.closest(quoteSelector)).slice(0, 4)
      .map(n => read(n, true)).filter(Boolean)}));
  const alerts = [...document.querySelectorAll('[role="alert"], [data-testid="error-detail"]')]
    .filter(visible).map(n => n.innerText).join(' ');
  return {posts, truncated: articles.length > limit, alerts,
    login: !!document.querySelector('input[autocomplete="username"]'),
    unavailable: !articles.length};
}"""


def make_readonly_browser():
    """Stock ephemeral Chromium; load existing cookies, never persist or disguise it."""
    from playwright.sync_api import sync_playwright
    from engagement_x_poster import _load_cookies
    cookies = _load_cookies()
    pw = sync_playwright().start()
    browser = None
    try:
        browser = pw.chromium.launch(headless=True, timeout=10000)
        context = browser.new_context(locale='en-GB')
        if cookies:
            context.add_cookies(cookies)
        return pw, browser, context, context.new_page()
    except Exception:
        try:
            if browser is not None:
                browser.close()
        finally:
            pw.stop()
        raise


def _record(raw, source, url, fetched_at, relation):
    from x_ingest import normalize_source
    href = raw.get('href') or ''
    match = re.fullmatch(r'/(?:[A-Za-z0-9_]{1,15}|i/web)/status/([0-9]+)', href)
    if not match or raw.get('repost'):
        return None
    # Reuse identity/time validation; the temporary origin is not emitted.
    row = normalize_source(dict(id=match[1], url='https://x.com' + href,
                                text=raw['text'], created_at=raw.get('created_at')), origin='search')
    if row is None or row['age_hours'] < 0:
        return None
    row.update(origin='conversation', source='conversation', source_id=source['id'],
               observed_url=url, fetched_at=fetched_at, relation=relation,
               relationship_verified=relation == 'quoted', action_eligible=False)
    return row


def fetch_context(page, source, *, article_limit=20):
    """One status navigation, no expansion clicks, scrolling, retries or API calls."""
    url = source['url']
    stamp = datetime.now(timezone.utc).isoformat()
    status = dict(state='unknown', complete=False, source_id=source['id'],
                  observed_url=url, fetched_at=stamp, reason='source_not_visible',
                  missing=['full ancestor chain', 'unrendered/deleted/private posts', 'unrendered quoted context'])
    records = []
    stopped = ''
    def response_seen(response):
        nonlocal stopped
        if (urlsplit(response.url).hostname in {'x.com', 'www.x.com', 'twitter.com', 'www.twitter.com'}
                and response.status in {401, 403, 429}):
            stopped = f'http_{response.status}'
    page.on('response', response_seen)
    try:
        response = page.goto(url, wait_until='domcontentloaded', timeout=10000)
        if response is not None and response.status >= 400:
            stopped = f'http_{response.status}'
        if not stopped:
            # A bounded wait also permits delayed client-side rendering, without retries.
            page.wait_for_selector('article[data-testid="tweet"], [role="alert"], input[autocomplete="username"], [data-testid="error-detail"]', timeout=5000)
            snapshot = page.evaluate(_SNAPSHOT, {'limit': article_limit})
            if snapshot['login'] or '/i/flow/login' in page.url:
                stopped = 'login_required'
            elif re.search(r'rate limit|too many requests|try again later|access denied|temporarily blocked|something went wrong', snapshot['alerts'], re.I):
                stopped = 'blocked_or_rate_limited'
            else:
                posts = snapshot['posts']
                target = next((i for i, item in enumerate(posts)
                               if item['post'] and re.search(r'/status/' + re.escape(source['id']) + r'$', item['post']['href'])
                               and not item['post']['repost']), None)
                if target is not None:
                    for item in posts[:target]:
                        if item['post']:
                            row = _record(item['post'], source, url, stamp, 'possible_ancestor')
                            if row:
                                records.append(row)
                    for item in posts[:target + 1]:
                        if not item['post'] or item['post']['repost']:
                            continue
                        parent = _record(item['post'], source, url, stamp, 'possible_ancestor')
                        if parent is None:
                            continue
                        for quote in item['quotes']:
                            row = _record(quote, source, url, stamp, 'quoted')
                            if row:
                                row['related_post_id'] = parent['id']
                                records.append(row)
                    status.update(state='partial', reason='article_budget' if snapshot['truncated'] else 'visible_snapshot_only')
    except Exception as exc:
        stopped = stopped or f'context_error:{type(exc).__name__}'
    finally:
        page.remove_listener('response', response_seen)
    if stopped:
        status.update(reason=stopped)
    seen = {source['id']}
    unique = []
    for row in records:
        if row['id'] not in seen:
            unique.append(row)
            seen.add(row['id'])
    status['context_ids'] = [row['id'] for row in unique]
    return unique, status, stopped


def enrich_sources(page, rows, *, max_sources=8, article_limit=20):
    """Global request budget and stop latch; rows retain their discovery provenance."""
    from x_ingest import source_freshness_issues
    max_sources = max(0, min(int(max_sources), 8))
    article_limit = max(1, min(int(article_limit), 40))
    enriched, stopped, requests = [], '', 0
    for row in rows:
        status = dict(state='unknown', complete=False, source_id=row['id'], context_ids=[],
                      reason='stopped:' + stopped if stopped else 'source_budget',
                      missing=['conversation context not fetched'])
        records = []
        if not stopped and requests < max_sources and not source_freshness_issues(row):
            if requests:
                time.sleep(1.2)
            records, status, stopped = fetch_context(page, row, article_limit=article_limit)
            requests += 1
        enriched.append({**row, 'thread_context': records, 'context_status': status})
    return enriched
