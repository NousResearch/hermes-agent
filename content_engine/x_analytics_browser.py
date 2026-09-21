"""Bounded, stock-headless X DOM observations. No API client or posting actions.

CLI requires explicit cookie and output paths. Cookie state is read, never saved.
Abbreviated/missing counts stay null; raw DOM observations remain in provenance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path


# Only rendered DOM and accessibility labels; never scripts or network payloads.
DOM_SNAPSHOT = r"""() => {
 const visible = e => !!e && e.getClientRects().length > 0 &&
   getComputedStyle(e).visibility !== 'hidden' && getComputedStyle(e).display !== 'none';
 const txt = e => visible(e) ? e.innerText : null;
 const main = document.querySelector('main') || document.body;
 const posts = Array.from(main.querySelectorAll('article[data-testid="tweet"]')).filter(visible).slice(0,100).map(a => {
   const user = a.querySelector('[data-testid="User-Name"]');
   const t = user?.querySelector('time');
   const link = t?.closest('a');
   const author = Array.from(user?.querySelectorAll('a') || []).map(e => e.getAttribute('href'))
     .find(h => /^\/[A-Za-z0-9_]{1,15}$/.test(h || ''))?.slice(1);
   const metrics = {};
   for (const [key, sel] of Object.entries({likes:'[data-testid="like"], [data-testid="unlike"]',
     replies:'[data-testid="reply"]', reposts:'[data-testid="retweet"], [data-testid="unretweet"]',
     views:'a[href$="/analytics"]'})) {
       const e = Array.from(a.querySelectorAll(sel)).find(visible);
       metrics[key] = e ? (e.getAttribute('aria-label') || e.innerText) : null;
   }
   const textEl = a.querySelector('[data-testid="tweetText"]');
   const users = a.querySelectorAll('[data-testid="User-Name"]');
   const quotedOnly = users.length > 1 && textEl && (users[1].compareDocumentPosition(textEl) & Node.DOCUMENT_POSITION_FOLLOWING);
   const text = quotedOnly ? null : txt(textEl);
   return {url:link?.href, author, created_at:visible(t) ? t.getAttribute('datetime') : null,
     text, metrics, action:null, truncated:!!a.querySelector('[data-testid="tweet-text-show-more-link"]')};
 });
 const profile = document.querySelector('a[data-testid="AppTabBar_Profile_Link"]');
 const followers = Array.from(main.querySelectorAll('a')).find(e => visible(e) &&
   [profile?.getAttribute('href') + '/verified_followers', profile?.getAttribute('href') + '/followers']
     .some(h => h.toLowerCase() === (e.getAttribute('href') || '').toLowerCase()));
 // Exclude post content so a post discussing rate limits does not become a block.
 const notices = [];
 const walk = e => {
   if (!visible(e) || e.tagName === 'ARTICLE') return;
   for (const n of e.childNodes) {
     if (n.nodeType === Node.TEXT_NODE) notices.push(n.textContent);
     else if (n.nodeType === Node.ELEMENT_NODE) walk(n);
   }
 };
 walk(main);
 return {source:location.href, profile:visible(profile) ? profile.getAttribute('href') : null, posts,
   followers:txt(followers), notice:notices.join(' ').slice(0,20000),
   visible_text:main.innerText.slice(0,20000)};
}"""


def _utc(value):
    if not isinstance(value, str):
        raise ValueError('timestamp must be UTC')
    dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if dt.tzinfo is None or dt.utcoffset().total_seconds() != 0:
        raise ValueError('timestamp must be UTC')
    return dt


def _account(account):
    if not isinstance(account, str) or not re.fullmatch(r'[A-Za-z0-9_]{1,15}', account):
        raise ValueError('account must be a literal X handle without @')
    return account


def _count(raw):
    if raw is None:
        return None
    # Full match prevents rounded K/M counts, decimals, negative values or locale ambiguity.
    m = re.fullmatch(r'\s*(\d+|\d{1,3}(?:,\d{3})+)(?:\s+(?:Likes?|Replies|Reply|Reposts?|Retweets?|Views?|Followers?)(?:\. .*)?)?\s*', str(raw), re.I)
    return int(m[1].replace(',', '')) if m else None


def _provenance(source, observation):
    encoded = json.dumps(observation, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()
    return {'kind': 'browser', 'source': source, 'sha256': hashlib.sha256(encoded).hexdigest(),
            'observation': observation}


def normalize_snapshot(account, snapshot, observed_at, max_posts=20):
    """Validate an extracted DOM snapshot; unknown classification is not invented."""
    _account(account)
    observed = _utc(observed_at)
    if type(max_posts) is not int or not 1 <= max_posts <= 100:
        raise ValueError('max_posts must be 1..100')
    result = {'account': account, 'posts': [], 'followers': [], 'capabilities': {
        'account_analytics': {'status': 'not_checked', 'reason': 'Only profile DOM observed; no entitlement assumption.'}}}
    seen = set()
    for row in snapshot.get('posts', [])[:100]:
        url = row.get('url') or ''
        match = re.fullmatch(r'https://(?:x\.com|twitter\.com)/([A-Za-z0-9_]{1,15})/status/([0-9]{1,20})', url)
        if not match or match[1].lower() != account.lower() or str(row.get('author', '')).lower() != account.lower():
            continue
        try:
            created = _utc(row.get('created_at'))
        except (ValueError, TypeError):
            continue
        if created > observed or not row.get('text') or match[2] in seen:
            continue
        seen.add(match[2])
        raw_metrics = row.get('metrics') or {}
        result['posts'].append({'id': match[2], 'url': url, 'author': row['author'],
            'created_at': created.isoformat().replace('+00:00', 'Z'), 'observed_at': observed_at,
            'text': row['text'], 'action': None, 'topic': None, 'truncated': bool(row.get('truncated')),
            'metrics': {k: _count(raw_metrics.get(k)) for k in ('likes', 'replies', 'reposts', 'views')},
            'provenance': _provenance(snapshot['source'], row)})
        if len(result['posts']) >= max_posts:
            break
    if snapshot.get('followers') is not None:
        observation = {'account': account, 'observed_at': observed_at, 'display': snapshot['followers']}
        result['followers'].append({'account': account, 'observed_at': observed_at,
            'count': _count(snapshot['followers']), 'provenance': _provenance(snapshot['source'], observation)})
    return result


def _blocked(snapshot):
    url = snapshot.get('source', '').lower()
    if any(x in url for x in ('/login', '/i/flow', '/account/access')):
        return 'Authentication or account access challenge; stopped without retry.'
    notice = snapshot.get('notice', '').lower()
    if any(x in notice for x in ('rate limit exceeded', 'too many requests', 'verify you are human',
                                 'unusual activity', 'temporarily restricted', 'something went wrong',
                                 'try reloading', 'account suspended')):
        return 'Visible block, challenge, rate limit or error; stopped without retry.'
    return None


def _cookies(path):
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict):
        data = data.get('cookies', list(data.values()))
    if not isinstance(data, list):
        raise ValueError('Cookie file must contain a cookie list')
    allowed = {'name', 'value', 'domain', 'path', 'expires', 'httpOnly', 'secure', 'sameSite'}
    result = []
    for cookie in data:
        if not isinstance(cookie, dict):
            raise ValueError('Invalid cookie record')
        domain = cookie.get('domain', '.x.com')
        if domain not in ('x.com', '.x.com', 'twitter.com', '.twitter.com'):
            continue
        c = {k: v for k, v in cookie.items() if k in allowed}
        c.update(domain=domain, path=cookie.get('path', '/'))
        if c.get('sameSite') not in ('Strict', 'Lax', 'None'):
            c.pop('sameSite', None)
        result.append(c)
    return result


def collect_own_analytics(account, *, cookies_path, max_posts=20, max_scrolls=2,
                          timeout_seconds=60, check_account_analytics=False, include_replies=False):
    """One ephemeral browser, profile + bounded scrolls, optional analytics-page check.

    Does not click, type, like, follow, post, subscribe, replay requests or save cookies.
    Responses are inspected for HTTP status only, not content. No retries after a block.
    """
    _account(account)
    if type(max_posts) is not int or not 1 <= max_posts <= 100:
        raise ValueError('max_posts must be 1..100')
    if type(max_scrolls) is not int or not 0 <= max_scrolls <= 5:
        raise ValueError('max_scrolls must be 0..5')
    if type(timeout_seconds) not in (int, float) or not 5 <= timeout_seconds <= 120:
        raise ValueError('timeout_seconds must be 5..120')
    result = {'account': account, 'posts': [], 'followers': [], 'capabilities': {
        'account_analytics': {'status': 'not_checked', 'reason': 'Not visited; no entitlement assumption.'},
        'collection': {'status': 'partial', 'reason': 'Bounded profile sample; not full history.'}}}
    started = time.monotonic()
    stop = []
    def remaining():
        seconds = timeout_seconds - (time.monotonic() - started)
        if seconds <= 0:
            raise TimeoutError('Collection deadline')
        return min(15000, int(seconds * 1000))
    def response_status(response):
        if response.status in (401, 403, 429) and re.match(r'https://(?:[^/]+\.)?(?:x\.com|twitter\.com)/', response.url):
            stop.append(f'HTTP {response.status}; stopped without retry.')
    try:
        cookies = _cookies(cookies_path)
        from playwright.sync_api import sync_playwright
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True, timeout=remaining())
            try:
                context = browser.new_context(locale='en-US', viewport={'width': 1280, 'height': 900})
                context.add_cookies(cookies)
                page = context.new_page()
                page.on('response', response_status)
                own_url = f'https://x.com/{account}' + ('/with_replies' if include_replies else '')
                page.goto(own_url, wait_until='domcontentloaded', timeout=remaining())
                seen = set()
                for step in range(max_scrolls + 1):
                    if stop:
                        break
                    # Bounded readiness polling of the existing page, never a network retry.
                    for _ in range(15):
                        remaining()
                        snapshot = page.evaluate(DOM_SNAPSHOT)
                        reason = _blocked(snapshot)
                        if reason:
                            stop.append(reason)
                        if stop or snapshot['posts']:
                            break
                        page.wait_for_timeout(min(500, remaining()))
                    if stop:
                        break
                    if str(snapshot.get('profile', '')).lower() != f'/{account}'.lower():
                        stop.append('Own logged-in profile identity could not be verified; no rows accepted.')
                        break
                    at = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
                    batch = normalize_snapshot(account, snapshot, at, max_posts)
                    if not result['followers']:
                        result['followers'] = batch['followers']
                    for row in batch['posts']:
                        if row['id'] not in seen and len(result['posts']) < max_posts:
                            seen.add(row['id'])
                            result['posts'].append(row)
                    if len(result['posts']) >= max_posts or step == max_scrolls or not snapshot['posts']:
                        break
                    page.mouse.wheel(0, 750)
                    page.wait_for_timeout(min(1500, remaining()))
                if check_account_analytics and not stop:
                    page.goto('https://x.com/i/account_analytics', wait_until='domcontentloaded', timeout=remaining())
                    page.wait_for_timeout(min(2000, remaining()))
                    analytics = page.evaluate(DOM_SNAPSHOT)
                    reason = _blocked(analytics)
                    if reason:
                        stop.append(reason)
                    visible = analytics.get('visible_text', '').lower()
                    paywall = any(x in visible for x in ('subscribe', 'upgrade to premium', 'get premium', 'unlock analytics'))
                    result['capabilities']['account_analytics'] = {
                        'status': 'unavailable' if stop or paywall else 'unverified',
                        'reason': stop[0] if stop else ('Visible subscription gate; no access attempted.' if paywall else
                            'Page visited; dashboard metrics/entitlement not reliably verified. Use owner CSV/manual export.'),
                        'provenance': _provenance(analytics['source'], {'visible_text': analytics['visible_text']})}
                    # Recognize only a visible own dashboard with an explicit period and
                    # exact labeled totals. These are account totals, never post views.
                    text = analytics.get('visible_text', '')
                    period = re.search(r'(?m)^Last (?:7|28|30|90) days$', text)
                    metrics = {}
                    for label in ('Impressions', 'Engagements', 'Profile visits', 'New followers'):
                        match = re.search(r'(?m)^'+re.escape(label)+r'\n([0-9,]+)$', text)
                        metrics[label.lower().replace(' ', '_')] = _count(match[1]) if match else None
                    if not stop and not paywall and analytics.get('profile', '').lower() == f'/{account}'.lower() and 'account analytics' in visible and period and any(v is not None for v in metrics.values()):
                        result['capabilities']['account_analytics'].update(
                            status='available_observed', reason='Visible own account dashboard; exact labeled counters only. No entitlement purchase or inference.',
                            observed_at=datetime.now(timezone.utc).isoformat(), period=period[0], metrics=metrics,
                            definitions='Account-level labeled totals over displayed period; not post-view denominators. No engagement-rate calculation.')
            finally:
                browser.close()
        if stop:
            result['capabilities']['collection'] = {'status': 'blocked', 'reason': stop[0]}
            if result['capabilities']['account_analytics']['status'] == 'not_checked':
                result['capabilities']['account_analytics']['reason'] = 'Not visited because collection stopped: ' + stop[0]
    except Exception as exc:
        # Exception strings may contain cookie values, URLs or browser dumps: type only.
        result['capabilities']['collection'] = {'status': 'unavailable', 'reason': type(exc).__name__}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--account', required=True)
    parser.add_argument('--cookies', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--max-posts', type=int, default=20)
    parser.add_argument('--max-scrolls', type=int, default=2)
    parser.add_argument('--timeout-seconds', type=float, default=60)
    parser.add_argument('--check-account-analytics', action='store_true')
    parser.add_argument('--include-replies', action='store_true')
    args = parser.parse_args()
    result = collect_own_analytics(args.account, cookies_path=args.cookies, max_posts=args.max_posts,
        max_scrolls=args.max_scrolls, timeout_seconds=args.timeout_seconds,
        check_account_analytics=args.check_account_analytics, include_replies=args.include_replies)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps({'posts': len(result['posts']), 'followers': len(result['followers']),
                      'capabilities': {k: {a: b for a, b in v.items() if a != 'provenance'}
                                       for k, v in result['capabilities'].items()}}))
    return 0 if result['posts'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
