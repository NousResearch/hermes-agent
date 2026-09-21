"""Playwright-based X.com web poster for engagement scout.

Bypasses X API Basic tier restrictions by driving X.com's web UI
via Playwright in headless mode, using cookies from your real
browser session.

SETUP (one-time):
  1. Open X.com in your LOCAL browser (not the VPS)
  2. Open DevTools → Application → Cookies → x.com
  3. Click "Export" (or use "Copy as JSON" button)
  4. Paste the JSON into a file on the VPS:
       cat > ~/.x-browser-state/cookies.json
       [paste the JSON from step 3, then Ctrl+D]

USAGE:
  python engagement_x_poster.py cookies           # Set cookies from DevTools export
  python engagement_x_poster.py qt <id> <text>     # Quote tweet
  python engagement_x_poster.py reply <id> <text>  # Reply
  python engagement_x_poster.py fetch <account>    # Fetch tweets w/ zero API cost
  python engagement_x_poster.py status             # Check auth status
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

STORAGE_DIR = Path.home() / ".x-browser-state"
COOKIES_FILE = STORAGE_DIR / "cookies.json"


def _load_cookies() -> list[dict]:
    """Load cookies from the cookies file.

    Accepts both Playwright JSON format (list of {name, value, domain, path})
    and the X.com DevTools export format (Netscape/standard cookie format).
    """
    if not COOKIES_FILE.exists():
        return []

    try:
        raw = COOKIES_FILE.read_text().strip()
        data = json.loads(raw)

        # If it's a plain list, assume it's already in Playwright-compatible format
        if isinstance(data, list):
            # Ensure each cookie has required domain/path fields
            for c in data:
                if "domain" not in c or not c["domain"]:
                    c["domain"] = ".x.com"
                if "path" not in c or not c["path"]:
                    c["path"] = "/"
            return data

        # If it's a dict with a "cookies" key, extract that
        if isinstance(data, dict) and "cookies" in data:
            return data["cookies"]

        # If it's a dict with individual cookie objects as values
        if isinstance(data, dict):
            cookies = []
            for key, val in data.items():
                if isinstance(val, dict) and "name" in val:
                    cookies.append(val)
                elif isinstance(val, dict) and "value" in val:
                    cookies.append({
                        "name": key,
                        "value": val.get("value", ""),
                        "domain": val.get("domain", ".x.com"),
                        "path": val.get("path", "/"),
                    })
            if cookies:
                return cookies

    except (json.JSONDecodeError, OSError) as exc:
        print(f"[xposter] ❌ Could not parse cookies: {exc}")

    return []


def _make_browser():
    """Create a headless Playwright browser with loaded cookies."""
    from playwright.sync_api import sync_playwright
    from playwright_stealth import Stealth

    pw = sync_playwright().start()
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)

    browser = pw.chromium.launch(
        headless=True,
        args=[
            "--no-sandbox",
            "--disable-blink-features=AutomationControlled",
            "--disable-web-security",
            "--disable-features=IsolateOrigins,site-per-process",
        ],
    )

    context = browser.new_context(
        viewport={"width": 1280, "height": 720},
        user_agent=(
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
        ),
        locale="en-GB",
        timezone_id="Europe/London",
    )

    # Add cookies from saved file
    cookies = _load_cookies()
    if cookies:
        context.add_cookies(cookies)
        print(f"[xposter] Loaded {len(cookies)} cookies")
    else:
        print("[xposter] ⚠️  No cookies found. Authentication will fail.")

    page = context.new_page()
    Stealth().apply_stealth_sync(page)
    return pw, browser, context, page


def _ensure_logged_in(page) -> bool:
    """Check if logged in by navigating to home. Return True if ok."""
    page.goto("https://x.com/home", wait_until="domcontentloaded", timeout=30000)
    time.sleep(3)
    if "login" in page.url.lower():
        print("[xposter] ❌ Not logged in. Cookies may be expired.")
        return False
    return True


def _click_with_retry(page, selectors: list[str], timeout: int = 3000) -> object | None:
    """Try multiple selectors until one matches. Returns the locator or None."""
    for sel in selectors:
        try:
            if sel.startswith("//"):
                btn = page.locator("xpath=" + sel).first
            else:
                btn = page.locator(sel).first
            if btn.is_visible(timeout=timeout):
                return btn
        except Exception:
            continue
    return None


def _type_text(page, text: str):
    """Type text into the active composer with realistic delay."""
    selectors = [
        'div[aria-label="Post text"]',
        '[data-testid="tweetTextarea_0"]',
        'div[role="textbox"]',
        'div[contenteditable="true"]',
    ]
    ta = _click_with_retry(page, selectors)
    if not ta:
        print("[xposter] ❌ Could not find text composer")
        return False
    ta.click()
    time.sleep(0.3)
    page.keyboard.type(text, delay=15)
    time.sleep(0.3)
    return True


def _click_post_button(page) -> bool:
    """Click the final post/reply button."""
    selectors = [
        'button[data-testid="tweetButton"]',
        'button:has-text("Post")',
        'button:has-text("Reply")',
        'div[data-testid="tweetButton"]',
    ]
    btn = _click_with_retry(page, selectors)
    if not btn:
        return False
    btn.click()
    time.sleep(3)
    return True


# ── Public API ──


def quote_tweet(tweet_id: str, text: str) -> bool:
    """Quote tweet via X.com web UI using saved cookies."""
    import playwright_stealth

    print(f"[xposter] Quote tweeting {tweet_id}...")
    pw, browser, context, page = _make_browser()

    try:
        if not _ensure_logged_in(page):
            browser.close()
            pw.stop()
            return False

        tweet_url = f"https://x.com/i/web/status/{tweet_id}"
        page.goto(tweet_url, wait_until="networkidle", timeout=30000)
        time.sleep(2)

        # Click Repost button
        repost = _click_with_retry(page, [
            'button[aria-label="Repost"]',
            'button[data-testid="retweet"]',
            '//div[@role="button" and .//span[text()="Repost"]]',
            'button:has-text("Repost")',
        ])
        if not repost:
            print("[xposter] ❌ Could not find Repost button")
            browser.close()
            pw.stop()
            return False
        repost.click()
        time.sleep(1)

        # Click "Quote" from dropdown
        quote = _click_with_retry(page, [
            '//span[text()="Quote"]/ancestor::div[@role="menuitem"]',
            'div[role="menuitem"]:has-text("Quote")',
            'a:has-text("Quote")',
            '[data-testid="quote"]',
        ])
        if not quote:
            print("[xposter] ❌ Could not find Quote option")
            browser.close()
            pw.stop()
            return False
        quote.click()
        time.sleep(1.5)

        # Type quote text
        if not _type_text(page, text):
            browser.close()
            pw.stop()
            return False

        # Post
        if not _click_post_button(page):
            print("[xposter] ❌ Could not find Post button")
            browser.close()
            pw.stop()
            return False

        print(f"[xposter] ✅ Quote tweet posted for {tweet_id}")
        browser.close()
        pw.stop()
        return True

    except Exception as exc:
        print(f"[xposter] ❌ Error: {exc}")
        browser.close()
        pw.stop()
        return False


def reply_to_tweet(tweet_id: str, text: str) -> bool:
    """Reply to a tweet via X.com web UI using saved cookies."""
    import playwright_stealth

    print(f"[xposter] Replying to {tweet_id}...")
    pw, browser, context, page = _make_browser()

    try:
        if not _ensure_logged_in(page):
            browser.close()
            pw.stop()
            return False

        tweet_url = f"https://x.com/i/web/status/{tweet_id}"
        page.goto(tweet_url, wait_until="networkidle", timeout=30000)
        time.sleep(2)

        # Click Reply button
        reply = _click_with_retry(page, [
            'button[aria-label="Reply"]',
            'button[data-testid="reply"]',
            '//div[@role="button" and .//span[text()="Reply"]]',
            'button:has-text("Reply")',
        ])
        if not reply:
            print("[xposter] ❌ Could not find Reply button")
            browser.close()
            pw.stop()
            return False
        reply.click()
        time.sleep(1.5)

        # Type reply text
        if not _type_text(page, text):
            browser.close()
            pw.stop()
            return False

        # Post reply
        if not _click_post_button(page):
            print("[xposter] ❌ Could not find post button")
            browser.close()
            pw.stop()
            return False

        print(f"[xposter] ✅ Reply posted for {tweet_id}")
        browser.close()
        pw.stop()
        return True

    except Exception as exc:
        print(f"[xposter] ❌ Error: {exc}")
        browser.close()
        pw.stop()
        return False


def _extract_tweets_from_page(page, account: str, limit: int = 15) -> list[dict]:
    """Extract tweet data from a loaded X.com account page.

    Navigates to https://x.com/<account>, waits for tweets to render,
    scrapes tweet ID, text, and engagement metrics from the DOM.

    Returns the same format as fetch_recent_tweets() for compatibility.
    """
    tweet_url = f"https://x.com/search?q=from%3A{account}&src=typed_query&f=live"
    print(f"[xposter]  Search @{account}...")
    page.goto(tweet_url, wait_until="domcontentloaded", timeout=20000)
    time.sleep(3)

    # Scroll to trigger lazy loading
    for scroll in range(2):
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        time.sleep(1)
        current = page.locator('article[data-testid="tweet"]').count()
        if current >= limit + 5:
            break

    # Wait for tweet articles
    try:
        page.wait_for_selector('article[data-testid="tweet"]', timeout=10000)
    except Exception:
        print(f"[xposter]  No tweets found for @{account}")
        return []

    articles = page.locator('article[data-testid="tweet"]')
    count = min(articles.count(), limit + 5)
    tweets = []

    for i in range(count):
        article = articles.nth(i)
        try:
            # Extract tweet ID from permalink
            links = article.locator('a[href*="/status/"]')
            href = links.first.get_attribute("href") if links.count() > 0 else None
            if not href or "/status/" not in href:
                continue
            tweet_id = href.split("/status/")[-1].split("?")[0]

            # Extract text
            text_el = article.locator('div[data-testid="tweetText"]').first
            tweet_text = text_el.inner_text() if text_el.count() > 0 else ""

            if not tweet_text.strip():
                continue

            # Extract engagement from aria-labels
            def _get_count(selector: str) -> int:
                btn = article.locator(selector)
                if btn.count() == 0:
                    return 0
                label = btn.first.get_attribute("aria-label") or ""
                # Extract first number from the label
                import re
                nums = re.findall(r'([\d,]+)', label.replace(",", ""))
                return int(nums[0]) if nums else 0

            likes = _get_count('[data-testid="like"]')
            replies = _get_count('[data-testid="reply"]')
            retweets_from_label = _get_count('[data-testid="retweet"]')

            tweets.append({
                "id": tweet_id,
                "text": tweet_text,
                "author": account,
                "author_name": account,
                "likes": likes,
                "replies": replies,
                "retweets": retweets_from_label,
                "engagement": likes + replies + retweets_from_label,
                "created_at": "",
                "url": f"https://x.com/{account}/status/{tweet_id}",
            })

            if len(tweets) >= limit:
                break

        except Exception as exc:
            print(f"[xposter]  Skipping tweet {i}: {exc}")
            continue

    print(f"[xposter]  Found {len(tweets)} tweets for @{account}")
    return tweets


def fetch_tweets(account: str, limit: int = 15) -> list[dict]:
    """Public API: fetch recent tweets from an account using browser scraping.

    Zero API cost — works entirely through X.com's web UI.
    Compatible return format with engagement_suggester.fetch_recent_tweets.

    Each tweet dict: id, text, author, author_name, likes, replies,
    retweets, engagement, created_at, url.
    """
    pw, browser, context, page = _make_browser()
    try:
        if not _ensure_logged_in(page):
            browser.close()
            pw.stop()
            return []
        tweets = _extract_tweets_from_page(page, account, limit)
        browser.close()
        pw.stop()
        return tweets
    except Exception as exc:
        print(f"[xposter] ❌ fetch_tweets error for @{account}: {exc}")
        browser.close()
        pw.stop()
        return []


def fetch_tweets_batch(accounts: list[str], limit: int = 15) -> dict[str, list[dict]]:
    """Fetch tweets for multiple accounts in a single browser session.

    Opens one browser, navigates through each account, returns a dict
    mapping account -> tweets. Much faster and more reliable than calling
    fetch_tweets() per account.

    Returns dict like: {"levelsio": [...], "shadcn": [...], ...}
    """
    pw, browser, context, page = _make_browser()
    result = {}
    try:
        if not _ensure_logged_in(page):
            browser.close()
            pw.stop()
            return result

        for account in accounts:
            print(f"[xposter]  @{account}...", end=" ", flush=True)
            try:
                # Navigate and force a fresh page load
                page.goto(f"https://x.com/search?q=from%3A{account}&src=typed_query&f=live",
                          wait_until="domcontentloaded", timeout=20000)
                time.sleep(3)
                tweets = _extract_tweets_from_page(page, account, limit)
                result[account] = tweets
                print(f"{len(tweets)} tweets")
                # Navigate away between accounts to reset page state
                page.goto("https://x.com/home", wait_until="domcontentloaded", timeout=10000)
                time.sleep(1)
            except Exception as exc:
                print(f"error: {exc}")
                result[account] = []
                # Don't stop - close browser and return partial results
                if "EPIPE" in str(exc) or "Target closed" in str(exc):
                    print(f"[xposter]  Browser disconnected, returning partial results")
                    break

        browser.close()
        pw.stop()
        return result
    except Exception as exc:
        print(f"[xposter] ❌ batch error: {exc}")
        browser.close()
        pw.stop()
        return result


def set_cookies() -> bool:
    """Set X.com cookies from pasted DevTools JSON."""
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    print("[xposter] Paste the cookies JSON from DevTools, then press Ctrl+D:")
    raw = sys.stdin.read().strip()
    if not raw:
        print("[xposter] ❌ No input received.")
        return False

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"[xposter] ❌ Invalid JSON: {e}")
        return False

    COOKIES_FILE.write_text(json.dumps(data, indent=2))
    count = len(data) if isinstance(data, list) else len(data.get("cookies", data)) if isinstance(data, dict) else 1
    print(f"[xposter] ✅ Saved {count} cookies to {COOKIES_FILE}")
    print("[xposter] ✅ Testing cookies by checking x.com/home...")

    # Quick test
    pw, browser, context, page = _make_browser()
    try:
        page.goto("https://x.com/home", wait_until="domcontentloaded", timeout=30000)
        time.sleep(2)
        if "login" in page.url.lower():
            print("[xposter] ❌ Cookies invalid or expired. Re-export fresh ones.")
            browser.close()
            pw.stop()
            return False
        # Try to extract username from page
        print(f"[xposter] ✅ Session valid at {page.url}")
        browser.close()
        pw.stop()
        return True
    except Exception as exc:
        print(f"[xposter] ❌ Error testing cookies: {exc}")
        browser.close()
        pw.stop()
        return False


def is_authenticated() -> bool:
    """Quick check if we have saved cookies."""
    return COOKIES_FILE.exists()


# ── CLI ──


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        return

    command = sys.argv[1]

    if command == "cookies":
        ok = set_cookies()
        print("✅ Cookies saved." if ok else "❌ Failed.")
    elif command == "qt" and len(sys.argv) >= 4:
        ok = quote_tweet(sys.argv[2], sys.argv[3])
        print("✅ Quote posted." if ok else "❌ Quote failed.")
    elif command == "reply" and len(sys.argv) >= 4:
        ok = reply_to_tweet(sys.argv[2], sys.argv[3])
        print("✅ Reply posted." if ok else "❌ Reply failed.")
    elif command == "fetch" and len(sys.argv) >= 3:
        tweets = fetch_tweets(sys.argv[2], limit=int(sys.argv[3]) if len(sys.argv) >= 4 else 10)
        print(f"✅ Fetched {len(tweets)} tweets for @{sys.argv[2]}")
    elif command == "status":
        print(f"Cookies file exists: {is_authenticated()}")
        if is_authenticated():
            c = json.loads(COOKIES_FILE.read_text())
            print(f"  {len(c) if isinstance(c, list) else '?'} cookies")
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
