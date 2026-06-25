"""Playwright browser crawl for SitDeck (authenticated dashboard OSINT)."""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home

from .credentials import (
    DEFAULT_APP_URL,
    DEFAULT_LOGIN_URL,
    GLOBAL_PULSE_URL,
    get_credentials,
    redact_secrets,
)

BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)
# SPAs like SitDeck rarely reach networkidle; domcontentloaded + selector waits are enough.
DEFAULT_GOTO_WAIT = "domcontentloaded"
POST_LOGIN_LOAD_STATE = "domcontentloaded"
STATE_DIR = get_hermes_home() / "sitdeck-osint"
LAST_CRAWL_PATH = STATE_DIR / "last_crawl.json"
STORAGE_STATE_PATH = STATE_DIR / "storage_state.json"

# JSON API paths observed on SitDeck SPA (best-effort; may change upstream).
_API_HINTS = (
    "/api/",
    "/v1/",
    "briefing",
    "pulse",
    "widget",
    "deck",
    "feed",
    "alert",
    "situation",
)


class _MetaParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.title = ""
        self.meta: dict[str, str] = {}
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = {k: (v or "") for k, v in attrs}
        if tag == "title":
            self._in_title = True
        elif tag == "meta":
            name = attr.get("name") or attr.get("property") or ""
            content = attr.get("content") or ""
            if name and content:
                self.meta[name] = content

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data


def _playwright_available() -> dict[str, Any]:
    try:
        import playwright  # noqa: F401

        return {"installed": True, "version": getattr(playwright, "__version__", "unknown")}
    except ImportError:
        return {
            "installed": False,
            "hint": 'pip install "playwright>=1.49,<2" && playwright install chromium',
        }


def fetch_public_global_pulse(*, timeout: float = 30.0) -> dict[str, Any]:
    """Fetch public Global Pulse page metadata (no login)."""
    req = urllib.request.Request(
        GLOBAL_PULSE_URL,
        headers={"User-Agent": BROWSER_UA, "Accept": "text/html,application/xhtml+xml"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except urllib.error.URLError as exc:
        return {"success": False, "url": GLOBAL_PULSE_URL, "error": str(exc.reason or exc)}

    parser = _MetaParser()
    parser.feed(html)
    snippet = re.sub(r"\s+", " ", parser.meta.get("description", "") or parser.title).strip()
    return {
        "success": True,
        "url": GLOBAL_PULSE_URL,
        "title": parser.title.strip(),
        "description": snippet[:2000],
        "og": {k: v for k, v in parser.meta.items() if k.startswith("og:")},
    }


def _save_last_crawl(payload: dict[str, Any]) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {**payload, "saved_at": time.time()}
    LAST_CRAWL_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_last_crawl() -> dict[str, Any] | None:
    if not LAST_CRAWL_PATH.is_file():
        return None
    try:
        return json.loads(LAST_CRAWL_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _interesting_api_url(url: str) -> bool:
    lower = url.lower()
    return any(hint in lower for hint in _API_HINTS)


def _trim_text(text: str, limit: int = 12000) -> str:
    cleaned = re.sub(r"\n{3,}", "\n\n", text.strip())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 20] + "\n…[truncated]"


def _looks_like_login_page(body_text: str) -> bool:
    lower = (body_text or "").lower()
    markers = ("sign in", "create account", "forgot password")
    hits = sum(1 for m in markers if m in lower)
    return hits >= 2 and "password" in lower


def _is_login_form_visible(page) -> bool:
    """True when the email/password form is on screen."""
    try:
        return (
            page.locator('input[name="password"]').count() > 0
            and page.locator('input[name="email"]').count() > 0
        )
    except Exception:
        return False


def _perform_login(page, creds: dict[str, str], *, timeout_ms: int) -> None:
    """Submit SitDeck login form (tab button is NOT submit — use type=submit)."""
    email_box = page.locator('input[name="email"]').first
    pass_box = page.locator('input[name="password"]').first
    email_box.click(timeout=timeout_ms)
    email_box.fill(creds["email"], timeout=timeout_ms)
    pass_box.click(timeout=timeout_ms)
    pass_box.fill(creds["password"], timeout=timeout_ms)
    submit = page.locator('button[type="submit"]')
    if submit.count() == 0:
        submit = page.get_by_role("button", name=re.compile(r"^sign\s*in$", re.I))
    submit.first.click(timeout=timeout_ms)


def _looks_like_dashboard(body_text: str) -> bool:
    lower = (body_text or "").lower()
    if _looks_like_login_page(body_text):
        return False
    hints = ("widget", "deck", "briefing", "dashboard", "map", "alert", "situation")
    return any(h in lower for h in hints) or len(body_text or "") > 400


def _goto_page(page, url: str, *, timeout_ms: int, wait_until: str = DEFAULT_GOTO_WAIT) -> None:
    """Navigate without networkidle — SitDeck keeps long-polling connections open."""
    page.goto(url, wait_until=wait_until, timeout=timeout_ms)


def _wait_after_login(page, *, timeout_ms: int) -> None:
    """Post-auth settle: DOM ready + optional dashboard chrome (not networkidle)."""
    from playwright.sync_api import TimeoutError as PlaywrightTimeout

    try:
        page.wait_for_load_state(POST_LOGIN_LOAD_STATE, timeout=min(timeout_ms, 30_000))
    except PlaywrightTimeout:
        pass
    for selector in (
        "text=Command Center",
        "text=Ops Center",
        "text=SITUATION",
        "[data-testid='dashboard']",
    ):
        try:
            page.wait_for_selector(selector, timeout=8_000)
            return
        except PlaywrightTimeout:
            continue
    page.wait_for_timeout(2_000)


def crawl_dashboard(
    *,
    headless: bool = True,
    timeout_ms: int = 90_000,
    reuse_session: bool = True,
    include_public_pulse: bool = True,
) -> dict[str, Any]:
    """Log into SitDeck and extract dashboard text + JSON API captures."""
    pw_info = _playwright_available()
    if not pw_info.get("installed"):
        return {"success": False, "error": "playwright_not_installed", **pw_info}

    creds = get_credentials()
    if not creds["email"] or not creds["password"]:
        return {
            "success": False,
            "error": "missing_credentials",
            "hint": "Set SITDECK_EMAIL and SITDECK_PASSWORD in ~/.hermes/.env",
            "credential_status": {
                "email": bool(creds["email"]),
                "password": bool(creds["password"]),
            },
        }

    from playwright.sync_api import TimeoutError as PlaywrightTimeout
    from playwright.sync_api import sync_playwright

    api_captures: list[dict[str, Any]] = []
    result: dict[str, Any] = {
        "success": False,
        "login_url": DEFAULT_LOGIN_URL,
        "app_url": DEFAULT_APP_URL,
        "headless": headless,
    }

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    storage = STORAGE_STATE_PATH if reuse_session and STORAGE_STATE_PATH.is_file() else None

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        context_kwargs: dict[str, Any] = {
            "user_agent": BROWSER_UA,
            "viewport": {"width": 1440, "height": 900},
        }
        if storage:
            context_kwargs["storage_state"] = str(storage)
        context = browser.new_context(**context_kwargs)
        page = context.new_page()

        def on_response(response) -> None:
            try:
                ctype = (response.headers.get("content-type") or "").lower()
                if "json" not in ctype or response.status >= 400:
                    return
                if not _interesting_api_url(response.url):
                    return
                body = response.json()
                api_captures.append(
                    {
                        "url": response.url,
                        "status": response.status,
                        "body_preview": json.dumps(body, ensure_ascii=False)[:4000],
                    }
                )
            except Exception:
                return

        page.on("response", on_response)

        pulse_future = None
        executor = ThreadPoolExecutor(max_workers=1) if include_public_pulse else None
        if executor is not None:
            pulse_future = executor.submit(fetch_public_global_pulse)

        try:
            _goto_page(page, DEFAULT_LOGIN_URL, timeout_ms=timeout_ms)
            page.wait_for_timeout(2000)

            logged_in = not _is_login_form_visible(page)

            if not logged_in:
                _perform_login(page, creds, timeout_ms=timeout_ms)
                try:
                    page.wait_for_function(
                        "() => !document.querySelector('input[name=\"password\"]')",
                        timeout=timeout_ms,
                    )
                except PlaywrightTimeout:
                    pass
                _wait_after_login(page, timeout_ms=timeout_ms)

            if _is_login_form_visible(page):
                _goto_page(page, DEFAULT_APP_URL, timeout_ms=timeout_ms)
                page.wait_for_timeout(2000)

            body_text = redact_secrets(page.inner_text("body"), creds)
            title = page.title()
            url = page.url

            if _is_login_form_visible(page) or _looks_like_login_page(body_text):
                if STORAGE_STATE_PATH.is_file():
                    STORAGE_STATE_PATH.unlink(missing_ok=True)
                result.update(
                    {
                        "success": False,
                        "error": "login_failed",
                        "hint": (
                            "SitDeck login form still visible after submit. "
                            "If credentials are correct, try `hermes sitdeck-osint crawl --no-headless`."
                        ),
                        "page_title": title,
                        "final_url": url,
                        "body_preview": _trim_text(body_text, 500),
                    }
                )
            else:
                context.storage_state(path=str(STORAGE_STATE_PATH))
                result.update(
                    {
                        "success": True,
                        "page_title": title,
                        "final_url": url,
                        "body_text": _trim_text(body_text),
                        "api_captures": api_captures[:40],
                        "api_capture_count": len(api_captures),
                        "session_saved": STORAGE_STATE_PATH.is_file(),
                        "dashboard_detected": _looks_like_dashboard(body_text),
                    }
                )
        except PlaywrightTimeout as exc:
            result.update({"error": "timeout", "detail": str(exc)[:500]})
        except Exception as exc:  # pragma: no cover
            result.update({"error": type(exc).__name__, "detail": str(exc)[:800]})
        finally:
            context.close()
            browser.close()
            if pulse_future is not None:
                try:
                    result["public_global_pulse"] = pulse_future.result(timeout=45.0)
                except Exception as exc:
                    result["public_global_pulse"] = {
                        "success": False,
                        "error": type(exc).__name__,
                        "detail": str(exc)[:200],
                    }
            if executor is not None:
                executor.shutdown(wait=False)

    if include_public_pulse and "public_global_pulse" not in result:
        result["public_global_pulse"] = fetch_public_global_pulse()

    if result.get("success"):
        _save_last_crawl(result)

    return result


def build_digest(crawl: dict[str, Any]) -> str:
    """Format crawl output for agent / cron delivery."""
    if not crawl.get("success"):
        return json.dumps(crawl, ensure_ascii=False, indent=2)

    lines = [
        "# SitDeck OSINT Digest",
        f"_URL: {crawl.get('final_url', DEFAULT_APP_URL)}_",
        "",
    ]
    pulse = crawl.get("public_global_pulse") or {}
    if pulse.get("success"):
        lines.extend(
            [
                "## Global Pulse (public)",
                pulse.get("title") or "",
                pulse.get("description") or "",
                "",
            ]
        )

    lines.extend(["## Dashboard extract", crawl.get("body_text") or "_(empty)_", ""])

    captures = crawl.get("api_captures") or []
    if captures:
        lines.append("## API captures (sample)")
        for item in captures[:8]:
            lines.append(f"- `{item.get('url', '')}`")
            preview = (item.get("body_preview") or "")[:400]
            if preview:
                lines.append(f"  ```\n  {preview}\n  ```")
        lines.append("")

    lines.append("_Source: SitDeck browser crawl (user account). Not World Monitor MCP._")
    return "\n".join(lines)
