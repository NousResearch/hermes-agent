"""One-off SitDeck login debug (no secrets printed)."""
from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from playwright.sync_api import sync_playwright

load_dotenv(Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes")) / ".env")
email = os.getenv("SITDECK_EMAIL", "")
password = os.getenv("SITDECK_PASSWORD", "")
responses: list[dict] = []


def main() -> int:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 900})

        def on_resp(response) -> None:
            url = response.url
            if not any(x in url.lower() for x in ("auth", "login", "session", "supabase", "token", "/api/")):
                return
            item: dict = {"url": url, "status": response.status}
            try:
                if "json" in (response.headers.get("content-type") or ""):
                    item["body"] = response.text()[:800]
            except Exception as exc:
                item["read_error"] = str(exc)[:120]
            responses.append(item)

        page.on("response", on_resp)
        page.goto("https://app.sitdeck.com/#login", wait_until="networkidle", timeout=90_000)
        page.wait_for_timeout(2000)
        page.locator('input[name="email"]').fill(email)
        page.locator('input[name="password"]').fill(password)
        page.locator('button[type="submit"]').click()
        page.wait_for_timeout(6000)

        body = page.inner_text("body")
        errs = page.evaluate(
            """() => [...document.querySelectorAll(
                '[role=alert], .text-destructive, p.text-destructive'
            )].map(e => e.innerText.trim()).filter(Boolean)"""
        )
        snip = body[:500].replace("\n", " | ").encode("utf-8", errors="replace").decode("utf-8")
        print("still_login", "Forgot password?" in body)
        print("ui_errors", errs)
        print("body_snip", snip)
        print("responses", json.dumps(responses[-20:], ensure_ascii=True, indent=2))
        browser.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
