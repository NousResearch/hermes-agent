#!/usr/bin/env python3
"""
Xiaomi Account Registration — Single-Process Automation
Register → Solve reCAPTCHA → Verify Email → Submit Code

Usage:
  python3 xiaomi_register.py --email cf2@nounrich.works --password "Kontol22@" --ref QFQAKP

Environment:
  CAPSOLVER_API_KEY  — Capsolver API key (required)
  IMAP_HOST          — default: imap.gmail.com
  IMAP_PORT          — default: 993
  IMAP_USER          — Gmail address (required)
  IMAP_PASS          — App password (required)
"""

import argparse, os, sys, time, re, imaplib, email as email_mod
from email.header import decode_header
from datetime import datetime, timezone

# ── Capsolver ──────────────────────────────────────────────────────────────
CAPSOLVER_CREATE_URL = "https://api.capsolver.com/createTask"
CAPSOLVER_RESULT_URL = "https://api.capsolver.com/getTaskResult"

import requests

def solve_recaptcha(site_key: str, page_url: str, api_key: str, timeout: int = 120) -> str | None:
    """Solve reCAPTCHA Enterprise v2 via Capsolver. Returns token or None."""
    log("CAPTCHA", f"Creating Capsolver task  site_key={site_key}")
    payload = {
        "clientKey": api_key,
        "task": {
            "type": "ReCaptchaV2EnterpriseTaskProxyLess",
            "websiteURL": page_url,
            "websiteKey": site_key,
        },
    }
    r = requests.post(CAPSOLVER_CREATE_URL, json=payload, timeout=30)
    r.raise_for_status()
    data = r.json()
    if data.get("errorId", 1) != 0:
        log("CAPTCHA", f"Create error: {data.get('errorDescription')}")
        return None
    task_id = data["taskId"]
    log("CAPTCHA", f"Task created  id={task_id}")

    start = time.time()
    while time.time() - start < timeout:
        time.sleep(3)
        r = requests.post(CAPSOLVER_RESULT_URL, json={"clientKey": api_key, "taskId": task_id}, timeout=30)
        r.raise_for_status()
        data = r.json()
        status = data.get("status")
        if status == "ready":
            token = data["solution"]["gRecaptchaResponse"]
            log("CAPTCHA", f"Solved  token_len={len(token)}")
            return token
        if status == "failed":
            log("CAPTCHA", f"Failed: {data.get('errorDescription')}")
            return None
        elapsed = int(time.time() - start)
        log("CAPTCHA", f"Processing... ({elapsed}s)")
    log("CAPTCHA", "Timeout")
    return None


# ── IMAP ───────────────────────────────────────────────────────────────────
def poll_imap_for_code(imap_host: str, imap_port: int, imap_user: str, imap_pass: str,
                       after_time: datetime, max_wait: int = 120) -> str | None:
    """Poll IMAP for the latest Xiaomi verification code received after after_time (UTC)."""
    log("IMAP", f"Polling for code after {after_time.isoformat()}  max_wait={max_wait}s")
    start = time.time()
    while time.time() - start < max_wait:
        try:
            imap = imaplib.IMAP4_SSL(imap_host, imap_port)
            imap.login(imap_user, imap_pass)
            imap.select("INBOX")
            status, messages = imap.search(None, '(FROM "xiaomi")')
            if status != "OK":
                raise Exception("IMAP search failed")
            msg_ids = messages[0].split()
            # Check from newest to oldest
            for mid in reversed(msg_ids):
                status, data = imap.fetch(mid, "(RFC822)")
                if status != "OK":
                    continue
                msg = email_mod.message_from_bytes(data[0][1])
                # Parse date
                date_str = msg.get("Date", "")
                try:
                    from email.utils import parsedate_to_datetime
                    msg_time = parsedate_to_datetime(date_str)
                    if msg_time.tzinfo is None:
                        msg_time = msg_time.replace(tzinfo=timezone.utc)
                except Exception:
                    continue
                # Only consider emails after registration started
                if msg_time < after_time:
                    break  # older emails won't match, stop
                # Extract body
                body = _get_email_body(msg)
                codes = re.findall(r"\b\d{6}\b", body)
                if codes:
                    code = codes[0]
                    log("IMAP", f"Found code: {code}  (email time: {date_str})")
                    imap.logout()
                    return code
            imap.logout()
        except Exception as e:
            log("IMAP", f"Error: {e}")
        elapsed = int(time.time() - start)
        log("IMAP", f"No code yet  ({elapsed}s/{max_wait}s)")
        time.sleep(5)
    log("IMAP", "Timeout — no code received")
    return None


def _get_email_body(msg) -> str:
    """Extract text body from email message."""
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                p = part.get_payload(decode=True)
                if p:
                    return p.decode("utf-8", errors="replace")
        for part in msg.walk():
            if part.get_content_type() == "text/html":
                p = part.get_payload(decode=True)
                if p:
                    return p.decode("utf-8", errors="replace")
    else:
        p = msg.get_payload(decode=True)
        if p:
            return p.decode("utf-8", errors="replace")
    return ""


# ── Logging ────────────────────────────────────────────────────────────────
def log(tag: str, msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [{tag}] {msg}", flush=True)


# ── Main Flow ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Xiaomi Account Registration")
    parser.add_argument("--email", required=True, help="Email address")
    parser.add_argument("--password", required=True, help="Account password")
    parser.add_argument("--ref", default="QFQAKP", help="Referral code")
    parser.add_argument("--captcha-site-key", default="6LeBM0ocAAAAAEwYcFUjtxpVbs-0rnbSVXBBXmh4")
    parser.add_argument("--max-imap-wait", type=int, default=120, help="Max seconds to wait for email code")
    args = parser.parse_args()

    capsolver_key = os.environ.get("CAPSOLVER_API_KEY", "")
    if not capsolver_key:
        print("ERROR: CAPSOLVER_API_KEY env var required", file=sys.stderr)
        sys.exit(1)
    imap_host = os.environ.get("IMAP_HOST", "imap.gmail.com")
    imap_port = int(os.environ.get("IMAP_PORT", "993"))
    imap_user = os.environ.get("IMAP_USER", "")
    imap_pass = os.environ.get("IMAP_PASS", "")
    if not imap_user or not imap_pass:
        print("ERROR: IMAP_USER and IMAP_PASS env vars required", file=sys.stderr)
        sys.exit(1)

    url = f"https://platform.xiaomimimo.com?ref={args.ref}"
    captcha_page_url = "https://global.account.xiaomi.com/fe/service/register"

    # ── Step 1: Solve reCAPTCHA ────────────────────────────────────────
    log("MAIN", "Step 1: Solving reCAPTCHA via Capsolver")
    token = solve_recaptcha(args.captcha_site_key, captcha_page_url, capsolver_key)
    if not token:
        log("MAIN", "FATAL: CAPTCHA solve failed")
        sys.exit(1)

    # ── Step 2: Browser automation ─────────────────────────────────────
    log("MAIN", "Step 2: Launching browser")
    from playwright.sync_api import sync_playwright

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        ctx = browser.new_context(viewport={"width": 1280, "height": 960})
        page = ctx.new_page()

        # Track API calls
        api_results = []
        def on_response(resp):
            u = resp.url
            if "sendEmailRegTicket" in u:
                try:
                    body = resp.text()[:200]
                except Exception:
                    body = "<error>"
                api_results.append(("sendEmailRegTicket", resp.status, body))
                log("API", f"sendEmailRegTicket  status={resp.status}  body={body[:120]}")
            if "registerEmail" in u or "verifyEmail" in u or "register/verify" in u:
                try:
                    body = resp.text()[:200]
                except Exception:
                    body = "<error>"
                api_results.append(("verify", resp.status, body))
                log("API", f"verify  status={resp.status}  body={body[:120]}")

        page.on("response", on_response)

        # Navigate
        log("MAIN", f"Navigating to {url}")
        page.goto(url, wait_until="networkidle", timeout=30000)
        log("MAIN", f"Page loaded: {page.title()}")

        # Accept cookies
        try:
            page.click("text=Accept cookies", timeout=3000)
            log("MAIN", "Cookies accepted")
        except Exception:
            pass

        # Click Sign up
        page.click("text=Sign up")
        page.wait_for_timeout(2000)
        log("MAIN", "On registration page")

        # Fill form
        page.fill('input[name="email"]', args.email)
        page.fill('input[name="password"]', args.password)
        page.fill('input[name="repassword"]', args.password)
        cb = page.query_selector('input[type="checkbox"]')
        if cb and not cb.is_checked():
            cb.click(force=True)
        page.wait_for_timeout(1000)
        log("MAIN", "Form filled")

        # Click Next — triggers CAPTCHA
        reg_start_time = datetime.now(timezone.utc)
        page.click('button:has-text("Next")')
        page.wait_for_timeout(4000)
        log("MAIN", "Next clicked — CAPTCHA panel should appear")

        # ── Step 2.5: Click reCAPTCHA checkbox in anchor iframe ────────
        # MiVerify only re-submits after it detects the checkbox was "checked".
        # We click the anchor-frame checkbox first, then inject our token.
        log("MAIN", "Step 2.5: Waiting for reCAPTCHA anchor iframe")
        anchor = None
        for attempt in range(10):
            frames = page.frames
            anchor = next((f for f in frames if "recaptcha/anchor" in f.url), None)
            if anchor:
                break
            page.wait_for_timeout(1000)
            log("MAIN", f"  Anchor frame not found yet, retry {attempt+1}/10")

        if anchor:
            try:
                ck = anchor.query_selector(".recaptcha-checkbox-border")
                if ck:
                    ck.click()
                    log("MAIN", "Clicked reCAPTCHA checkbox")
                    page.wait_for_timeout(3000)
                else:
                    log("MAIN", "Checkbox element not found in anchor frame")
            except Exception as e:
                log("MAIN", f"Checkbox click error (non-fatal): {e}")
        else:
            log("MAIN", "Anchor frame not found after 10 retries — proceeding with token inject only")

        # ── Step 3: Inject reCAPTCHA token + trigger callback ──────────
        log("MAIN", "Step 3: Injecting CAPTCHA token")
        inject_js = """
        (token) => {
            // 1. Set textarea value
            const ta = document.getElementById('g-recaptcha-response');
            if (ta) ta.value = token;

            // 2. Walk ___grecaptcha_cfg clients to find callback
            try {
                const cfg = window.___grecaptcha_cfg;
                if (cfg && cfg.clients) {
                    const findCallback = (o) => {
                        if (!o || typeof o !== 'object') return false;
                        for (const [k, v] of Object.entries(o)) {
                            if (k === 'callback' && typeof v === 'function') {
                                v(token);
                                return true;
                            }
                            if (typeof v === 'object' && v !== null) {
                                if (findCallback(v)) return true;
                            }
                        }
                        return false;
                    };
                    for (const [, cd] of Object.entries(cfg.clients)) {
                        if (findCallback(cd)) break;
                    }
                }
            } catch(e) { console.error('callback error', e); }

            // 3. Also try calling miVerify callback directly
            try {
                if (window.miVerify && typeof window.miVerify.onCaptchaVerify === 'function') {
                    window.miVerify.onCaptchaVerify(token);
                }
            } catch(e) {}

            // 4. Dispatch custom event for MiVerify
            try {
                document.dispatchEvent(new CustomEvent('recaptcha-success', { detail: { token } }));
            } catch(e) {}

            return true;
        }
        """
        result = page.evaluate(inject_js, token)
        log("MAIN", f"Token injected  evaluate_result={result}")

        # ── Step 4: Wait for navigation to verify page ─────────────────
        log("MAIN", "Step 4: Waiting for navigation to verify page (up to 30s)")
        try:
            page.wait_for_url("**/verify**", timeout=30000)
            log("MAIN", f"Navigated to verify page: {page.url}")
        except Exception:
            # Check if API call succeeded
            ok_calls = [r for r in api_results if r[0] == "sendEmailRegTicket" and '"result":"ok"' in r[2]]
            rate_limited = [r for r in api_results if r[0] == "sendEmailRegTicket" and '20332' in r[2]]
            if rate_limited:
                log("MAIN", "FATAL: Email send rate limited (code 20332) — too many registration attempts for this email")
                log("MAIN", "Wait or use a different email address")
                browser.close()
                sys.exit(1)
            elif ok_calls:
                log("MAIN", "API call succeeded but URL didn't change — trying manual wait")
                page.wait_for_timeout(5000)
                if "/verify" in page.url:
                    log("MAIN", f"Verify page reached after extra wait: {page.url}")
                else:
                    log("MAIN", f"Still on: {page.url}")
                    log("MAIN", "FATAL: Verify page not reached despite API success")
                    browser.close()
                    sys.exit(1)
            else:
                log("MAIN", "FATAL: No successful sendEmailRegTicket response")
                log("MAIN", f"Current URL: {page.url}")
                browser.close()
                sys.exit(1)

        # ── Step 5: Poll IMAP for verification code ────────────────────
        log("MAIN", "Step 5: Polling IMAP for verification code")
        code = poll_imap_for_code(
            imap_host, imap_port, imap_user, imap_pass,
            after_time=reg_start_time,
            max_wait=args.max_imap_wait,
        )
        if not code:
            log("MAIN", "FATAL: No verification code received")
            browser.close()
            sys.exit(1)

        # ── Step 6: Enter code and submit ──────────────────────────────
        log("MAIN", f"Step 6: Entering code {code}")
        icode = page.query_selector('input[name="icode"]')
        if icode:
            icode.fill(code)
            log("MAIN", "Filled icode input")
        else:
            # Try OTP-style individual inputs
            otps = page.query_selector_all('input[maxlength="1"]')
            if len(otps) >= 6:
                for i in range(6):
                    otps[i].fill(code[i])
                log("MAIN", "Filled OTP inputs")
            else:
                # Fallback: try any visible input
                inputs = page.query_selector_all('input[type="text"], input:not([type])')
                for inp in inputs:
                    if inp.is_visible():
                        inp.fill(code)
                        log("MAIN", f"Filled visible input")
                        break

        page.wait_for_timeout(1000)
        log("MAIN", "Clicking Submit")
        page.click('button:has-text("Submit")')

        # ── Step 7: Wait for result ────────────────────────────────────
        log("MAIN", "Step 7: Waiting for registration result")
        page.wait_for_timeout(10000)

        final_url = page.url
        final_body = page.evaluate("() => document.body?.innerText?.substring(0,1500) || ''")
        log("MAIN", f"Final URL: {final_url}")
        log("MAIN", f"Final body: {final_body[:500]}")

        # Check success
        if "console" in final_url or "balance" in final_url or "sts" in final_url:
            log("MAIN", "✅ REGISTRATION SUCCESSFUL — redirected to platform")
        elif "register" not in final_url:
            log("MAIN", "✅ REGISTRATION LIKELY SUCCESSFUL — left verify page")
        else:
            log("MAIN", "⚠️  Still on registration/verify page — may need retry")

        browser.close()
        log("MAIN", "Browser closed. Done.")


if __name__ == "__main__":
    main()
