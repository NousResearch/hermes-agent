#!/usr/bin/env python3
"""
Test reCAPTCHA relay using Google's official test sitekey.
Always passes No CAPTCHA. Use this to verify infrastructure before a real solve.

Usage:
    python3 captcha_test.py [--port PORT]

Then tunnel: cloudflared tunnel --url http://localhost:PORT

Open the tunnel URL on your phone → tap "I'm not a robot" → token auto-returns.

Google test keys: https://developers.google.com/recaptcha/docs/faq

The HTTP server, /token capture, timeout, and stale-token handling live in
`_relay_common.py`. This script only owns the HTML template + CLI parsing.
"""
import argparse

from _relay_common import run_relay

# Google's official test reCAPTCHA v2 sitekey — always returns No CAPTCHA
TEST_SITEKEY = "6LeIxAcTAAAAAJcZVRqyHh71UMIEGNQ_MXjiZKhI"


_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
<title>Test Captcha Solver</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,sans-serif;background:#1a1a2e;color:#eee;display:flex;align-items:center;justify-content:center;min-height:100vh}
.card{background:#16213e;padding:40px;border-radius:16px;max-width:400px;text-align:center}
.test-badge{display:inline-block;background:#ff9800;color:#fff;padding:3px 10px;border-radius:20px;font-size:0.75rem;font-weight:600;margin-bottom:12px}
.rc{display:flex;justify-content:center;margin:16px 0;min-height:78px}
.status{padding:10px 16px;border-radius:12px;font-size:0.85rem;margin-top:12px}
.status.ready{background:#1b4332;color:#4ade80}
.status.solving{background:#fff3e0;color:#e65100}
.status.done{background:#e3f2fd;color:#1565c0}
textarea{width:100%;margin-top:12px;padding:8px;border-radius:8px;border:1px solid #444;font-family:monospace;font-size:0.75rem;background:#0f172a;color:#4ade80;display:none}
.fallback{background:#fef3cd;color:#856404;padding:12px;border-radius:10px;font-size:0.85rem;margin-top:16px;display:none}
</style>
</head>
<body>
<div class="card">
  <div class="test-badge">TEST MODE</div>
  <h1>Test Captcha</h1>
  <p>Using Google's test keys — always passes</p>
  <div class="rc" id="rc"></div>
  <div class="status ready" id="status">Waiting for you...</div>
  <textarea id="token-box" readonly></textarea>
  <div class="fallback" id="fallback">
    If captcha doesn't load, open in Safari/Chrome:<br>
    <b>https://recaptcha-demo.appspot.com/recaptcha-v2-checkbox.php?sitekey=__SITEKEY__</b>
  </div>
</div>
<script>
var container=document.getElementById('rc'),status=document.getElementById('status'),box=document.getElementById('token-box'),fb=document.getElementById('fallback');
function render(){try{if(typeof grecaptcha!=='undefined'&&grecaptcha.render){grecaptcha.render('rc',{'sitekey':'__SITEKEY__','callback':onSolved,'expired-callback':onExpired});status.className='status solving';status.textContent='Tap the checkbox above';return}}catch(e){}fb.style.display='block';status.className='status error';status.textContent='Failed to load'}
function onSolved(t){box.style.display='block';box.value=t;status.className='status done';status.textContent='Solved!';fetch('/token?t='+encodeURIComponent(t)).then(r=>r.ok?status.textContent='Token sent! Close this page.':status.textContent='Server error.')}
function onExpired(){status.className='status error';status.textContent='Expired. Refresh.';box.value=''}
function onError(){fb.style.display='block';status.className='status error';status.textContent='Error. Use fallback link.'}
var s=document.createElement('script');s.src='https://www.google.com/recaptcha/api.js?onload=onLoad&render=explicit';s.async=s.defer=true;window.onLoad=render;document.head.appendChild(s);
setTimeout(function(){if(!box.value&&container.childElementCount===0){fb.style.display='block';status.className='status error';status.textContent='Did not load. Use fallback.'}},8000);
</script>
</body>
</html>"""


def make_html() -> str:
    # Substitute the sitekey into the pre-built template. Single-pass replace
    # so we don't have to deal with Python f-string brace-escaping inside the
    # giant HTML literal above.
    return _HTML.replace("__SITEKEY__", TEST_SITEKEY)


def main() -> int:
    ap = argparse.ArgumentParser(description="Test captcha relay with Google test keys")
    ap.add_argument(
        "--port", type=int, default=8443, help="Local port (default: 8443)"
    )
    args = ap.parse_args()

    def html_factory() -> str:
        return make_html()

    token = run_relay(
        html_factory=html_factory,
        banner=f"Test captcha server on http://0.0.0.0:{args.port}",
        port=args.port,
    )
    return 0 if token else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
