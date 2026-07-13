#!/usr/bin/env python3
"""
reCAPTCHA relay server. Serves a page with the target's captcha widget,
waits for a human to solve it, returns the token.

Usage:
    python3 captcha_relay.py --sitekey SITEKEY [--port PORT]

Then tunnel: cloudflared tunnel --url http://localhost:PORT

Dependencies: Python stdlib only. No pip packages needed.

The HTTP server, /token capture, timeout, and stale-token handling live in
`_relay_common.py`. This script only owns the HTML template + CLI parsing.
"""
import argparse

from _relay_common import run_relay


def make_html(sitekey: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
<title>Solve Captcha</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,sans-serif;background:#1a1a2e;color:#eee;display:flex;align-items:center;justify-content:center;min-height:100vh;margin:0;padding:16px}}
.card{{background:#16213e;padding:32px 24px;border-radius:16px;max-width:400px;width:100%;text-align:center}}
h1{{font-size:1.3rem;margin-bottom:4px}}
p{{color:#aaa;font-size:0.9rem;margin-bottom:20px}}
.rc{{display:flex;justify-content:center;margin:16px 0;min-height:78px}}
.status{{padding:10px 16px;border-radius:12px;font-size:0.85rem;margin-top:12px}}
.status.ready{{background:#1b4332;color:#4ade80}}
.status.solving{{background:#fff3e0;color:#e65100}}
.status.done{{background:#e3f2fd;color:#1565c0}}
textarea{{width:100%;margin-top:12px;padding:8px;border-radius:8px;border:1px solid #444;font-family:monospace;font-size:0.75rem;background:#0f172a;color:#4ade80;display:none}}
</style>
</head>
<body>
<div class="card">
  <h1>🔐 Solve Captcha</h1>
  <p>Complete this to authorise the request</p>
  <div class="rc" id="rc"></div>
  <div class="status ready" id="status">⏳ Waiting for you...</div>
  <textarea id="token-box" readonly></textarea>
</div>
<script>
var container=document.getElementById('rc'),status=document.getElementById('status'),box=document.getElementById('token-box');
function render(){{try{{if(typeof grecaptcha!=='undefined'&&grecaptcha.render){{grecaptcha.render('rc',{{'sitekey':'{sitekey}','callback':onSolved,'expired-callback':onExpired}});status.className='status solving';status.textContent='🔄 Tap the checkbox above';return}}}}catch(e){{}}status.className='status error';status.textContent='⚠ Captcha failed to load'}}
function onSolved(t){{box.style.display='block';box.value=t;status.className='status done';status.textContent='✅ Solved!';fetch('/token?t='+encodeURIComponent(t)).then(r=>r.ok?status.textContent='✅ Token sent! Close this page.':status.textContent='⚠ Server error.')}}
function onExpired(){{status.className='status error';status.textContent='⚠ Captcha expired. Refresh.';box.value=''}}
var s=document.createElement('script');s.src='https://www.google.com/recaptcha/api.js?onload=onLoad&render=explicit';s.async=s.defer=true;window.onLoad=render;document.head.appendChild(s);
setTimeout(function(){{if(!box.value&&container.childElementCount===0){{status.className='status error';status.textContent='⚠ Captcha did not load. Try again.'}}}},8000);
</script>
</body>
</html>"""


def main() -> int:
    ap = argparse.ArgumentParser(description="reCAPTCHA relay server")
    ap.add_argument(
        "--sitekey", required=True, help="The reCAPTCHA sitekey from the target page"
    )
    ap.add_argument(
        "--port", type=int, default=8443, help="Local port to listen on (default: 8443)"
    )
    args = ap.parse_args()

    sitekey = args.sitekey

    def html_factory() -> str:
        return make_html(sitekey)

    token = run_relay(
        html_factory=html_factory,
        banner=f"reCAPTCHA relay server on http://0.0.0.0:{args.port}",
        port=args.port,
    )
    return 0 if token else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())