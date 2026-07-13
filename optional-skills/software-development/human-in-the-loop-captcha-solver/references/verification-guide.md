# End-to-End Verification Guide

Test the full human-in-the-loop captcha relay pipeline without a real captcha-protected site.

## Prerequisites

- `cloudflared` binary installed (download from cloudflare.com)
- Python 3.8+
- A phone or second device on a different network (mobile data)
- This skill's `scripts/captcha_relay.py` available

## Step 1: Start the test server with Google's test keys

```bash
cd ~/.hermes/skills/software-development/human-in-the-loop-captcha-solver

python3 scripts/captcha_relay.py \
  --sitekey 6LeIxAcTAAAAAJcZVRqyHh71UMIEGNQ_MXjiZKhI \
  --port 8443
```

Expected output:
```
reCAPTCHA relay server on http://0.0.0.0:8443
Tunnel: cloudflared tunnel --url http://localhost:8443
Waiting for human to solve...
```

## Step 2: Expose via Cloudflare Tunnel (separate terminal)

```bash
cloudflared tunnel --url http://localhost:8443
```

Expected output (within seconds):
```
Your quick Tunnel has been created! Visit it at:
  https://<random>.trycloudflare.com
```

Copy the `trycloudflare.com` URL.

## Step 3: Solve on phone

1. Open the `trycloudflare.com` URL on your phone (use mobile data, NOT the same WiFi as the server)
2. You should see a dark card with "🔐 Solve Captcha" and the reCAPTCHA widget
3. Tap the "I'm not a robot" checkbox
4. Since this is Google's test key, **no image challenge appears** — it passes immediately
5. The status text changes to "✅ Solved! Token sent!"
6. On the server terminal, you'll see:
   ```
   ✓ Token received (2101 chars)
   {"token": "03AFcWeA..."}
   ```

## Step 4: Verify the token file

```bash
cat /tmp/captcha_token.txt | wc -c
# Should output ~2101

# Verify it's a valid reCAPTCHA token format (starts with 03AFcWeA or similar)
head -c 20 /tmp/captcha_token.txt
```

## Step 5: Verify with Google's verification API

```bash
TOKEN=$(cat /tmp/captcha_token.txt)
curl -s "https://www.google.com/recaptcha/api/siteverify" \
  -d "secret=6LeIxAcTAAAAAGG-vFI1TnRWxMZNFuojJ4WifJWe" \
  -d "response=$TOKEN" | python3 -m json.tool
```

Expected response:
```json
{
  "success": true,
  "challenge_ts": "...",
  "hostname": "localhost",
  "score": 1.0
}
```

## Automated Test Suite

Two automated test suites ship with this skill. Run them any time you modify
the relay scripts, and ideally before submitting changes upstream.

### `scripts/test_relay_pipeline.py` — end-to-end smoke tests

Exercises the public entry-point scripts and asserts the full pipeline is
sound (HTML rendering, /token handler, timeout behavior, Google pre-flight
probe). Runs offline by default; pass `INCLUDE_NETWORK=1` to also exercise
the live `siteverify` round-trip.

```bash
cd ~/.hermes/skills/software-development/human-in-the-loop-captcha-solver
python3 scripts/test_relay_pipeline.py                 # offline: 4 tests, ~1s
INCLUDE_NETWORK=1 python3 scripts/test_relay_pipeline.py # with probe: 4 tests, ~2s
```

What it covers:
- `test_captcha_relay_happy_path` — `--sitekey` flag is honored; served HTML
  contains the sitekey; /token writes to `/tmp/captcha_token.txt` and exits 0.
- `test_captcha_test_happy_path` — Google test-key page renders with the
  TEST MODE badge; /token works.
- `test_timeout_path` — with a 1s timeout and no /token hit, the relay
  exits non-zero with a "Timeout" message within ~10s (not blocking forever).
- `test_preflight_probe_against_google_test_keys` (opt-in) — confirms the
  pre-flight probe works against real Google `siteverify`.

### `scripts/test_no_stale_token.py` — regression tests

Guards against the two bugs found in upstream review of PR #32331:
"a token file left by any earlier run makes the timeout condition false,
so the advertised two-minute timeout never shuts down; the result path
also reads that stale file."

```bash
python3 scripts/test_no_stale_token.py
```

What it covers:
- `test_token_file_cleared_on_startup` — relay removes any pre-existing
  token file on startup.
- `test_stale_file_does_not_leak_into_new_run` — a leftover file from a
  prior run does not appear in a new run's output if no /token was hit.
- `test_timeout_fires_even_when_stale_file_exists` — the timeout shuts
  the server down regardless of whether a stale file is present.
- `test_solve_writes_file_and_returns_token` — happy path still works.

These tests are designed to **fail on the pre-fix code** and pass on the
fix. If you revert `scripts/_relay_common.py` to the old behavior, three
of the four tests will fail with messages pointing at exactly the bug.

## Common Failure Modes

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `connection reset by peer` | SSL wrapping on tunnel | Use `--url http://localhost:8443` (not https) |
| Tunnel URL shows 502 Bad Gateway | Server not running or wrong port | Verify cloudflared arg matches server port |
| Captcha doesn't appear on phone | reCAPTCHA JS blocked by adblocker/privacy extension | Try a different browser (Safari/Chrome incognito) |
| "Captcha did not load" on page | reCAPTCHA not rendering due to CSP or network issue | Check phone has internet; retry tunnel URL |
| No token after solving | XHR from page can't reach tunnel | Verify cloudflared tunnel is active and port matches |
| Token expires before injection | Too long between solve and form submit | Prepare submit script before asking human to solve |

## Infrastructure Tests (Pre-Flight Checks)

Before attempting a real captcha, verify components independently. The
automated `scripts/test_relay_pipeline.py` (see above) replaces the
hand-rolled checks below — it exercises the full pipeline including a
live `siteverify` round-trip in ~2 seconds (pass `INCLUDE_NETWORK=1`).

If you want the manual pre-flight for diagnostics:

```bash
# 1. Can Python serve HTTP?
python3 -m http.server 8443 --bind 0.0.0.0 &>/dev/null &
curl -s -o /dev/null -w "%{http_code}" http://localhost:8443
# Expected: 200 (directory listing)
kill %1 2>/dev/null

# 2. Can cloudflared tunnel HTTP?
python3 -m http.server 8443 --bind 0.0.0.0 &>/dev/null &
tunnel_url=$(cloudflared tunnel --url http://localhost:8443 2>&1 | grep -oP 'https://\S+\.trycloudflare\.com' | head -1)
curl -s -o /dev/null -w "%{http_code}" "$tunnel_url"
# Expected: 200 (tunnel works!)
# Then kill both

# 3. Can reCAPTCHA JS load behind tunnel?
# Do the full test above with Google test keys (Steps 1-4)
```
