# Browser Auth Recovery Packet — 2026-04-21 19:42 CDT

## Goal
Recover or prove a real signed-in X browser session in the same Hermes publish environment before attempting launch publish.

## Current marker state
- File: /Users/hermesmasteragent/.hermes/state/x-access.json
- Status: stale
- Handle: KelEvur
- Notes: Browser-session marker exists, but live Hermes browser verification at 2026-04-21 19:05 CDT still reached logged-out X surfaces (`https://x.com/` landing page and `/compose/post` login redirect). Re-auth or a successful live composer check is required before browser-first publish.

## Required live proof
- Reach https://x.com/home while signed in, or reach https://x.com/compose/post without a login redirect.
- Capture one screenshot proving the signed-in surface.
- Record the screenshot path and exact verified surface.

## Recovery steps
1. Open https://x.com/home in the actual Hermes publish browser session.
2. If X shows a login form, landing page, or "Already have an account?", sign in to the intended account.
3. Re-open https://x.com/compose/post and confirm the composer loads instead of redirecting to /i/flow/login.
4. Save a screenshot of the signed-in surface.
5. Mark the session verified with:
   bash starter-kits/agent-launch-closeout-kit/scripts/browser-auth-recovery.sh --verified --surface-url https://x.com/compose/post --screenshot-path /absolute/path/to/screenshot.png

## Blocking rule
Do not publish from a stale marker alone.
