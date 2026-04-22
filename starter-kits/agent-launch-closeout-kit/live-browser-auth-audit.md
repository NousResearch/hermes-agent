# Live Browser Auth Audit — 2026-04-21

## Purpose
Prove the real publish blocker from the live Hermes publish surface instead of trusting a stale readiness marker.

## Evidence checked
1. `~/.hermes/state/x-access.json`
   - reports `mode: browser-session`
   - reports `status: ready`
   - handle: `KelEvur`
2. `bash starter-kits/agent-launch-closeout-kit/scripts/publish-preflight.sh`
   - returns `Publish preflight OK`
   - returns `Publish path: browser-session ready (KelEvur)`
   - X API env vars still missing: `5/5`
3. Live browser inspection in the Hermes browser session
   - `https://x.com/` rendered the logged-out landing page
   - visible evidence included `Already have an account?` and `Sign in`
   - `https://x.com/compose/post` redirected into the login flow instead of reaching the composer

## Finding
`x-access.json` is only a local browser-session marker. It is not sufficient proof that the actual Hermes publish session is authenticated right now.

## Decision
Treat publish as blocked until the real publish session reaches a logged-in X surface (ideally the composer) in the same environment that will do the post.

## Required next proof
Before claiming publish is unblocked, capture all of the following in the same block:
- run `bash starter-kits/agent-launch-closeout-kit/scripts/browser-auth-recovery.sh --prepare`
- live browser session reaches `https://x.com/home` or the composer while signed in
- a screenshot-backed recovery artifact is created and then recorded with `--verified`
- composer is available for `starter-kits/agentic-cron-orchestration-kit/launch/launch-thread.md`
- post URL and timestamp are recorded in `starter-kits/agent-launch-closeout-kit/launch-execution-log.md`

## Consequence for this MVP
The closeout kit should keep the browser-first publish path, but it must require a live browser auth check in addition to the `x-access.json` marker.

## Marker invalidation rerun — 2026-04-21 19:05 CDT
- Live browser rerun still reached logged-out X surfaces:
  - `https://x.com/` rendered the public landing page with `Already have an account?` / `Sign in`
  - `https://x.com/compose/post` redirected to `/i/flow/login?redirect_after_login=%2Fcompose%2Fpost`
- Direct correction shipped in the same block:
  - `~/.hermes/state/x-access.json` was downgraded from `status: ready` to `status: stale`
  - `publish-preflight.sh` now prints the browser marker status when the browser-first path is not actually ready
- New rule: a stale marker must be treated as blocked until a new live browser check reaches signed-in home/composer in the same publish environment.

## Recovery packet scaffold — 2026-04-21 19:42 CDT
- Added `starter-kits/agent-launch-closeout-kit/scripts/browser-auth-recovery.sh` as the canonical stale-marker recovery path.
- Verified it can generate a timestamped recovery packet without claiming publish is fixed yet:
  - `bash starter-kits/agent-launch-closeout-kit/scripts/browser-auth-recovery.sh --prepare`
  - output: `starter-kits/agent-launch-closeout-kit/auth-artifacts/browser-auth-recovery-2026-04-21T19-42-32-0500.md`
- Purpose of the packet: freeze the exact re-auth + screenshot evidence steps so the next publish block attacks the real blocker instead of restating it.


## Browser login proof refresh — 2026-04-21 20:18 CDT
- Fresh browser verification hit the same logged-out surface again:
  - attempted `https://x.com/compose/post`
  - final URL stayed on the login flow: `https://x.com/i/flow/login?redirect_after_login=%2Fcompose%2Fpost`
  - visible body text included `Happening now`, `Join today`, `Already have an account?`, and `Sign in`
- Durable proof artifact created: `starter-kits/agent-launch-closeout-kit/auth-artifacts/browser-auth-live-check-2026-04-21T20-18-51-0500.md`
- Screenshot captured for the recovery packet: `/Users/hermesmasteragent/.hermes/cache/screenshots/browser_screenshot_50b1574757d6428eb925d97058c41ae5.png`
- Decision: keep publish blocked and keep `~/.hermes/state/x-access.json` at `status: stale` until a signed-in home/composer screenshot is captured in the real publish session.
