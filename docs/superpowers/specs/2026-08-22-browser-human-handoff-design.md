# Browser Human Handoff Design

## Outcome

When Hermes reaches a human-only step while driving a Browser Use cloud browser, it can pause the task and send Alex a Discord DM containing a no-login handoff link. The link opens a mobile-friendly page containing the exact instruction, the live Browser Use controller for the existing browser, and a **Done — resume Hermes** button.

The handoff link is a bearer credential. It expires after 30 minutes, is bound to one Hermes task and one Browser Use session, and is revoked immediately when the user completes the handoff or the browser session ends. Completing the handoff wakes the exact paused Hermes session, which verifies browser state before continuing.

This design does not attempt to bypass CAPTCHA, MFA, consent, or account-selection controls. It hands those steps to the owner.

## User Experience

Hermes recognizes that progress requires human interaction, such as a CAPTCHA, password entry, MFA, consent screen, or ambiguous account selection. Hermes invokes the handoff action with a short imperative instruction, for example:

> Sign in to Shopify and complete the verification prompt. Do not navigate away after the dashboard loads.

Alex receives this Discord DM:

> yo, I need you to sign in to Shopify and complete the verification prompt. This link controls the browser I am using and expires in 30 minutes: <handoff URL>

The handoff page shows:

1. The instruction and remaining validity time.
2. The live Browser Use controller embedded in the page.
3. A fallback **Open live browser** link if embedding is blocked by the client.
4. A fixed **Done — resume Hermes** button.

Pressing Done changes the page to a completed state, disables further browser access through the handoff URL, and resumes Hermes. Reopening a completed or expired link shows only a terminal status and never reveals the Browser Use live URL.

## Architecture

### Browser session metadata

The Browser Use provider will preserve `liveUrl` as `live_url` in the existing session metadata alongside `bb_session_id`, `cdp_url`, and `expires_at`. The provider contract will define `live_url` as optional so Browserbase, local Chrome, CDP overrides, and providers without human-control links remain compatible.

The Browser Use CLI/backend path will reuse Hermes's existing Browser Use provider session instead of asking the harness to auto-provision a second, opaque browser. The harness attaches to that session's CDP endpoint, while Hermes retains its ID, expiry, and live URL. Handoff is available only when the active session has both a live Browser Use session and a non-empty `live_url`; local-browser handoff is outside this change.

Every Hermes bot/task receives a deterministic private harness-daemon name and a distinct Browser Use cloud session key, even when the model omits the optional session name. Two bots that choose the same friendly session name still receive separate managed Chrome instances, so they can work concurrently in the background without sharing tabs, focus, cookies, or browser process state. Repeated calls from the same bot and friendly session reuse its instance. Local Chrome remains one shared browser and can isolate only tabs; the multi-bot guarantee therefore uses Browser Use cloud.

### Handoff manager

A focused handoff manager owns lifecycle and persistence. It creates a cryptographically random token and persists only its SHA-256 digest with:

- Hermes profile/home scope
- raw session ID and task ID
- browser session ID and live URL
- human instruction
- Discord recipient user ID
- creation and expiry timestamps
- state: `pending`, `completed`, `expired`, or `cancelled`
- completion timestamp and wake-delivery status

State lives in a small profile-scoped SQLite database under the active Hermes home so a gateway restart does not lose pending handoffs. Updates use transactions so two Done requests cannot resume the agent twice.

The public token is returned once for URL construction and is never logged, persisted in plaintext, or placed in agent history. Logs use a short handoff ID that is not sufficient to authenticate.

### Public handoff routes

Hermes's always-running gateway API server will expose two deliberately unauthenticated, token-authenticated routes:

- `GET /browser-handoff/{token}` renders the handoff page.
- `POST /browser-handoff/{token}/complete` atomically marks the handoff complete and schedules the wake.

These two handlers intentionally do not use `API_SERVER_KEY`; possession of a valid handoff token is their authorization mechanism. Every other API-server route retains its existing authentication. The page response sets `Cache-Control: no-store`, `Referrer-Policy: no-referrer`, a restrictive Content Security Policy, and no session cookie. The live URL is never returned for an invalid, expired, completed, cancelled, or browser-mismatched token.

The operator configures `browser.handoff.public_base_url` in `config.yaml`. Hermes refuses to create a handoff unless this is an absolute HTTPS origin. On Alex's deployment, the existing Cloudflare tunnel/reverse-proxy process will route this prefix to the Hermes HTTP service; tunnel URL rotation remains an operational concern of the existing tunnel supervisor.

### Agent-facing action

The capability extends the existing Browser Use tool surface instead of adding a new permanent core tool. The Browser Use schema gains an optional handoff action with one required field: a concise instruction describing exactly what Alex must do.

The action:

1. Resolves the active Browser Use session for the calling task/session name.
2. Validates that a live URL exists and that the browser has not expired.
3. Creates the 30-minute handoff record.
4. Opens or resolves a Discord DM for configured owner user `1063878950851448853` and sends the message.
5. Returns a structured `waiting_for_human` result to the agent loop without exposing the bearer URL to model context.
6. Suspends further work for that Hermes session until completion, expiry, cancellation, or interruption.

The Discord delivery path uses the connected Discord adapter when available and a bot-token REST fallback otherwise. It creates a DM channel from the recipient user ID rather than treating the user ID as a channel ID. A failed DM cancels the handoff immediately and returns an actionable error.

### Resume path

The completion endpoint claims the pending handoff in one transaction, revokes its public capability, and uses Hermes's existing session wake mechanism to enqueue an internal continuation for the raw originating session ID. The continuation says that the owner completed the browser handoff and instructs Hermes to inspect the current page before taking the next action.

The browser remains alive during the wait. Handoff completion does not automatically submit forms, close the browser, or assume success. Hermes must verify page state after waking. Normal task cleanup closes every cloud browser owned by that bot after the resumed task finishes; the inactivity reaper also closes abandoned instances after the handoff is completed, cancelled, or expired.

## State and Concurrency Rules

- At most one pending handoff may exist for a given Hermes task and browser session. A retry returns the existing pending handoff and may re-send the DM without minting another live capability.
- Done is idempotent. Only the transaction that changes `pending` to `completed` schedules a wake.
- Expiry is the earlier of 30 minutes and the provider-authoritative browser expiry.
- Browser cleanup cancels all pending handoffs bound to that browser before stopping it.
- A gateway restart reloads pending rows, expires stale rows, and can still accept Done for valid rows.
- A session reset or explicit interruption cancels its pending handoffs and does not wake the replacement session.
- The live URL remains server-side. The public page may embed it, but the DM contains only the Hermes handoff URL.

## Security Properties

- Tokens contain at least 256 bits of randomness and are compared by digest using constant-time comparison.
- Tokens are single-purpose, single-session, one-time, and bounded to 30 minutes.
- No login is required, so anyone possessing the URL can control that browser during the validity window. The DM copy states that the link must not be forwarded.
- Query strings, bearer tokens, Browser Use live URLs, CDP URLs, and credentials are redacted from logs and error messages.
- The page cannot navigate its parent, access Hermes cookies, or call unrelated Hermes endpoints. Completion uses a same-origin POST and rejects cross-origin requests.
- Responses are never cached. The page sends no analytics or third-party telemetry beyond the Browser Use controller it intentionally embeds.
- The completion route validates current handoff state, expiry, profile scope, and browser-session binding before waking anything.
- Rate limiting applies per source IP and per handoff digest, while valid single-click completion remains reliable.

## Failure Handling

- No Browser Use live URL: Hermes reports that this backend cannot be handed off and does not send a DM.
- Public base URL missing or non-HTTPS: handoff creation fails closed with setup guidance.
- Discord DM failure: the new handoff is cancelled and the agent receives the delivery error.
- Browser expires while waiting: the handoff becomes expired, its page loses access, and the originating session is woken with an expiry result when possible.
- Done wake delivery fails: completion remains recorded and a retry worker re-attempts the internal wake; the button never reopens the capability.
- Duplicate Done requests: all but the first receive the already-completed page and cause no additional wake.
- Tunnel unavailable: the DM send may succeed but the page cannot load. Hermes keeps the pending row until expiry; a future operational health check can surface tunnel reachability without weakening token validation.

## Configuration

Behavioral settings live in `config.yaml`:

```yaml
browser:
  handoff:
    enabled: true
    public_base_url: "https://hermes.example.com"
    ttl_minutes: 30
    discord_user_id: "1063878950851448853"
```

`ttl_minutes` defaults to 30 and is capped at 30 for the initial implementation. The Discord bot credential remains a secret in Hermes's existing secret store; this feature introduces no new secret environment variable.

## Verification

Focused automated coverage will prove behavior rather than source layout:

- Browser Use provider create responses preserve `liveUrl`; other providers remain valid without it.
- Browser and daemon keys are stable within one bot/task but distinct across bots, including unnamed sessions and identical friendly session names.
- Token plaintext is absent from SQLite and logs.
- Pending GET renders the instruction and live controller; invalid, expired, completed, cancelled, and browser-mismatched tokens never expose `live_url`.
- Completion is atomic and concurrent Done requests schedule exactly one wake.
- Effective expiry is capped by both the configured 30 minutes and Browser Use `timeoutAt`.
- Discord user IDs are resolved into DM channels and delivery failure cancels the record.
- Session reset, interruption, browser cleanup, and provider expiry revoke pending handoffs.
- A persisted pending handoff survives service restart and completes against the exact raw originating session.
- The handoff routes work without dashboard login while unrelated routes remain protected.
- Response headers enforce no-store, no-referrer, and the intended CSP.

One end-to-end test will create a fake Browser Use session, request a handoff through the real tool/manager boundary against a temporary Hermes home, receive the DM through a test adapter, load the public page, press Done, and assert that exactly the originating session receives the continuation.

## Rollout Boundary

The first release supports Browser Use cloud sessions and one configured Discord owner DM. It does not support local Chrome remote desktop, multiple approvers, arbitrary messaging targets, approval decisions, file uploads, or automatic CAPTCHA solving. Those are separate features.

The change will ship through a production PR because it introduces a public bearer-authenticated route and cross-session wake behavior. It will not be merged without explicit owner approval.
