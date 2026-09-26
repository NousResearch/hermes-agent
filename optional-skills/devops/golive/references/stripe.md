# Stripe (payments): notes for Hermes

Load this when the plan uses `payments=stripe`.

Status: a disposable test-mode run passed end to end — sandbox identity bound into approval (a
restricted operator key plus a standard app key), env wiring verified, webhook endpoint registered
and re-checked, the unsigned probe rejected with 400, and a real test-card payment delivered a
signature-verified event (HTTP 200). Live-mode payments, refunds, entitlements and subscriptions
remain open.

## 1. Giving golive Stripe keys

Stripe has **no API that creates or hands out account API keys**, and golive does **not** use the
Stripe CLI's login (`stripe login` credentials are short-lived and may lack live write access). So the
human provides a secret key once per mode in use through private credential entry:

- **How:** on macOS, run `credentials --prompt STRIPE_TEST_SECRET_KEY --json` for test mode,
  or `credentials --prompt STRIPE_LIVE_SECRET_KEY --json` only when live mode is needed. The human
  enters the value in the native dialog; golive saves it locally and returns metadata only. Their own
  editor is the fallback if unavailable, unsupported, or preferred; follow
  [How the human connects accounts](../SKILL.md#how-the-human-connects-accounts) for fallback,
  replacement and cancellation. Never put a value in chat or argv, or tell them to export it in an
  unrelated terminal (that shell is not Hermes's). Use the exact variable name `doctor` identifies.
- **Which modes:** `payments.modes` in `golive.yaml`, default `{ preview: test, production: live }`, so
  by default both are needed:
  - `STRIPE_TEST_SECRET_KEY` (`sk_test_…` or `rk_test_…`)
  - `STRIPE_LIVE_SECRET_KEY` (`sk_live_…` or `rk_live_…`)
  - `STRIPE_SECRET_KEY` counts only for the mode its prefix matches, so one key never covers both.
- **No live key yet** (e.g. KYC not finished): set `payments.modes.production: test` in `golive.yaml`
  (the plan then warns that production takes no real charges), ship with test keys, and switch back
  to `live` later. While any needed key is missing, Stripe stays "not connected" and every Stripe
  step is left out of the plan.

### Operator key vs. app key

- The **operator key** (above) is what golive itself calls Stripe with. It may be a **restricted key**
  (`rk_…`) with **Webhook Endpoints: Write**, **Events: Read** and **Account: Read**.
- The **app key** is what golive writes into the app's `STRIPE_SECRET_KEY` on the host. It must be a
  standard `sk_<mode>_…` key. golive uses `STRIPE_APP_TEST_SECRET_KEY` / `STRIPE_APP_LIVE_SECRET_KEY`
  if set (an `rk_` or wrong-mode value there is rejected, with no fallback). Otherwise it uses the
  operator key, but **only if that is a standard `sk_` key**. A restricted operator key is never
  copied into the app. The app key is then a `stripe:secret-key:<mode>` handoff: the human adds it to
  the host themselves, or uses `credentials --prompt STRIPE_APP_TEST_SECRET_KEY --json` /
  `credentials --prompt STRIPE_APP_LIVE_SECRET_KEY --json` for the needed mode (same private-entry
  and editor-fallback rules).
- **Simplest path:** one standard `sk_` key per mode serves both golive and the app. A restricted
  operator key gives golive less power, but then the app key is a separate step.
- The operator key must successfully read the exact Stripe account. If Account: Read is denied
  or the account ID is missing, `doctor` blocks payments wiring even if webhook listing works.
  The approved account ID, mode and operator-key fingerprint bind each payment step. They are
  checked again before writes, and the step uses the captured credentials throughout.
- A separate app key must belong to the same approved Stripe account and mode. Account mismatch
  or unprovable identity stops before writing the key into hosting env. Changing accounts or
  rotating the planned credentials requires a new plan and approval.

## 2. What golive does vs. what stays with the human

golive automates (after plan approval; live-mode steps also need `--confirm-live`):
- **Keys per target** (`payments:keys:<target>`): writes the app key (§1) and the publishable key
  (from `payments.publishableKeys`, set with `init --stripe-publishable`) into the names the code
  reads, in each target's mode. A key rotation re-runs the step even though its preview text is the
  same (the key fingerprint is part of the step's intent).
- **The production webhook** (`payments:webhook:production`), for the production URL + webhook path
  with the events in `golive.yaml` (`init` defaults them to the events `detect` found in the handler;
  override with `--events a,b`). Endpoint choice, the same rule for `plan` and `apply`: one created by
  golive for this app, else any golive-tagged one (`metadata.managed_by=golive`), else any endpoint with
  the same URL (URL-normalised). An endpoint golive didn't create keeps its existing events (missing
  ones are added) and is never deleted. Drifted events or URL, or a disabled endpoint, are fixed in
  place.
- **In the same step**, writes the endpoint's signing secret (`whsec_…`) to the host's **Production**
  env only (usually `STRIPE_WEBHOOK_SECRET`). Stripe shows it only once, which is why it's one step.
  Preview gets no webhook and no signing secret (preview URLs change per deployment). golive records
  which endpoint the stored secret belongs to (its source includes the endpoint id). After a mode or
  URL round trip, the secret is replaced unless it provably belongs to the endpoint the URL resolves
  to now.
- **Before creating an endpoint** it asks the host whether it would accept the secret. If not (a var
  shared with other environments, an integration-owned var, production vars hidden from the login),
  the plan shows a blocking `stripe:webhook-env` handoff and creates nothing. If the write still fails
  after creating one, golive deletes the new endpoint and the error names it.
- **Production URL changed** (e.g. a custom domain added later): the endpoint golive created for the
  old URL is deleted after the new secret is stored (never one golive didn't create). The plan
  preview says so.
- **Guided host:** golive can't store the secret there, so a blocking `stripe:webhook-guided` handoff
  asks the human to create the endpoint in the dashboard and copy its `whsec_…` straight into the
  host's Production env (see `guided.md`).
- Verifies: `webhook-unsigned` (an unsigned POST gets 4xx; a non-HTML 401/403 only warns, since an auth
  wall looks the same as a handler rejecting the signature), `webhook-registered` (enabled endpoint,
  right URL, events covered), `stripe-live-ready` (the account can take live payments), and
  `env-parity` (key names present).

Stays with the human (and why):
- **Keys** (§1). Alternatives the `plan` handoff mentions: connect Vercel's Stripe integration (its
  vars are adopted, never overwritten), or the human pastes the key **directly into the host's
  dashboard**.
- **Publishable key.** Stripe's API can't hand it out either, so when the code reads one and
  `golive.yaml` has none for that mode, `plan` shows `stripe:publishable-key:<mode>`: ask the human
  for the `pk_test_…` / `pk_live_…` key. It is public (it ships in the browser bundle), so chat is
  fine for this one; never `sk_`, `rk_` or `whsec_`. Run `init --stripe-publishable <mode>=pk_<mode>_…` and `plan` again. Or the human adds it
  to the host dashboard.
- **Identity verification (KYC) for live payments** (`stripe:activate`). `stripe-live-ready` shows what
  Stripe still needs. Only the human can submit it; never try to fill it in.
- **Deleting old endpoints** golive didn't create, and cleanup at the 16-endpoints-per-mode limit.
- **Copying products to live.** "Copy to live mode" creates a new copy every time. Prefer prices with
  lookup keys in each mode.

## 3. Explain these in plain words

- **The signing secret is shown once.** If golive doesn't have it for an existing endpoint, the plan
  replaces the endpoint: create a new one with the same URL/events, write its secret, then delete the
  old one **only if golive created it**. An endpoint golive didn't create is left in place, and the
  change log says so; the human deletes it in the Webhooks tab in Workbench (until then Stripe
  also delivers to it, and those deliveries fail signature checks). The log says "old endpoint
  deleted" only when it was.
- **Test and live are separate worlds.** Same URL, different secrets. The secret from `stripe listen`
  on a laptop is different again. Mixing them gives "No signatures found matching the expected
  signature for payload".
- **One sandbox for everything.** Use a single sandbox (or the account's test mode) for keys,
  endpoints and event tests: golive cannot tell sandboxes apart, so a key from one sandbox and a
  signing secret from another silently disagree.
- **The handler must read the raw body.** Parsing JSON first breaks the signature. Next.js App Router:
  `await req.text()`. Express: `express.raw({type:'application/json'})` on the webhook route, mounted
  before `express.json()`.
- **Edge runtimes** (Vercel Edge, Cloudflare Workers, Deno, Supabase Edge Functions) must use
  `await stripe.webhooks.constructEventAsync(body, sig, secret, undefined,
  Stripe.createSubtleCryptoProvider())`. The non-async version throws there.
- **Supabase Edge Functions** reject Stripe by default (they expect a Supabase JWT). Set
  `verify_jwt = false` for that function; `detect` notes it. See `supabase.md`.
- **Vercel Deployment Protection** blocks Stripe on protected URLs. The webhook goes to the production
  domain or the project's public production alias.
- **Redirects count as failures.** Register the final URL (www vs. bare domain, trailing slash).
- **Middleware** (CSRF, auth) must skip the webhook route.
- A **static export** (e.g. Next.js `output: 'export'`) has no server, so it can't receive webhooks.

## 4. Troubleshooting

| Symptom | What to do |
|---|---|
| `doctor`: Stripe not connected, a key is missing | The human adds the per-mode key(s) `doctor` names to the credentials file (§1). `stripe login` doesn't help. No live key yet: `payments.modes.production: test`. |
| "`STRIPE_LIVE_SECRET_KEY` holds a test-mode key" (or similar) | Wrong key in that line; replace it with the right mode's key. |
| Stripe rejected the key (401) | Expired, revoked or mistyped. Copy a current key from the API keys page (dashboard.stripe.com/apikeys) into the credentials file. |
| Permission error (403) with a restricted key | Grant Webhook Endpoints: Write, Events: Read and Account: Read, or use the standard key. |
| `stripe:secret-key:<mode>` handoff with a restricted key | Expected: restricted keys are never given to the app. Set `STRIPE_APP_<MODE>_SECRET_KEY` (standard `sk_` key) or the human adds the app's key to the host. |
| "No signatures found matching the expected signature for payload" | Wrong secret (test vs. live, `stripe listen`, old endpoint), or the body was parsed before verifying. |
| `webhook-unsigned` gets 2xx | The handler doesn't verify signatures. Security fix in code, then re-run `verify`. |
| `webhook-unsigned` warns on a 401/403 | Either an auth wall in front of the route (Supabase `verify_jwt`, Deployment Protection) or a handler that returns 403 for bad signatures on purpose. Fix the wall, or accept the warning. |
| `webhook-unsigned` gets 404/405 | Wrong path. Fix the route, or `init --webhook-path <path>`. |
| `webhook-unsigned` gets 3xx | Register the canonical URL. |
| `webhook-registered` fails on events | `golive.yaml` events differ from the endpoint: `plan` + `apply` fixes it. If the handler needs more events, `init --events a,b` first. |
| "Use `await constructEventAsync(...)`" | Edge runtime; switch to the async form above. |
| `stripe-live-ready` fails (`charges_enabled` false) | KYC handoff. Show `requirements.currently_due` in plain words; the human completes it in the Dashboard. |
| Endpoint limit (16 per mode) | The human deletes unused endpoints in the Webhooks tab in Workbench (stale golive ones are tagged `managed_by=golive`), then re-run. |
| Apply says the webhook endpoints changed since approval | Someone changed endpoints meanwhile. Run `plan` again and get approval again. |
| `stripe:webhook-env` handoff (host would refuse the secret) | Fix the var at the host as its `action` says (e.g. split a multi-environment var per environment in the Vercel dashboard), then `plan` again. |

## Unverified

- Whether live webhook endpoints can be created before KYC is complete.
- The exact restricted-key toggle name for reading one's own account.
- Whether Stripe Projects can deliver the user's live key to a host.
