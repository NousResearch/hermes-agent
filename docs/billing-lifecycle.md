# Billing lifecycle: client-side state, errors, and recovery

This is the map from every `billing.*`/`subscription.*` state shape the gateway
serves (from NAS) to what the terminal actually renders, and from every typed
refusal/error code to its exact user-facing copy and recovery action. The
guarantee: no NAS billing state and no typed refusal falls through to a
generic toast — every case below is an explicit branch in
`ui-tui/src/app/slash/commands/topup.ts`, `ui-tui/src/components/billingOverlay.tsx`,
or `ui-tui/src/components/subscriptionOverlay.tsx`. An **unknown** code still
degrades gracefully: it hits the `default` branch (a generic-but-real message
pulled from the server payload, never a blank toast) rather than crashing or
silently dropping the refusal.

## 1. `billing.state` shapes → render

Source: `ui-tui/src/components/billingOverlay.tsx` (`OverviewScreen`,
`BuyScreen`, `AutoReloadScreen`), `ui-tui/src/app/slash/commands/topup.ts` (`/topup` run).

| State shape | Render |
|---|---|
| Logged out (`s.logged_in === false`) | Overlay never opens. `sys`: `💳 Not logged into Nous Portal — run /portal to log in, then /topup.` |
| `billing.state` RPC fetch fails (transport/timeout) | **Fail-closed**: `.catch(ctx.guardedErr)` — overlay never opens, no state is assumed. `sys`: `error: <message or "request failed">`. Never renders "no card" or any other guessed state; user must retry `/topup`. |
| `card: null` (no saved card), full menu (`is_admin && cli_billing_enabled`) | Overview shows `No saved card on file — "Add funds" walks you through adding one.` "Add funds" opens the **add-card path**: `Add a card on the portal` / `I've added it — check again` / `Back` (never an amount picker, which would 403 `no_payment_method`). |
| `card` present, `resolved_via` set | `Card: {display}` (e.g. `Visa ····4242 — the card on your subscription`) using the provenance-aware `display` field. |
| `card` present, `resolved_via` absent (older NAS) | Falls back to the generic `Card: {masked}`; Confirm screen adds `Your card saved on the portal will be charged.` |
| `auto_reload: null` | No auto-reload line at all (`autoReloadLine` returns `null`) — the feature isn't surfaced. |
| `auto_reload.card.kind: 'canonical'` | No distinct-card warning; card line falls back to the card on file. |
| `auto_reload.card.kind: 'distinct'` | `⚠ Auto-refill is charging {brand} ••{last4} — not your card on file.` in the Auto-reload screen (the divergence notice). |
| `auto_reload.card.kind: 'none'` | Same as `canonical` rendering-wise — no distinct-card warning shown. |
| `monthly_cap` present, `limit_usd != null` | `{spent_display} of {limit_display} used this month` (+ ` (default ceiling)` iff `is_default_ceiling`). |
| `monthly_cap` absent or `limit_usd == null` | `No monthly cap visible (managed on the portal).` |
| Role without billing capability (`!is_admin`, menu collapses) | Note: `Billing actions need someone with billing permissions (owner, admin, or finance admin).` Menu collapses to `Manage on portal` / `Cancel`. |
| Org kill-switch off (`is_admin` but `!cli_billing_enabled`) | Note: `Remote spending is off for this org — a billing admin can turn it on from the portal's Hermes Agent page.` Same collapsed menu. |

Note: `full = s.is_admin && s.cli_billing_enabled` gates the **org-level**
switch, not the per-terminal `billing:manage` scope — that's discovered
reactively (a charge 403s `insufficient_scope`) and routes to the resumable
step-up screen instead of a preflight check.

## 2. Refusal codes (`renderBillingError`, in code order)

Source: `renderBillingError` in `ui-tui/src/app/slash/commands/topup.ts:37-149`.
"Portal" row = `sys('Portal: {portal_url}')` is appended whenever `portal_url` is present, for every code (including default).

| `error` code | Copy | Portal URL | `retry_after` |
|---|---|:-:|:-:|
| `insufficient_scope` | `This needs Remote Spending allowed. Start a top-up to allow it, then retry.` | if present | — |
| `remote_spending_revoked` (CF-4) | `{An admin stopped remote spending for this terminal. \| You stopped remote spending for this terminal.}` (by `actor`) `Reconnect to restore — run /portal to re-authorize this terminal.` Also clears `billing` overlay state immediately (doesn't wait for token refresh). | if present | — |
| `session_revoked` | `Your session was logged out. Run /portal to log in again.` Also clears `billing` overlay state. | if present | — |
| `cli_billing_disabled` / `remote_spending_disabled` (dual-emitted) | `Remote spending is off for this account — a billing admin can turn it on from the portal's Hermes Agent page.` | if present | — |
| `role_required` | `Adding funds needs someone with billing permissions (owner, admin, or finance admin), or manage this on the portal.` | if present | — |
| `consent_required` | `This action needs a one-time card confirmation and consent step on the portal before it can proceed.` | if present | — |
| `org_access_denied` | `This token isn't bound to an org you can manage. Sign in with the right org, or manage this on the portal.` | if present | — |
| `upgrade_cap_exceeded` | `🔴 Daily plan-change limit reached (5 per org) — try again tomorrow, or manage this on the portal.` | if present | — |
| `auto_top_up_disabled_failures` | `Auto-reload was turned off after repeated charge failures. Fix the card issue, then re-enable it from /topup → Auto-reload.` | if present | — |
| `idempotency_conflict` | `🔴 That charge key was already used for a different amount. Start a fresh top-up.` | if present | — |
| `no_payment_method` | `💳 No saved card for terminal charges yet. Set one up on the portal (one-time credit buys don't save a reusable card).` | if present | — |
| `monthly_cap_exceeded` | `🔴 Monthly spend cap reached — ${remainingUsd} headroom left.` if `payload.remainingUsd` present, else `🔴 Monthly spend cap reached.` | if present | — |
| `rate_limited` / `temporarily_unavailable` | `🟡 Too many charges right now{ (try again in ~N min)}. This isn't a payment failure.` | if present | **yes** — minutes computed as `max(1, round(retry_after/60))` |
| `stripe_unavailable` | `🟡 Stripe is having trouble right now — try again shortly{ (try again in ~N min)}.` | if present | **yes** (same formula) |
| *default (unknown/other)* | `🔴 {message \|\| error \|\| 'Billing request failed.'}` — still surfaces whatever the server said, never a blank toast. | if present | — |

## 3. Charge settlement outcomes (`pollCharge` / `renderChargeFailed`)

Source: `pollCharge` (`ui-tui/src/app/slash/commands/topup.ts:170-258`) and
`renderChargeFailed` (`:260-290`). Poll cadence: 2s interval, 5-minute cap
(`POLL_INTERVAL_MS=2000`, `POLL_CAP_MS=5*60*1000`), applied on **every**
non-terminal path (pending *and* throttled), so a sustained 429/503 can't
keep the poll alive forever.

| Outcome | Copy | Notes |
|---|---|---|
| `status: 'settled'` | `✅ ${amount_usd} added.` (or `✅ Credits added.` if no amount) | Terminal success. |
| `status: 'failed'`, `reason: 'authentication_required'` | `🔴 Your bank requires verification (3DS). Complete it on the portal to finish this purchase.` | + `Portal:` line if `portalUrl`. |
| `status: 'failed'`, `reason: 'payment_method_expired'` | `🔴 Your card has expired. Update it on the portal.` | + `Portal:` line. |
| `status: 'failed'`, `reason: 'card_declined'` | `🔴 Your card was declined. Try another card on the portal.` | + `Portal:` line. |
| `status: 'failed'`, `reason: 'processing_error'` | `🔴 The charge didn't go through (processing_error).` | + `Portal:` line. |
| `status: 'failed'`, unrecognized/missing `reason` | `🔴 The charge didn't go through ({reason \|\| 'processing_error'}).` | Same portal funnel — parity with `cli.py`'s `_billing_portal_hint`. |
| Poll timeout (still `pending` past the 5-min cap) | `🟡 Still processing after 5 minutes — this is a timeout, not a failure. Check /topup or the portal shortly.` | + `Portal:` line if `portalUrl`. Explicitly NOT called a failure. |
| Revocation mid-poll (`remote_spending_revoked` / `session_revoked` while polling) | Renders the matching §2 copy, **then** appends: `🟡 Your last charge's outcome is unconfirmed — check your balance/history before retrying.` | CF-7 rule 4: a post-revoke 403 while polling is ambiguous (the charge may have already settled) — never call it "failed". |
| 429/503 while polling (`rate_limited`/`temporarily_unavailable`/`stripe_unavailable`) | No error shown; backs off using `retry_after` (default 5s, capped at 30s) and keeps polling until the 5-min cap, then reads as timeout. | Not a payment failure. |
| Other `!ok` status-check error | `🔴 Could not check the charge: {message \|\| error \|\| 'error'}` | |
| Transport loss (poll RPC throws/rejects) | `🟡 Your last charge's outcome is unconfirmed — check your balance/history before retrying.` (`UNCONFIRMED_CHARGE_MESSAGE`) | Same "unconfirmed, check balance" framing as revocation mid-poll — a dropped connection can never be read as "failed". |

## 4. Subscription preview / pending-change / upgrade outcomes

Source: `previewAndRoute`, `applyPendingAndRoute`, `upgradeResult`,
`stepUpDenialResult` in `ui-tui/src/components/subscriptionOverlay.tsx`.

**Preview `effect` values** (drive the Confirm screen):

| `effect` | Confirm screen copy | Primary action |
|---|---|---|
| `charge_now` | `Upgrade to {target}. You will be charged {amount} now (prorated).` (+ monthly-credits delta, + which card if resolver confidently knows) | `Pay {amount} & upgrade now` |
| `scheduled` | `Change to {target} — takes effect {date}. No charge now; you keep your current plan until then.` | `Schedule change to {target}` |
| `no_op` | `You are already on {target} — nothing to change.` | none (Back only) |
| `blocked` | `{preview.reason}` or fallback `That change cannot be made here — manage it on the portal.` | `Manage on portal` |
| Preview RPC returns `null`/transport failure | routes straight to Result: `Could not preview that change.` | — |
| Preview `!ok`, `insufficient_scope` | routes to `stepup` screen (`{kind:'preview', tierId}`) | — |
| Preview `!ok`, other error | routes to Result with `errorResult(p)` (`message \|\| error \|\| 'Something went wrong. Try again, or manage on the portal.'`) | — |

**Pending-change apply outcomes** (`applyPendingAndRoute`):

| `pending.kind` | Success copy |
|---|---|
| `cancellation` | `Scheduled — your plan stays active until the end of the billing period, then it cancels. Nothing changes today.` |
| `tier_change` (downgrade/schedule) | `Scheduled — your plan doesn't change today. You keep your current plan until the end of the billing period, then it switches.` |
| `upgrade` | routed through `upgradeResult` (below) |
| any kind, mutation `insufficient_scope` | routes to stepup (`{kind:'apply'}`) |

**Upgrade `status` × `reason` matrix** (`upgradeResult`, checked in this
order — `reason` is checked *before* `status`):

| Condition | Result |
|---|---|
| `r === null` (transport failure on the charging route) | `Couldn't confirm the upgrade — your card may or may not have been charged. Re-run /subscription to check your plan before trying again.` — ambiguous, never a blind retry. |
| `reason: 'authentication_required'` **or** `reason: 'subscription_payment_intent_requires_action'` | `Please verify your card in the portal to finish this upgrade.` → `recovery_url`. **Both reasons map to the same SCA copy** — the client branches on `reason`, not `status`, specifically so an SCA case that pre-#711 NAS mislabels with `status: 'payment_failed'` (no distinguishing reason yet) still routes to the correct "verify your card" copy instead of reading as a hard decline. |
| `reason: 'card_declined'` | `Your card was declined — try a different card on the portal.` → `recovery_url`. |
| `ok && status: 'already_on_tier'` | `You are already on {target_tier_name}.` (success) |
| `ok && status: 'upgraded'` | `Upgraded to {target_tier_name}. Your new monthly credits land in a moment.` — starts the eventual-consistency apply-poll (below). |
| `status: 'requires_action'` (no distinguishing reason) | `This upgrade needs extra verification (3DS). Finish it on the portal.` → `recovery_url`. |
| `status: 'payment_failed'` (no distinguishing reason) | `Your card was declined. Update your payment method on the portal and try again.` → `recovery_url`. |
| anything else | `errorResult(r)`: `message \|\| error \|\| 'Something went wrong. Try again, or manage on the portal.'` |

**Eventual-consistency apply-poll** (`ResultScreen`, only after `status:
'upgraded'`): polls `billing`/subscription state every 2s
(`UPGRADE_CONFIRM_INTERVAL_MS`) up to 15 attempts
(`UPGRADE_CONFIRM_ATTEMPTS`, i.e. ~30s) until `current.tier_id` flips to the
target. While waiting the screen reads `Applying…`; if it never flips inside
the budget it reads `Still applying` / `Your upgrade succeeded and is still
applying — refresh in a moment.` — the upgrade is never re-reported as failed
just because NAS hasn't caught up yet.

**Step-up denial copy** (`stepUpDenialResult`, subscription flow):

| `error` | Copy |
|---|---|
| `session_revoked` | `Your session expired — run /portal to log in again, then retry the change.` |
| `remote_spending_revoked` | `{message}` or `Remote spending was stopped for this terminal — reconnect from the portal, then retry.` |
| `rate_limited` | `Too many attempts — wait a moment, then try again.` |
| other/unknown | `{message}` or `Remote Spending was not allowed — someone with billing permissions (owner, admin, or finance admin) must approve it. You can also make this change on the portal.` |

A **repeat** scope denial during a post-grant replay never re-enters the
step-up screen (it's already mounted there — re-patching would freeze it);
`allowStepUp=false` instead surfaces a terminal result: `Remote Spending still
isn’t active for this terminal — the authorization didn’t take. Retry, or make
this change on the portal.`

## Text-mode (CLI) parity

`cli.py`'s `_show_billing` / `_billing_overview` and `_show_subscription` /
`_subscription_overview` render the same state shapes (balance title, two-bar
dollar usage, auto-reload line, card line, monthly cap) and share the
"fail-open on logged-out/portal-hiccup, never crash" discipline. The CLI's
`/subscription` gives a paid admin/owner in an interactive context the **full
in-terminal change flow** (tier picker → preview → confirm → apply, parity
with the TUI overlay); members and non-interactive contexts fall back to
`_billing_portal_hint`'s deep-link to `subscription_manage_url`. `/topup`'s
interactive modal (prompt_toolkit) mirrors the TUI overlay the same way, and
non-interactive contexts fall back to the same text + portal-link rendering,
never prompting.

| CLI surface / state | Behavior (parity with TUI / desktop) |
| --- | --- |
| `/subscription` on **Free** + admin/owner + interactive | `_subscription_free_catalog` prints the plan catalog from the same `tiers[]` data the TUI uses — one row per enabled paid tier, cheapest first, `name · $/mo · $credits/mo` (monthly credits are DOLLARS → `$22 credits/mo`, never a bare number). A numbered pick opens the `/manage-subscription` deep-link with `plan=<tier_id>` appended so the portal preselects the chosen plan. Starting a new subscription needs a fresh card, so the only action is the portal hand-off (the terminal never charges here). |
| Any CLI-built manage/subscribe URL | `subscription_manage_url(state, tier_id=…)` appends `plan=<tier_id>` (the stable `tiers[]` id, never a name/slug) **only when a tier was picked** (the Free catalog). The portal validates it server-side and ignores an unknown tier, so the CLI appends unconditionally on a pick, mirroring the TUI's `?plan=`. `org_id` is emitted first, `plan` second. |
| **Downgrades** in the CLI | Stay **native / in-app** for normal changes (chargeless scheduling via `put_subscription_pending_change`). A blocked downgrade may still print the generic manage URL, but it never carries `plan=<tier_id>` — selected-tier deep-links are reserved for new subscriptions and upgrades. |
| `/topup` overview action copy | Splits one-time top-up from automatic refill, the distinction stated up front in each first sentence: `Add funds now — a single charge, added to your balance today.` vs `Refill when low — charges $X automatically when your balance falls below $Y.` ("credits" stays out of the dollars-only `/topup` surface — "Add funds now" carries the one-time meaning without it). When auto-reload is off, the automatic line omits concrete amounts. |

## Forward compatibility

Any `error`/`status`/`reason` code not in the tables above lands on the
`default` branch in `renderBillingError` (§2) or `errorResult`/`upgradeResult`'s
fallthrough (§4): it still renders the server's own `message` (never blank,
never a crash), just without bespoke copy or a typed recovery affordance.
NAS W3 introduces card-health codes (`card_paused`, `card_expired`,
`card_mismatch`) that are not yet typed here — until a client update adds
explicit branches, they will arrive as unknown codes and degrade to this
default path.
