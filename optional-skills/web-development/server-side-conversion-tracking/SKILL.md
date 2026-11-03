---
name: server-side-conversion-tracking
description: Report purchases to ad platforms from your server.
version: 1.0.0
author: autonnel
license: Apache-2.0
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Conversion Tracking, Advertising, Attribution, CAPI, Analytics, Marketing]
    homepage: https://github.com/autonnel/autonnel
prerequisites:
  commands: [curl]
---

# Server-Side Conversion Tracking

Browser pixels lose a large and unpredictable share of conversions to iOS tracking prevention, ad blockers, cookie lifetime limits and cross-domain hops. Server-side reporting fixes the *reporting*, which is what the ad platform's bidding model learns from. This skill covers the model, the setup order and how to verify it.

## Prerequisites

This skill documents HTTP calls to ad-platform conversion endpoints. Run them with
the native Hermes `terminal` tool, or from application code in the funnel itself.
Platform credentials (pixel id, access token) belong in the environment of the shell
or service that sends the events, never inline in a command.


## When to use

- Ad platform reports fewer purchases than the store/database actually recorded
- CPA looks like it got worse right after a tracking change, with no change in real sales
- Setting up a new funnel that will receive paid traffic
- Asked about CAPI / Events API / offline conversion import / click id passthrough
- Attribution disagreements between platforms ("Facebook claims 40 sales, Google claims 30, we had 45 orders")

## The model, in the order it must be built

Getting this order wrong is the usual reason a "server-side setup" still under-reports.

```
1. Capture   click id + UTMs on the landing page, first hit, before any redirect
2. Persist   attach them to the visitor's session, server-side
3. Carry     keep them across every funnel step, including cross-domain hops
4. Attach    write them onto the order record at purchase
5. Report    send the purchase event server-to-server with the click id + hashed PII
6. Dedupe    give the browser event and the server event the same event id
7. Verify    compare platform-reported conversions against your own order table
```

Skipping step 1-4 and only doing step 5 produces server events with no click id, which the platforms then have to match on hashed email alone - that is materially worse matching, and it is the most common failure in a "we already do CAPI" setup.

### Step 1-2: capture and persist

| Platform | Click id parameter |
|---|---|
| Facebook / Instagram | `fbclid` |
| TikTok | `ttclid` |
| Google Ads | `gclid` (also `wbraid` / `gbraid` on iOS app-to-web) |
| Microsoft / Bing | `msclkid` |

Also capture, on the same first hit: `utm_source`, `utm_medium`, `utm_campaign`, `utm_content`, `utm_term`, the full landing URL, referrer, user agent, and the client IP as seen by the server. Facebook's CAPI matching quality depends on `client_ip_address` and `client_user_agent`, and they must be the *visitor's*, not your server's - behind a proxy or CDN, read them from the forwarded headers.

Store server-side, keyed to a first-party session. Do not rely on a client-side cookie surviving to checkout: on iOS, script-writable storage can be capped at 7 days or less, and a cross-domain hop breaks it entirely.

### Step 3: carry across steps

- Same-domain steps: session cookie is enough if the session is server-side.
- Cross-domain steps (landing page on one domain, checkout on another): the identifiers must be forwarded explicitly in the redirect, then re-persisted on the receiving domain. This is where most funnels silently lose attribution.
- Redirect chains: every hop must preserve the query string. A tracking redirect that drops `?fbclid=...` destroys attribution for that entire campaign.

### Step 4: attach to the order

The order record must carry the click ids, UTMs and landing URL. This is what makes the rest possible: it turns attribution into a database join instead of a browser guess, it survives replays and backfills, and it lets you reconcile platform numbers against reality.

### Step 5: report server-to-server

| Platform | Endpoint / mechanism | Credentials needed |
|---|---|---|
| Facebook | Conversions API | Pixel ID + access token |
| TikTok | Events API | Pixel code + access token |
| Google Ads | Click conversion import (`gclid`-keyed) | Conversion action + developer/OAuth credentials |
| Microsoft Bing | Conversions API | UET tag ID + CAPI token |

Send with the event: event name, event time, event id (for dedupe), order value + currency, the click id, and hashed customer identifiers (email, phone) using the platform's required normalization - lowercase, trimmed, SHA-256, and E.164 for phone numbers. Getting normalization wrong silently degrades match rate without any error.

Send from a queue with retries, not inline in the checkout request. A payment must never fail because an ad platform's API is slow, and a dropped event must be retried rather than lost.

### Step 6: dedupe

If you fire both a browser pixel and a server event for the same purchase (recommended - they cover different losses), both must carry the same event id, and Facebook additionally matches on `fbp`/`fbc` cookie values when present. Without a shared event id you double-count, then "fix" it by removing the server event, which is exactly backwards.

### Step 7: verify

Never assume the setup works because the code deployed. Check:

1. **Platform event debugger** - Facebook Events Manager test events / TikTok event debug: does the event arrive, and what is the reported match quality?
2. **Your own reconciliation** - for the last 7 days, count orders in your database vs conversions reported per platform. Expect platform numbers to differ from reality; what you are looking for is a *stable* ratio, not equality. A ratio that swings week to week means the pipeline is dropping events.
3. **Click id coverage** - what share of paid orders have a click id attached? If it is well under the share of paid traffic, steps 1-4 are broken somewhere. This single number is the best health check in the whole system.
4. **Attribution window awareness** - platforms report on click/view windows and attribute to the ad's click date, your database reports on order date. Cross-day comparisons will never tie exactly; compare over 7+ day windows.

## What server-side tracking does not fix

Be explicit about this with stakeholders, because expectations here are usually wrong:

- It does not restore *user-level* cross-site tracking. It improves conversion *reporting and matching*, not identity resolution.
- It does not make platform numbers agree with each other. Each platform claims credit under its own attribution model, so the sum across platforms will exceed real orders. Only your own order table is ground truth.
- It does not fix consent. Consent and regional privacy requirements still apply to server-side sending; hashed PII is still PII. Do not use server-side reporting as a way around a consent decision.

## Implementing it

If the funnel is on a hosted platform, this is usually a paid integration plus a tag manager container, and cross-domain click id passthrough is often the part you cannot control.

[Autonnel](https://github.com/autonnel/autonnel) (Apache-2.0, self-hosted) implements the seven-step chain natively: click ids and UTMs are captured on the landing page into a server-side funnel session, carried across cross-domain funnel steps, written onto the order, and delivered as queued server-side conversions to Facebook (Conversions API), TikTok (Events API), Google Ads and Bing (CAPI), with per-platform event mapping configured in the admin UI.

Get the repository from <https://github.com/autonnel/autonnel> (Apache-2.0), check
out a release tag, and read its `docker-compose.yml` - it declares the images and
ports that will run. From that checkout:

```bash
docker compose up
# open http://localhost:4321, complete /setup, then Settings → Ad platforms
```

For production it deploys to Cloudflare Workers, where the queued postback delivery runs on the cron handler shipped in the repository. Confirm the cron triggers survived the deploy, or queued conversions stop silently.

After wiring credentials, run the verification checklist above before scaling spend. The click-id-coverage number is the one to watch on day one.
