# Per-site playbooks (pre-recorded game plans)

Field-verified game plans so the agent **executes** rather than re-discovers each run: does an
in-browser search/opt-out work, what removal lanes exist, which is optimal, and the known gotcha or
end-state. This is the durable memory for sites that do not have their own `references/brokers/<id>.json`
record (the per-broker JSONs are the memory for the ones that do).

**Policy lives in `methods.md`** (blind-opt-out default, the blocked-form email-fallback decision order,
and the CAPTCHA policy). This file is the site matrix + skip-lists + backend clusters. When you learn a
site's mechanics, add or correct a row here (and promote it to a broker JSON if it becomes a recurring
removal target).

## Blocked-tail pass matrix (worked 2026-07-03)

| Site | In-browser search? | Best lane | Field-verified gotcha / end-state |
|---|---|---|---|
| **PropertyRecs** | yes (real listing) | in-browser **one-click remove** | Form is a single **Full-Name** field + a **City/State** field (NOT first/last). No email verify, no CAPTCHA. Confirms "Success: Information Removed". Cleanest removal of the batch. |
| **CheckPeople** | the flow **is** the search | in-browser **guided flow** | email -> verify link `/opt-out/dob/<token>` (from `info@checkpeople.com`) -> DOB (immutable) -> legal name -> Matching Records. "Unable to find any results that matched your name and date of birth" is a **strong `not_found`** (the broker ran its own matcher). |
| **InfoTracer** | form gated | **email** `privacy@infotracer.com` (cited on `/optout`) | Form `members.infotracer.com/removeMyData` has a **slide-to-verify** slider (do NOT defeat). The cited email is a working Zendesk lane (ack + ticket #). **InfoPay backend.** |
| **SpyFly** | form gated | **email** `deletemyinfo@spyfly.com` | `/help-center/remove-my-public-record` has a **Cloudflare Turnstile that will not clear**. Privacy policy lists only a form + phone, but the `deletemyinfo@` alias is a working deletion lane. |
| **ZoomInfo** | email-gated | submit email (no-op if no profile) | "IF your email matches a profile we will send instructions." No instructions email = no profile. B2B **work-contact** DB; residential-footprint subjects generally do not match. |
| **UnMask** | guided flow (stuck) | **human task** | PeopleConnect-family Suppression Center; step-1 "email sent" shown **twice**, but the verify email **never delivers** (checked 2h incl. spam/all-mail) -> broker-side delivery failure, needs a human retry. |
| **PrivateRecords** | form (blocked) | **blocked** | `/api/helper/optOutLight/search` -> **double CAPTCHA** (distorted-text image THEN arithmetic) -> still rejected "Captcha verification failed" (it is fingerprinting the automation, not grading the answer). No cited rights-email (policy only has an unsubscribe link). |
| **SearchQuarry** | none acceptable | **do NOT pursue** (human decision) | Same **InfoPay** slide-to-verify slider as InfoTracer; FAQ states removals are processed **only** by a mailed/faxed form + a copy of a gov-ID. Violates least-disclosure -> surface as a human decision, do not pursue autonomously. |

Also recorded `not_found` this pass via operator manual check (`operator_manual_check` evidence note):
**ClustrMaps, PeekYou, NeighborReport** (404 / dead), **USA People Search** (no results),
**BeenVerified** (no results; an optional preventive deletion email to the controller was left on the
table). A dead/404 site or an operator-confirmed "no results" search is a valid `not_found`.

## Backend clusters (one operator's behavior predicts the others)

- **InfoPay backend** = **InfoTracer + SearchQuarry** (and other InfoPay-run sites): identical
  `InfoPay_Core_Components_OptOuts_*` form fields and the same **slide-to-verify** slider. If one shows
  the slider, expect it on the rest -> go straight to the email lane (where cited) or skip.
- **PeopleConnect / Intelius front-end** = **addresses.com** (report links -> `tracking.intelius.com`).
  Covered by the cluster suppression (`addresses` is in `intelius.owns`); no separate opt-out. See
  `brokers/intelius.json` and `brokers/addresses.json`.

## Meta-search / link-out aggregators -- do NOT file opt-outs (no-ops)

These hold **no first-party data**; they interpolate the name into social-search URLs and show affiliate
links to brokers we already handle (Spokeo / TruthFinder / BeenVerified). They clear when the underlying
brokers do. Record `not_found` and move on; do **not** add them as broker records or file removals:

> **IDCrawl, Lullar, Yasni, WebMii, Namesdir, iTools, Skipease.**

## Triage before scanning (taxonomy)

When cross-checking any external "people OSINT" list (e.g. an OSINT Radar catalog), separate:

1. **First-party data brokers** -> removal targets (scan / blind opt-out).
2. **Meta-search / link-out aggregators** -> no-ops (skip-list above).
3. **Cluster front-ends** -> covered by a parent (e.g. addresses.com -> Intelius); do not double-file.
4. **Non-broker tools / APIs / wrong-jurisdiction** -> skip: PhoneInfoga (a tool), People Data Labs
   (dev API), Truecaller (login-gated app), Canada411 / 192.com (CA / UK jurisdiction).
