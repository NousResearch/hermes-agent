---
name: express-bus-booking
description: Use when searching or assisting Korean 고속버스/KOBUS bookings, including 고속버스 예매, 고속버스 시간표, 프리미엄/우등 고속버스, terminal/route lookup, seat availability, temporary seat holds, and assisted checkout-entry links. Prefers official HTTP/API-first flows over browser automation.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [korea, bus, travel, booking, kobus, transportation]
    related_skills: [intercity-bus-booking]
---

# Express Bus Booking (KOBUS)

## Overview

Use this skill for Korean 고속버스 / KOBUS timetable lookup and reservation assistance. The preferred workflow is HTTP/API-first: resolve routes, query timetables, inspect remaining seats and fares, and only use browser automation when endpoint discovery or an official web-only step blocks progress.

This skill is intentionally separate from 시외버스. KOBUS terminal codes, route structures, and checkout pages differ from Tmoney 시외버스. Use `intercity-bus-booking` for 시외버스/Tmoney flows.

## When to Use

- The user asks for `고속버스 예매`, `고속버스 시간표`, `고속버스 예약`, `프리미엄 고속버스`, or `우등 고속버스`.
- The route is clearly a KOBUS 고속버스 route or the user names KOBUS/코버스/고속버스통합예매.
- The user wants assisted reservation up to seat selection, temporary hold, or a payment-entry link.

Do **not** use this for:

- 시외버스/Tmoney routes — use `intercity-bus-booking`.
- Final card submission or payment without explicit, narrowly scoped confirmation.
- Blind browser automation before trying the official HTTP flow.

## Core Principles

- Use official KOBUS surfaces: `https://www.kobus.co.kr`.
- Keep a cookie jar and reuse referers. Stateless one-off POSTs are less reliable.
- Prefer desktop User-Agent and HTTP/1.1 during probing if the server behaves differently under HTTP/2.
- Default scope is assisted manual checkout: lookup, candidate presentation, seat-stage readiness, temporary seat hold, and official payment-entry page handoff.
- Do not submit card fields, agree to terms, or complete payment unless the user explicitly confirms that exact action.

## Known HTTP Flow

See `references/kobus-http-flow.md` for session-proven endpoint details and parameter examples.

### 1. Start a Session

Fetch either the main page or route/search page with cookies enabled:

```text
GET https://www.kobus.co.kr/main.do
GET https://www.kobus.co.kr/mrs/rotinf.do
```

Use a realistic desktop User-Agent and a cookie jar.

### 2. Resolve Route / Terminal Candidates

```text
POST /mrs/readRotLinInf.ajax
```

The response is JSON and can include `rotInfList`, `tfrInfList`, `len`, and `codeYn`. Prefer code-confirmed routes over display-name matching because terminal names are not always unique.

### 3. Query Timetable

```text
POST /mrs/alcnSrch.do
```

Typical fields include:

```text
deprCd=010
arvlCd=700
pathDvs=sngl
pathStep=1
deprDtm=YYYYMMDD
busClsCd=0
rtrpChc=1
timeLinkMin=00
timeLinkMax=23
```

Parse the returned HTML for schedule rows/cards and `fnSatsChc(...)` onclick arguments. Respect `mrsPsbYn=N` or any equivalent booking-disabled marker.

### 4. Enter Seat / Fare Stage

```text
POST /mrs/satschc.do
```

Send the original search form fields plus selected values from `fnSatsChc(...)`, commonly including:

```text
deprTime
alcnDeprTime
alcnDeprTrmlNo
alcnArvlTrmlNo
indVBusClsCd
cacmCd
dcDvsCd
prvtBbizEmpAcmtRt
chldSftySatsYn
dsprSatsYn
```

The response should include `form#satsChcFrm` and hidden fare/seat values such as `adltFee`, `rmnSatsNum`, `totSatsNum`, and terminal/time fields.

### 5. Temporary Seat Hold

```text
POST /mrs/setPcpy.ajax
```

Submit the `satsChcFrm` hidden fields plus selected seat/count fields. A successful response includes:

```text
MSG_CD=S0000
pcpyNoAll
satsNoAll
ESTM_AMT
DC_AMT
TISSU_AMT
```

Treat the hold as short-lived. If testing, if the user does not proceed, or if a new seat is chosen, release it explicitly.

### 6. Release Temporary Hold

```text
POST /mrs/cancPcpy.ajax
```

Use the same relevant form fields plus the returned `pcpyNoAll` and `satsNoAll`. A successful response returns `MSG_CD=S0000`.

## Checkout-Entry Link Helper

A plain official checkout URL is not enough because KOBUS expects a POST body containing the selected schedule, seat, fare, and hold identifiers. The practical user-facing pattern is:

1. Create the temporary hold server-side via `setPcpy.ajax`.
2. Generate a short helper page that auto-submits a POST form to the official KOBUS checkout endpoint.
3. Send the helper link to the user.
4. The user completes card/payment fields manually on the official KOBUS page.

Desktop action:

```text
https://www.kobus.co.kr/mrs/stplcfmpym.do
```

Mobile-friendly action:

```text
https://www.kobus.co.kr/mrs/stplcfmpym.do?keep=/mrs/pay
```

The POST body should include the original seat form fields plus returned hold values such as `pcpyNoAll`, `satsNoAll`, `estmAmt`, `dcAmt`, `tissuAmt`, and `nonMbrsYn=Y` when using a non-member checkout flow.

## Mobile Redirect Caveat

KOBUS common JavaScript can redirect narrow/mobile screens to `/mblIdx.do` unless `location.href` contains a mobile-allowed path fragment such as `/mrs/pay`. In testing, posting to:

```text
/mrs/stplcfmpym.do?keep=/mrs/pay
```

preserved the same checkout POST body while making the final browser URL contain `/mrs/pay`, avoiding the client-side mobile-main redirect condition. Use this variant for Discord/mobile users, but still verify with the user because in-app browsers may add their own quirks.

If mobile still lands on the homepage, recommend opening the helper link in a normal external browser or desktop browser. Do not keep retrying holds indefinitely; cancel stale holds.

## Suggested Output Format

Keep candidate lists concise and actionable:

```text
고속버스 서울경부 → 부산 / 2026-05-09
1. 00:30 심야우등 / 천일고속 / 잔여 10석 / 성인 47,600원
2. 13:50 우등 / ...
```

When a hold/checkout helper is created, state that the next step opens the official KOBUS payment page and that payment remains manual.

For Discord/mobile, provide links as normal text links, not fenced code blocks, so the user can tap them directly.

## Common Pitfalls

1. **Mixing KOBUS and Tmoney codes.** KOBUS terminal codes are not Tmoney 시외버스 terminal codes.
2. **Assuming a cart exists.** KOBUS does not expose a shopping-cart style hold list in the tested web flow. The realistic UX is temporary seat hold plus checkout-entry handoff.
3. **Ignoring hold cleanup.** Always cancel test holds or abandoned holds with `cancPcpy.ajax`.
4. **Treating lookup success as payment permission.** Lookup, seat-stage entry, and temporary hold are not authorization to submit payment.
5. **Mobile homepage redirects.** Use the `?keep=/mrs/pay` helper action for mobile; otherwise KOBUS JS may send the user to `/mblIdx.do`.
6. **Browser automation too early.** Try direct HTTP first. Use browser tooling only to discover changed endpoints or verify final user-facing behavior.

## Verification Checklist

- [ ] Route/terminal codes were resolved from KOBUS, not guessed.
- [ ] Timetable was queried with cookies and a realistic User-Agent.
- [ ] Candidate output includes date, departure/arrival terminals, time, class/operator when available, fare, and remaining seats.
- [ ] Seat-stage response contains expected hidden fields before attempting a hold.
- [ ] Temporary hold success was confirmed with `MSG_CD=S0000` before sending a checkout helper link.
- [ ] Stale/test holds were cancelled with `cancPcpy.ajax`.
- [ ] Payment/card fields were left for the user unless they explicitly confirmed otherwise.
- [ ] Mobile helper links use the `/mrs/pay` marker variant when sent to mobile-heavy platforms.
