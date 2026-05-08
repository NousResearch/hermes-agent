---
name: intercity-bus-booking
description: Use when searching or assisting Korean 시외버스/Tmoney bookings, including 시외버스 예매, 시외버스 시간표, terminal/route lookup, remaining seats, fares, temporary seat holds, and assisted card-information entry links. Prefers official HTTP/API-first flows over browser automation.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [korea, bus, travel, booking, tmoney, intercity, transportation]
    related_skills: [express-bus-booking]
---

# Intercity Bus Booking (Tmoney 시외버스)

## Overview

Use this skill for Korean 시외버스 timetable lookup and reservation assistance via the official Tmoney intercity bus site. Prefer HTTP/API-first flows for route lookup, timetable search, fare/seat-stage entry, and checkout-entry handoff.

Keep 시외버스 separate from 고속버스. Tmoney terminal codes and POST flows are different from KOBUS 고속버스. Use `express-bus-booking` for KOBUS routes.

## When to Use

- The user asks for `시외버스 예매`, `시외버스 시간표`, `시외버스 예약`, or a known 시외버스 route.
- The user names 티머니/Tmoney/시외버스 통합예매.
- The task is to present timetable candidates, check remaining seats/fares, or assist up to the official card-information page.

Do **not** use this for:

- 고속버스/KOBUS routes — use `express-bus-booking`.
- Final card submission or payment unless the user explicitly confirms the exact action.
- Browser automation as the first option when HTTP endpoints are available.

## Core Principles

- Use official Tmoney 시외버스 surfaces: `https://intercitybus.tmoney.co.kr`.
- Use cookies, referers, and a normal browser User-Agent.
- Default scope is lookup and assisted manual checkout: timetable, fare/seat stage, selected-seat temporary hold, and official card-information page handoff.
- Do not submit card fields, final payment, or irreversible reservation steps without explicit confirmation.
- If a test or user-abandoned hold is created, use the official cancellation/back flow when available.

## Known HTTP Flow

See `references/tmoney-intercity-http-flow.md` for session-proven endpoint details and parameter examples.

### 1. Start a Session

Fetch a landing or booking-entry page with cookies enabled:

```text
GET https://intercitybus.tmoney.co.kr/
GET https://intercitybus.tmoney.co.kr/otck/trmlInfEnty.do
```

### 2. Query Timetable

```text
POST /otck/readAlcnList.do
```

Typical fields include:

```text
depr_Trml_Cd=0511601
arvl_Trml_Cd=2482701
depr_Trml_Nm=동서울
arvl_Trml_Nm=속초
ig=1
im=0
ic=0
iv=0
depr_Dt=YYYYMMDD
depr_Time=000000
```

Parse schedule buttons/rows. The next-stage parameters are often embedded in `readSasFeeInf(...)` onclick arguments.

### 3. Enter Fare / Seat-Count Stage

```text
POST /otck/readSatsFee.do
```

Send selected values from `readSasFeeInf(...)` plus passenger counts and original search fields. The response should include `form#readPcpySats` and hidden values such as:

```text
rot_Id
alcn_Sqno
depr_Trml_Cd
arvl_Trml_Cd
depr_Time
igFee
imFee
icFee
total
```

### 4. Temporary Hold and Card-Information Entry

```text
POST /otck/readPcpySats.do
```

Submit the `readPcpySats` hidden fields plus selected seat/count fields, commonly including:

```text
pcpy_Num
sats_No
bus_Tck_Knd_Cd
cty_Bus_Dc_Knd_Cd
dcrt_Dvs_Cd
rtrp_Depr_Dt
```

A successful response lands on the official `카드정보 입력` page and includes a temporary seat hold identifier such as `sats_Pcpy_Id`.

## Checkout-Entry Link Helper

A helper-served HTML page can auto-submit a POST form directly to:

```text
https://intercitybus.tmoney.co.kr/otck/readPcpySats.do
```

This creates the hold in the user's browser/session and opens the official Tmoney card-information page. The user should manually finish payment on the official page.

For Discord/mobile, provide helper links as normal text links rather than fenced code blocks so the user can tap them.

## Mobile and In-App Browser Notes

Tmoney 시외버스 is more mobile-tolerant than KOBUS in the tested flow: an iPhone Safari-style mobile User-Agent returned the official `카드정보 입력` page with `sats_Pcpy_Id` present.

However, in-app browser User-Agents can behave differently or fail if the same seat/hold payload is replayed. If a Discord/Android in-app test returns a generic error such as `발행을 실패하였습니다`, do not assume route lookup is broken. Recreate a fresh hold payload, retry once in an external browser, and cancel stale holds.

## Suggested Output Format

Keep candidate lists concise:

```text
시외버스 동서울 → 속초 / 2026-05-09
1. 06:05 우등 / 금강고속 / 잔여 8석 / 성인 21,300원
2. ...
```

When a checkout-entry helper is created, say that it opens the official Tmoney card-information page and that payment remains manual.

## Common Pitfalls

1. **Mixing terminal code systems.** Tmoney 시외버스 codes are not KOBUS codes.
2. **Assuming checkout-entry equals final payment.** `readPcpySats.do` can open the card-information page, but final payment remains a separate manual step.
3. **Replaying stale hold payloads.** A repeated POST for the same route/seat can fail or create confusing results. Generate a fresh seat-stage payload for real use.
4. **Skipping cancellation/back flow.** Use the official cancellation/back form (`pcpyCanc=C` via `readSatsFee.do` when available) for abandoned holds.
5. **Overusing browser automation.** Use browser only for endpoint discovery or visual verification after HTTP probing.

## Verification Checklist

- [ ] Route/terminal codes were resolved from Tmoney 시외버스, not guessed or copied from KOBUS.
- [ ] Timetable response was parsed for schedule rows/buttons and next-stage parameters.
- [ ] Fare/seat-stage response contains `form#readPcpySats` and expected hidden fields.
- [ ] Checkout-entry response contains `카드정보 입력` and a hold identifier such as `sats_Pcpy_Id` before reporting success.
- [ ] Stale/test holds were released through the official cancellation/back flow where possible.
- [ ] Payment/card submission was left to the user unless explicitly confirmed.
- [ ] Mobile links are provided as tappable text links on Discord/mobile platforms.
