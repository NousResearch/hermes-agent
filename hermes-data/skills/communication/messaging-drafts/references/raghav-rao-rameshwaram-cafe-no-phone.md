# Raghav Rao — IDC / Rameshwaram Cafe
**Date:** 2026-06-04 (phone discovery) / **2026-10-08 (emails + content share verified)**
**Contact found:** Row 2633 via `contact_resolver` (row 2632 in an earlier sheet read — rows shift as contacts are added)
**Outcome:** Phone numbers found in far-right columns — initial A:R query returned empty phones. Emails ARE resolvable via `contact_resolver` — the file name "no-phone" is stale; the phone data was found the same day.

## What Happened

1. Searched sheet with range `A:R` (columns 1–18) — found Raghav Rao (Row 2632) immediately
2. Reported "no phone number stored" — user corrected: "I can see phone number in both AC & AE columns"
3. Re-queried with range `A:AE` (columns 1–31) — found **3 phone numbers** in cols AD (29), AE (30), AF (31)

## Sheet Column Layout (31 columns, A:AE)

| Column | Index | Content |
|--------|-------|---------|
| A | 1 | First Name |
| C | 3 | Last Name |
| K | 11 | Organization |
| P | 16 | Photo URL (Google Contacts link) |
| Q | 17 | Labels |
| R | 18 | Location |
| **AC** | **28** | **Phone 1 - Label** (e.g. "Mobile") |
| **AD** | **29** | **Phone 1 - Value** (e.g. `+91 70226 49471`) |
| **AE** | **30** | **Phone 2 - Label** (e.g. "Mumbai") |
| **AF** | **31** | **Phone 2 - Value** (e.g. `+91 97691 04054`) |

## Confirmed Numbers (2026-06-04)

| Type | Number |
|------|--------|
| Mobile | +91 70226 49471 |
| Mobile | +91 63636 88698 |
| Mumbai | +91 97691 04054 |

## Emails (2026-10-08, via `contact_resolver` row 2633)

| Label | Email |
|-------|-------|
| Primary (Home) | raghavidc8@gmail.com |
| Home | raghavrao1983in@gmail.com |
| Work | ceo@therameshwaramcafe.org |

**Share convention (verified 2026-10-08):** when granting NDR's content to Raghav, share
BOTH gmail addresses used (contacts resolver lists three). In practice, sharing the primary
`raghavidc8@gmail.com` AND the work `ceo@therameshwaramcafe.org` both resolve as ACTIVE
VIEWER grants; NDR asked to "give him viewer access" as one person, so two grants for the
same person is the expected outcome — report both with the same expiry. WhatsApp deep link
goes to **+91 70226 49471** (Mobile #1 — the number NDR actually sent to).

## Key Lesson

**"Contact found in A:R" ≠ "phone columns are empty."** The range `A:R` covers cols 1–18 and stops before phone columns (AC=28, AD=29, AE=30, AF=31).

**Rule:** When the task is to find a contact's phone number, ALWAYS request range `A:AE` (or `A:AF`). The partial range `A:R` is safe for name/org lookups but will silently miss all phone data. This caused a false negative and an unnecessary user correction.

## Drafted Message (approved)

```
Hi Raghav,

Heartiest congratulations to you and Divya on the birth of your daughter! 🎉

Wishing you both all the very best as you settle into this beautiful new chapter. Hope mother and baby are doing wonderfully.

Would love to catch up when you're next in Bangalore. Give me a call whenever you're free.

Best regards,
Nishant
```

Sent to: +91 70226 49471 (Mobile #1)