# Government Official Messaging (Pattern C)

Class-level pattern for WhatsApp messages to government officials (JDTPs, BBMP engineers, BDA officers, town planning authorities).

> **Related pattern — message FROM an official, not TO one.** When the user wants a message
> written in the voice of a senior official (MP, MLA, senior officer) for an intermediary to
> forward to a department — "this is going from the MP to the DC", "we are writing on behalf of
> the MP" — use `references/third-party-message-via-intermediary.md`. Same deferential register,
> but inverted authorship plus strict fact-hygiene rules (a request for a report is not the
> report, never claim an approval that has not issued).

## When this pattern applies

- The recipient is a government or civic official (JDTP, BBMP, BDA, KSPCB, etc.)
- The user wants to congratulate them on a public achievement before making a request
- The message needs to be **more formal and deferential** than standard WhatsApp style
- The context involves approvals, sanctions, NOCs, site inspections, or other regulatory processes

This intentionally **breaks** the "no pleasantries / direct style" rule that applies to most other NDR WhatsApp messages. Government officials expect respectful, formal language.

## Structure

```
Good morning Sir,

[1. Sincere congratulations — name the specific achievement]

[2. Gentle segue — "Gently following up on..." or "Just checking in on..."]

[3. The request — brief, acknowledge their busy schedule, easy to say yes to]

[4. Tie to outcome — "so we can..." / "this would help us..."]

[5. Open-ended close — leave it to them]
```

## Concrete example: JDTP site visit after bill amendment

```
Good morning Sir,

First of all, congratulations on the Karnataka Apartment Owners (Amendment) Bill — it's fantastic. I think it's a great effort and a step in the right direction. Kudos to you and everyone involved.

Gently following up on the site visit for Ranka Northstar near Allasandra Lake. I know you had mentioned Thursday would be free, but I understand that was also the last day of activities. If it's at all possible for you to spare half an hour today, I'd be very grateful. I know you have a very busy schedule, but this would help us put in the sanction application and move things forward.

Please let me know whatever works best for you.
```

## Key rules

| Element | Do | Don't |
|---------|----|-------|
| Greeting | "Good morning Sir" / "Good afternoon Sir" | First name only |
| Congratulation | Specific and genuine — name the bill/report/achievement | Vague "great work" |
| Tone | Warm, respectful, deferential | Casual, direct, abbreviation-heavy |
| Acknowledge time | "I know you have a very busy schedule" | Assume availability |
| Request | Frame as brief — "half an hour", "a few minutes" | Open-ended time demand |
| Close | "What works best for you" — defer to them | Urgency or pressure |
| Sign-off | No formal sign-off needed (it's WhatsApp), but the whole message stays respectful | |

## Finding government official contacts (researched, not in Google Contacts)

Gov officials (IAS/KAS officers, GBA/BBMP/BDA) are almost never in Google Contacts or the contacts sheets. Don't run the standard lookup chain — research directly (NDR's no-Tavily/no-Apify directive applies; use the `research-web-tools` skill's direct-browser recipe):

- **GBA/BBMP officers:** the official "GBA HOD OFFICERS CONTACT DETAILS" PDF lives on updates.bbmpgov.in — search `site:updates.bbmpgov.in HODs Contact list`, or directly fetch `https://updates.bbmpgov.in/v1/api/file/<fileid>-HODs%20Contact%20list_<date>.pdf` (needs `curl -k` — self-signed cert). Contains IAS officer name, email, designation, mobile for the Chief Commissioner, Special Commissioners, HODs.
- **DDG-via-Jina search proxy** (`https://r.jina.ai/https://html.duckduckgo.com/html/?q=<query>`) surfaces LinkedIn/RocketReach/IAS directory hits; RocketReach masks digits (+91 94481 9XXXX) but confirms mobile prefixes so you can match the official PDF number.
- **Voice-name traps:** dictation garbles non-obvious spellings — "Munish Mudgal/Muthgal/Motgil" = **Munish Moudgil** (IAS, Special Commissioner Revenue & IT, GBA). Confirm against official PDF + LinkedIn before writing.
- Websites have bad certs (gba.karnataka.gov.in ERR_CERT_COMMON_NAME_INVALID, works over curl -k).

## Concrete example 2: khata/software intervention — Munish Moudgil (IAS, Special Commissioner Revenue & IT, GBA)

Scenario: revenue software rejects khata for an old project (Ranka Iris, Domlur — 3B+G+13, 50% TDR setback relaxation under then-new TDR policy, started 2014, stalled, only recently completed + OC). System can't handle the old relaxation norms; backend override needed from his technical team (Santosh & Amar). NDR's message arc — intro → software issue → project brief → specific ask (intervention/override) → gentle press → offer to meet:

```
Sir, this is Nishant Ranka, son of late Mr. Dinesh Ranka. I am not sure if you would
remember, but I met you briefly, many years ago.

I am reaching out to you with respect to a software issue — the revenue software is
not accepting the khata for one of our projects. Just to give you a quick brief on
the issue:

The residential building named Ranka Iris, in Domlur, Indiranagar — three basements,
ground plus 13 upper storeys of construction — had received 50% setback relaxation
based on the then newly introduced TDR policy. We started the work in 2014, and
subsequently, for various reasons, we had to stall the project. We have only recently
completed it and obtained the OC.

Apparently the system is now not accepting the khata because of the new relaxation
policy fed into the system... This is an old file that was approved as per the then
prevailing norms. I believe that a special intervention from your technical team —
comprising Santosh and Amar — would have to approve this in the back end. I also
understand that the caseworker has prepared all of the documentation for your review.

Given that this project has been very, very delayed — we are delayed on RERA as well —
and this khata issue has been stuck for more than a month, with all the customers
having occupied the units, I request your kind attention in this matter.

I would also like to come and meet you briefly, sir — just to say hello, nothing in
particular — as per your convenience, whenever you are at the head office.

Regards,
Nishant Ranka
```

Key moves: intro re-establishes relationship (son of late Mr. Dinesh Ranka — the founder identity), names the project precisely, names the technical team he expects to act (shows he knows the machinery), presses politely with concrete facts (RERA delay, customers occupying units, >1 month stuck), and closes with a low-pressure personal request (say hello, not business). Research → WhatsApp + Gmail draft (HTML/CSS, saved to DRAFTS — never send) delivered in the same turn.

## Common contexts

- **Site inspection** (this session): Official was busy with Karnataka Apartment Owners Bill work, postponed site visit, user congratulates + re-requests
- **Sanction application follow-up**: Official approved a new policy/byelaw, user seeks inspection for sanction
- **NOC/approval**: Official's department completed a reform, user has a pending application
- **Land use change**: Official was involved in a zoning regulation update, user has a related matter
