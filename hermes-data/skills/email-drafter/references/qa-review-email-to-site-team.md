# The QA-verdict email to the site team (internal, issue-by-issue)

Verified 2026-10-24 (Ranka Oasis underground services → Anbu, Cc Kanthesh). Fires immediately after
`construction-qa-photo-review` produces the plain-text inspector verdict. NDR's ask, near-verbatim:

> "Send a detailed email to <site lead>, marking <design/PMC person> in CC, attaching the N
> images I gave. Give the issues, the rating, the recommendations. Tell them I want them to study
> this and have a chat issue by issue."

That is a fixed deliverable shape — do not re-derive it. Draft only, never send.

## Recipients — resolve from Gmail headers, never the contacts sheet

| Person | Address to use | Note |
|---|---|---|
| Anbarasan M (Anbu) | `anbarasan@draas.com` | He SENDS as this. `pm2.blr@draas.com` is an older mailbox he also sent from — the contacts sheet row points there and a draft addressed to it is a mis-send. |
| Kantesha B G (Kanthesh) | `kanteshbgme@gmail.com` | Appears on Oasis threads as `Kantesha B G <kanteshbgme@gmail.com>`; WhatsApp display name "Kantesh Dra". **No `@draas.com` identity exists** — `gws_dwd_resolve(target="kanthesh")` returns "No identity matches". |

Confirm with `gws_gmail_search(query="Kanthesh")` before building — the address is visible in the
`from`/`to` headers of the design threads. When a person has only a personal gmail, **say so in the
delivery message to NDR** ("Kanthesh is Cc'd on his gmail — if there is a @draas.com mailbox I don't
have it") rather than silently substituting another address.

## Body structure (7 sections, same order every time)

1. **Header band** — navy `#1F3864`, amber `#f5a623` rule under it. Project name, "Underground
   Services & Chambers — Site Quality Observations", then the photo-record line: date span, roads/
   areas, number of photographs examined.
2. **Opening** — address both men by name. State what the photo record covers and that the read is a
   *quality* check, not a progress check. Then the three framing beats, in this order:
   - **Credit first and mean it** — "the intent, the sequence and the choice of materials are better
     than most layouts I have seen, and I say so in the next section. This is not a criticism of
     effort."
   - **The stakes line** — "whatever is wrong now gets paid for twice: once now, and once when the
     road is cut open in year three." This is what makes a contractor read past the first paragraph.
   - **The caveat, labelled as such** — "I am judging from photographs only … I am measuring
     execution against good practice rather than against our own design." Name the three documents
     that would close the gap.
3. **What the photographs show is happening** — proves the batch was actually read.
4. **What is being done right** — numbered, 5–6 items. A review with no credit gets dismissed.
5. **Where it falls short** — numbered issues. Each issue is a tinted block (amber left rule,
   `#f7f8fa` fill) with three lines: **title**, a `Ref:` line naming the photo numbers, then
   **"What the photographs show:"** and **"Why it matters:"**. The "why" must be the physical
   failure mode in 2–3 years, not the word "non-compliant".
   Then, as a **separate red `#fbe6e4` box outside the numbering**, safety — shoring, spoil setback,
   PPE, toolbox talk — closed with *"I would like this confirmed in writing as done, not as
   planned."* Keep it out of the numbered list so the number of quality issues stays clean.
6. **Rating** — one table, colour-coded cells: green `#e6f2e6` ≥7, amber `#fdf1dc` 5–6, red
   `#fbe6e4` <5. Eight dimensions from the rubric in `construction-qa-photo-review`, then a tinted
   overall box: "Overall: 5.5 to 6 out of 10 — Grade B minus", with the one-line reason (usually:
   *not* the material or the intent, it's the four things that get buried).
7. **Recommendations** — three-column table: `#` / CHANGE / WHY. Ordered by value for money, and the
   single highest-return item is called out as such in its WHY cell.
8. **The photographic record** — every photo embedded inline with a caption written from what is
   *actually in the frame*, never from the WhatsApp caption. Then:
9. **What I want from both of you** — the ask block:
   - Study it **issue by issue**; "I do not want a summary reply." For each point, agree or disagree;
     where they disagree, the drawing / specification / IS code that supports their position; and
     "if your position is that this is normal practice in this market, say that too, plainly."
   - Written reply within **three working days**, then a meeting point by point.
   - Bring three named documents to that meeting.
   - A one-page revised method statement, with the material source and the compaction equipment named.
   - Adopt the 8-item daily QA record **from Monday**, replacing the manpower-plus-caption format.
   - One named design-side ask to the Cc'd person (e.g. confirm invert levels against the layout,
     issue the separation detail).

## Lines that earn their place (reuse, don't paraphrase away)

- "If it is not recorded, it did not happen."
- "No evidence is the same as no grade." (against missing invert/level records)
- "A draw pit that holds water through a monsoon is an earth-fault waiting to happen."
- "The reporting format is itself a finding." (manpower counts + captions = progress log, not QA log)

## Opening the issues honestly

Do not assert a spec violation you cannot point at in pixels. Write "appears placed without edge
formwork", "no formwork visible in the photograph". Where one earlier photo showed something done
*right* (e.g. a granular bed present under one reach), say so inside the issue — that is what makes
the rest credible.

## Delivery message to NDR

Report: draft is in Drafts (not sent), To/Cc and why those addresses, the subject, section count and
issue count, the rating, the photo selection rationale, total mail size, and any address caveat. Attach
the rendered screenshot of the mail as `MEDIA:` so he can see the formatting without opening Gmail.
