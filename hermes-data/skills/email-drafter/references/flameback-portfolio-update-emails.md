# Flameback Capital — AI-generated portfolio update emails (recurring)

Recurring daily sender from **Kishan Murjani Nair** (Flameback Capital). First handled 2026-09-15.

## The email pattern
- From: **Kishan Murjani Nair <kishan@flamebackcapital.com>**, subject `Fwd: URGENT Portfolio Update - <Month DD, YYYY> | PV: $XXX,XXX | Return: +XX.XX% (<headline defect>)` — a FORWARD of the system-generated daily report.
- Original report From: **Kishan Nair <kishan.99@gmail.com>** (Kishan's personal Gmail) → To `kishan@flamebackcapital.com`.
- Kishan's top-of-body note `All ai` (bare line) = the report is AI/algo-generated.
- Body = daily portfolio summary: portfolio value (cost basis), realized P&L, win rate, overnight closes, new opens, "SCAN CATASTROPHE" / universe-filter violations, overdue positions, recommended immediate actions. Ends with the live tracker sheet:
  `https://docs.google.com/spreadsheets/d/1ky8gxrccCaQmtfEQCe3U98ASz6X1ZvQw-5WPyKkg7EU` ("US Stock Daily Movers Tracker | Paper Portfolio ($100K)")
- To header on the forwarded copy: `Nishant Ranka <ndr@drahomes.in>` (old forward address; mailbox is ndr@draas.com).

## Locating the thread
Search by sender domain, NOT subject keywords (subject headline changes daily):
```python
q = "(from:flamebackcapital.com OR from:kishan@flamebackcapital.com) after:YYYY/MM/DD"
```
Arrives as a single-message thread — threadId == message id. Reply via `draft_reply_create` with that message_id to stay threaded.

## NDR's reply framing for these updates (dictated 2026-09-15)
When NDR asks to reply to the AI-generated update, the draft should convey:
1. Acknowledge depth ("extensive, serious engineering") but state plainly he can't grasp it as-is.
2. Ask Kishan to walk him through **line by line, step by step** until the recommendation is actually understood.
3. Three explicit asks:
   - What is the recommendation in plain terms, and who is the target audience (is it meant to go to a customer?)
   - **Most important: what is genuinely AI vs algorithmic / deterministic rules & workflows**
   - Lingo must become plain-English, layman-friendly for customers.
4. Marketing framing NDR believes in (use when he wants to amplify): "constantly monitoring the portfolio from multiple perspectives and angles, which is not humanly possible" — whether positioned as AI or as an algo, this plus the performance story (vs what traditional portfolio management can't do) is a marketable hook; "everybody needs exposure to this."

## Reply mechanics (verified 2026-09-15)
- **DRAFT-ONLY** (Hermes never sends — user reviews in Drafts).
- Account `google-draas` (ndr@draas.com — NOT ahfl, per NDR's Flameback alerts rule).
- To = `Kishan Murjani Nair <kishan@flamebackcapital.com>` (From of the arrive-copy; no Cc).
- `call('draft_reply_create', service_name='google-draas', message_id=<id>, to='Kishan Murjani Nair <kishan@flamebackcapital.com>', body=..., html=False)`
- Verify: `drafts().get(id=..., format='full')` → To, threadId, In-Reply-To must match the source thread. Keep the reply tight (~1 screen); offer to expand on a specific section (e.g. the SCAN CATASTROPHE batch) if NDR wants.