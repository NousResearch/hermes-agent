# Channel Choice: WhatsApp vs Email for NDR's Follow-ups

**Lesson (corrected 2026-09-05, Pankaj Kumar / Blue Hat Solutions apology):** NDR dictated a
personal apology follow-up ("write back to him with sincere apologies…") with NO channel word.
Hermes drafted a Gmail draft; NDR corrected: **"Not email I need a WhatsApp message."**

## Rule

When NDR dictates a **personal apology / catch-up / re-engage message to an external contact**
and does not name a channel, default to the **WhatsApp deep link** (`messaging-drafts` → `references/messaging-links-workflow.md` —
`whatsapp_link` tool), NOT an email draft.

Email remains the right default only for **formal content**: documents, terms, governance,
project briefs to professionals/consultants.

## What to do

1. Resolve the contact (People API + contacts sheet + Gmail) — phone needed for WhatsApp.
2. If the dictation is personal/re-engage → generate the WhatsApp link in the same turn.
3. If you already created an email draft before the channel correction, offer to delete it via
   `gmail.users().drafts().delete()` so Drafts stays clean, and deliver the WhatsApp link instead.
4. Optionally deliver BOTH (draft + link) when content could fit either channel — let NDR pick.

## "When did I last meet X" — calendar anchor

For apology/follow-up dictations referencing a past meeting, look up the last meeting before
drafting:

- `calendar.events().list(calendarId='primary', q='<name>', timeMin=…, timeMax=…, singleEvents=True)`
  across the relevant months (use `q` for the person's name; search a wide window, e.g. 1–2 months
  before "today").
- Ground the apology in the real date ("since our meeting in June") and venue — it makes the
  follow-up concrete and correct.
- Cross-check session history (`session_search`) for the meeting creation record if the calendar
  event matches a known project/person (e.g. "Discussion about Partnership Opportunity - Pankaj
  Kumar / Blue Hat Solutions").