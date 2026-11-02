# Chasing a counterparty who never replied (NDR's "please send a follow-up")

NDR's trigger: *"I had sent an email to X and Y about <subject>, asking for their go-ahead.
They have not responded. Can you please send a follow-up?"* — then a dictated body.

This is a recurring class for NDR (board / co-director / promoter matters), not a one-off.
The follow-up must be threaded, must prove the silence, and must restate the decision from the
record rather than from the dictation.

## 1. Decode the dictated subject first

The subject in NDR's dictation is often a garbled proper noun. It is a **phonetic rendering of a
project name + noun**, not a person or a company.

Observed: *"Hamitaj Kaapak"* = **Prestige Hermitage car parks** (Hamitaj→Hermitage, Kaapak→car park).

Procedure:
- Do NOT search the literal garbled string alone. Search it OR'd with plausible variants **and**
  with a nearby business noun from the same message (`khata`, `car park`, `RERA`, `JDA`, `sale deed`).
- Search `google-draas` first; if zero, try the other vault accounts (`google-ahfl`, `google-gmail`)
  — NDR's outbound on company matters is usually `google-draas`.
- A `from:(A OR B) newer_than:7d` sweep on the counterparties confirms whether the missing reply
  exists in some *other* thread under a different subject.
- State the decoded subject back to NDR in the reply ("X = Prestige Hermitage car parks") so a wrong
  decode is caught immediately.

## 2. Prove the silence before writing

- `gws_gmail_thread_get(thread_id=...)` on the original mail. **`count == 1` and `labelIds: [SENT]`
  is the proof** — a reply would have joined the same thread.
- Then the `newer_than:7d` sender sweep, to rule out a reply filed under a different subject.
- Say which of the two checks you ran in your report. Never write "they haven't replied" without it.

## 3. Restate from the meeting note, not from the dictation

When NDR says *"we discussed this at the board meeting two or three days ago — please find that"*,
he means: locate the circulated note of that meeting and quote the decision from it.
- Find the note by searching the counterparties + `meeting` and by date (notes usually go out 1–2
  days after the meeting, in the original "Call for a meeting" thread).
- Pull the specific decision line (counts, areas, who moves it, what is gated) into the follow-up.
- Cross-check the note against the original email's numbers. They must agree (e.g. 4 existing parks +
  2 lower-basement parks attached to 2401 = the 6 per unit in the khata email).
- Do NOT invent or "improve" figures. Every number in the draft must be traceable to one of the two
  documents you just read.

## 4. Body shape that works for NDR

Short, flat, no bullets-in-bullets, no apology:

1. One line: what is being followed up, dated, and *"I have not heard from either of you."*
2. Point them to the meeting note by name/thread/date — *"please find the note of that meeting"*.
3. The agreed position, restated compactly (the actual numbers, unit by unit).
4. What it is blocking, concretely and in NDR's words (*"holding up the khata process — the khata
   person is waiting on me and I cannot let him file"*).
5. The smallest possible ask: *"a one-line go ahead from each of you is all that is needed."*
6. Any person-specific dependency (e.g. *"Dharmesh — you will need to be available for the OTP"*).
7. A same-reply ask for anything else that has gone unanswered and has a deadline attached
   (e.g. the site allotment slot two days away) — one line, not a second email.

Sign off `Regards, / Ndr` + the one-line signature block. Plain hyphens, not em dashes, unless the
source note uses them.

## 5. Threading — the empty-References case

A follow-up thread where the only message is NDR's own has **no `References` header at all**.
The native `gws_gmail_reply_draft` will mis-address `To` to NDR himself (see
`threaded-reply-recipient-pitfall.md`), so build the raw MIME:

```python
h = metadata headers of the LAST message (Message-ID, References, Subject)
m['In-Reply-To'] = mid
m['References']  = (refs + ' ' + mid).strip() if refs else mid   # refs is '' here
drafts().create(userId='me', body={'message': {'raw': raw, 'threadId': THREAD}})
```

Explicit `To` with both counterparties, `From: Nishant Ranka <ndr@draas.com>`, `Re: <original subject>`.
Verified 2026-10-29: draft landed in the original thread with `labelIds: ['DRAFT']` and correct
`To`/`Cc`/`In-Reply-To`/`References`.

## 6. Verify, then report draft-only

- Read the headers back **and** decode the body (`drafts().get(format='full')`, walk `payload.parts`)
  — the numbers are the whole point of the mail, so confirm they survived encoding.
- Report: original mail (date/time/recipients), the two silence checks, the draft id, and the fact
  that **nothing has been sent** — NDR sends from Drafts himself.

## Pitfalls

- Do not reply-all to the board thread that carries the *meeting note*; the follow-up belongs in the
  thread of the **go-ahead request** it is chasing.
- Do not restate only the ask; restate the decision and its numbers, because the whole complaint is
  that they never acknowledged the decision.
- Do not add a new attachment or re-attach the meeting note images unless asked — point to the thread.
- `google-ahfl` can be expired (`invalid_grant`) while `google-draas` is live; a search failure there
  is a token-state issue, not evidence the mail is missing. Search `google-draas` first.
