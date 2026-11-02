# Draft state and dictation artefacts — two checks before you draft or describe a draft

Two small failures that cost real work on 2026-10-24/25. Both are one tool call to avoid.

## 1. "No drafts" means the human already SENT it — not that drafting failed

A draft sits in the human's Drafts folder, which is *their* workspace. NDR opens it, edits it, sends
it, and the draft disappears. So:

- **`gws_gmail_draft_list` returning `count: 0` is not a failure and not an empty mailbox.** Same for
  `gws_gmail_search query="in:drafts"` → 0.
- **Before reporting on a previous draft, confirm state with two calls:**
  1. `gws_gmail_search query='subject:"<subject>" newer_than:14d'` — read the `labelIds` of every hit.
     `SENT` = you sent it (or the human did), `INBOX`/`IMPORTANT` = a reply has already come back.
  2. `gws_gmail_get` the **newest inbound** message in the thread and read it in full.
- **Never say "the draft is ready in your Drafts folder" without having listed drafts in this turn.**
  If the list is empty, the honest statement is "that mail has already gone out — here is the thread
  and here is what came back."
- **The thread may have moved on since the draft went out.** Worked case: the "Call for a meeting"
  confirmation was sent Sat 22:59; Manish replied at 00:01 Sunday adding a completely new agenda item
  (Golfshire leave-and-license expiring 31 Oct, tenant asking to stay to 10 Jan 2027, Induslaw to
  advise). Had the reply not been read, the meeting would have run off a stale 17-item agenda. It is
  now 18 items.
- Corollary for meetings/agendas: **the counterparty's latest mail is an agenda input. Re-read the
  thread tail before any meeting you are about to chair**, and fold their additions into the tracked
  list (sheet and/or body) rather than mentioning them in chat only.

## 2. "Pag" / "Pay" is a dictation artefact, not a matter

NDR's voice notes sometimes end with a stray token transcribed as **"Pag"** (or **"Pay"**) — usually
after a list or a sign-off. It is breathing/pause noise on the transcriber.

- **Never treat it as a person, a project, a matter or an agenda item.** Two occurrences, both
  trailing: "…please draft it and keep it ready. **Pag**." (2 Oct, Bhuvanesh reply) and
  "…Nil Mangla Property Matter, **Pag**." (24 Oct, meeting agenda).
- **Do not chase it.** Searching Gmail for "Pag" returns only ETRealty newsletters mentioning PAG
  Asia — a coincidental match that tempts you to invent an agenda line. There is nothing there.
- **If the deliverable is otherwise complete, drop it silently.** Only if the user asks what it is,
  answer plainly: "it's the tail of your dictation, not a matter — the same token closed your 2 Oct
  note."
- Not the same species as a genuine homonym: a misheard **place** ("Aarti Nagar" → R.T. Nagar) or a
  misheard **person** must be resolved, not dropped. The test is position and grammar — a token at the
  very end of a dictation, after a completed sentence or list, with no sentence around it, is noise.
