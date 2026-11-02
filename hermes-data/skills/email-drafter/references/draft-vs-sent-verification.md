# "Send" — resolving draft vs already-sent, and recipient completeness

How to handle NDR's one-word instruction "send" without either sending mail (never
allowed) or reporting something that is not true.

---

## 1. Before you touch anything, establish the actual state

`drafts().list()` first. If the draft NDR is referring to is not there, **do not
assume it was lost**. Search SENT for the subject.

Real failure this came from: an update email that was described as "sitting unsent in
your Drafts" had in fact already gone out — it was in SENT with label `SENT`, dated the
previous day. Drafts were empty in all three accounts. Reporting "draft is ready"
would have been wrong twice over: there was no draft, and the mail had already gone.

Checklist:
1. `gws_gmail_draft_list` on the originating account (and the other accounts if the
   user has more than one — ndr@draas.com, ndr@ahfl.in, nishantranka@gmail.com).
2. If empty, `gws_gmail_search` with the subject / key phrases, `in:sent` implied.
3. If a sent copy exists, read it in full with `gws_gmail_get` and compare its
   To / Cc against the intended recipient list.

## 2. Search snippets hide the Cc list — always read the full message

The search result renders only `"to"`. A mail that "went to the board" can have gone
to two people with an empty Cc. Only `gws_gmail_get` / `gws_gmail_thread_get` show the
real distribution.

Real failure this came from: an update went to Manish and Piyush only. Bhavik,
Dharmesh and Sanjeev Ranka — all three intended directors — never received it, because
there was no Cc at all. The gap was invisible until the full message was read.

So: build the intended list, then diff it against To + Cc of what actually went, and
report the difference explicitly.

## 3. What to do with "send"

Never call a send function — drafting is the only sanctioned path (see the hardening
rule in the system prompt and this skill's description). So:

- **Not yet sent** → create the draft, verify it by listing drafts, then tell NDR it is
  in his Drafts folder and he sends it. Do not say "sent".
- **Already sent, but missing recipients** → say so plainly ("that was never a pending
  draft; it went out on <date> to X and Y only"), then create a draft covering the
  people who missed it, noting in the opening line that the earlier recipients already
  have it. Do not re-send to everyone and duplicate.
- **Already sent to everyone** → say there is nothing to send, and offer a follow-up
  instead.

## 4. Duplicate-send guard

The identity registry matches on full names, not first names. "Rahul" may not resolve
where "Vinod Kumar Das" does. Trying a second variant after the first appears to fail
can deliver the same message twice to the same chat/person.

Rule: resolve the full name first (`contact_resolver` / `send_message action=list`),
send once, then verify delivery before trying any alternative spelling. If a duplicate
does go out, say so rather than letting NDR discover it.

## 5. Draft shape for a board / directors update

NDR's explicit ask when a distribution was missed: keep the original body, add one
line at the top saying who received it earlier and why the rest are getting it now,
and correct any relative date ("yesterday" → the actual date) so the text is true on
the day it is read.

Subject stays identical to the original so it groups in the recipient's mailbox.
