# Recalling a deliverable by description (briefing / deck / note) — Gmail + DRA Content + WhatsApp

Complements `references/email-forensic-thread-lookup.md` (which covers To/Cc forensics and
building the thread deep-link). This file covers the *"there is an email I shared with X about
Y — give me the subject line / link"* class of request, where the target is usually a document
NDR himself wrote and published, not a received mail.

## Trigger

- "There is an email I shared with <person> and possibly <person> regarding <topic> …"
- "Give me the subject line or link to that email first. I want to review it."
- "…and then share it with <external party>, who has followed up asking for the brief."

## Step 1 — search by the CONTENT of the document, not by recipient

NdR authored these, so the hit comes back labelled `SENT` from his own account. Search the
default mailbox (`google-draas`) with two passes:

1. `<project/topic> briefing` — e.g. `Riverstone architecture briefing`
2. one bare, high-recall word — e.g. `riverstone` — to enumerate the neighbouring documents

Do **not** lead with `to:<recipient>`: "shared with Roshni and possibly Aamir" is a
half-memory, and the actual send is often one of many forwards in the same thread.
Recipient searches are for step 4 (confirming who was on it), not for finding it.

## Step 2 — expect a published briefing, not an attachment

Modern DRA briefings are DRA Content artifacts. The body carries
`https://content.ahfl.in/a/DRA-ART-YYYY-NNNNNN` plus "17 pages", "process note linked from
inside", "image study" etc. When that is the case, the artifact URL **is** what he wants to
review — deliver it as the primary link, with the Gmail thread as the secondary link to the
covering note:

```
https://mail.google.com/mail/u/0/#all/<threadId>
```

(use `#all/`, not `#sent/` — account index varies across his mailboxes).

## Step 3 — the near-duplicate document trap (this is the real risk)

The same project accumulates similarly-named briefs across months. Picking the wrong one
silently is worse than asking. Verified pair:

- **Terra Green - Estate Home Architectural Briefing (Riverstone Farms & Portfolio)** —
  8 Oct 2026, to rnr@draas.com + Aamir, artifact `DRA-ART-2026-000058`. Contains: the estate-home
  concept, the portfolio, the design language and elements/compositions/forms, the AI-led design
  *process note*, the image study (103 refs + 200+ variations), scope of work, and a
  ready-to-forward opening note.
- **River Stone Farms — Villa Design Briefing** — 4 May 2026, to Arvind. A different, older
  document: the 4,000 sft G+1 villa typology, material palette, Abelion reference, 2–3 concept
  options, fee structure.

Disambiguation rule: mention of *process / how the design process runs / elements / compositions /
forms / portfolio* → the Terra Green briefing. Mention of *villa typology / material palette /
built-up area* → the older villa brief.

Always surface what you chose and name the alternative you rejected in one line. Do not merge
them, and do not present only the chosen one as if it were uncontested.

## Step 4 — when the person "emailed" but no mail exists, check WhatsApp before saying no

NdR's premise is often slightly wrong about the channel. Before reporting a negative on a named
person's follow-up:

1. `gws_gmail_search(query="from:<their-address> newer_than:14d")` on the default account,
   and again on the other vault accounts if empty.
2. `wa_list_chats(query="<First name>")` → `wa_read_messages(chatId, limit=15)`.

Real case (Oct 2026): "there is an email in which Arvind has followed up on the briefing note" —
no Arvind mail in 20 days. His nudge was on WhatsApp, 13 Oct: *"Regarding riverstone, waiting for
the brief"*, answered by NdR *"Sending today w/o fail"*. The correct report is: not email,
WhatsApp, with the quote, the date, and his own reply — plus the flat statement that nothing came
on email. Never let the correction imply the email exists.

## Step 5 — mailboxes that are not in the vault

Vault services today: `google-draas` (ndr@draas.com), `google-ahfl` (ndr@ahfl.in),
`google-gmail` (nishantranka@gmail.com), `google-o3infotech`. There is **no `@drahomes.in`
service**, yet a large volume of external follow-up lands there. Report such a gap as
"not in the mailboxes I can reach" — never "doesn't exist". A single account returning
`invalid_grant` means that one account needs re-auth; the hunt is not failed.

## Step 6 — delivery shape NdR actually wants

- `**Subject:**` · `**Sent:**` date+time · `**To:**` real recipients, then the artifact link,
  then the thread link — each URL on its own line, **plain, never in a code block**.
- Two short paragraphs of context: what is behind the link; then numbered "notes" for the
  ambiguity and the channel correction.
- Close by offering the next step as a draft only (adapt the brief / draft the email to the
  architect with the link and the opening note) — never send.

## Follow-on: adapting the brief for an external architect

The Terra Green email already contains a "OPENING NOTE — FORWARD THIS TEXT TO ANY ARCHITECT"
block written to be forwarded unchanged. When NdR asks to share the brief with an architect,
reuse that text verbatim as the draft body (plus whatever adaptations he asks for), and point
out that it exists so he can compare before editing.
