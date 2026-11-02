# Internal briefing mail → external recipient's mail (sanitise; never forward verbatim)

## When this fires

A colleague — nearly always NDR — sends a **working mail addressed to the staff member**
("Dear Sarthak, Sudheer Desu has asked for X. Below is X, taken from our records. Y I will
send separately."), and the staff member must get that substance to the **external party**
who asked for it: a lender, banker, vendor, authority, or counsel.

Typical phrasing from the user: *"forward the 29th October mail to him"*, *"draft a mail to
Sudhir using the details Nishant sir has shared"*, *"send him whatever Nishant sent me"*.

## 🔴 Rule 1 — an internal mail is written to a colleague; never forward it verbatim

It carries things that must not reach the recipient. Verified 2026-10-30 (NDR's inventory +
land-bank mail to Sarthak, being sent to Sudheer Desu / VGRO Finserv). The verbatim forward
would have shipped all of this:

| Leak | Actual string in the internal mail |
|---|---|
| Internal salutation | `Dear Sarthak,` |
| Sequencing instruction | `please keep this mail on hold with Sudheer until the full set is ready, so he gets one complete response rather than it going out in pieces` |
| Promise the outgoing mail is not making | `Shareholding, upcoming projects and borrowings I will send separately` |
| Internal section | `4 · Still to come` — an entire section of what is *not* being sent |
| Internal routing | `share with X`, `cc Y`, `check with Eshwari`, `keep on record` |

**Leak checklist — strip every one of these before the draft exists:**

- the colleague's name in the salutation
- "hold / wait / don't send until…" and any other sequencing instruction
- "I will send … separately" / "will follow" / "still to come" / "pending" sections
- internal routing and cc instructions
- notes-to-self, reminders, assessments of the recipient ("he will ask for more")
- the colleague's own commentary about the user's work

Keep the *substance* and its caveats (Rule 6). Strip the *conversation around it*.

## 🔴 Rule 2 — the deliverable is usually a COMPILED mail, not a forward

The recipient asked a numbered list of questions. Answer **in their order**, in the staff
member's own voice, with a short covering note — do not hand them the internal mail's order
and hope they re-map it.

Worked shape (VGRO / Sudheer, 2026-10-30): his ask was *shareholding pattern · unsold
inventory · upcoming projects · land bank · borrowing details*. The compiled mail carried
exactly those five headings in that order; the shareholding came from the deeds, items 2 and
4 from NDR's mail, item 5 was answered "nil", and item 3 pointed at the pipeline already
described in item 4. Nothing in the mail reads as "an internal note someone forwarded".

If a "---------- Forwarded message ---------" block is kept at all, it is *edited content*
(see Rule 4) — and a compiled mail is normally the better answer.

## 🔴 Rule 3 — verification is a leak grep, not a visual read

**Presence checks pass on a leaked body.** My own failure mode: I verified To/Cc/Subject/
threading, attachment names and sizes, and grepped for the *expected* content (`Kothanur`,
`₹320 crore`, `2,500 equity shares`) — all green — and the leaked "keep this mail on hold"
line was sitting in the middle of the body the whole time. It only surfaced because I also
grepped for the phrase.

Before reporting a draft ready, assert **absence**:

```python
LEAKS = ["keep this mail on hold", "Dear Sarthak", "Still to come",
         "I will send these separately", "will follow separately",
         "share with", "for your reference", "as discussed"]
body = <plain-text body read back from the draft>
hits = [s for s in LEAKS if s.lower() in body.lower()]
assert not hits, hits          # any hit = do not ship
```

Then, separately, grep for **presence** of each substance item so a bad slice doesn't drop
half the content. Absence check + presence check, both, every time.

## 🔴 Rule 4 — if you edit a colleague's words, tell the user what you removed

The user is accountable to the colleague for what goes out under the firm's name. Report it
plainly: *"NDR's internal line 'keep this mail on hold with Sudheer' is out of every version —
it would have reached the lender on a straight forward."* One line, in the delivery message,
not buried.

Same for the salutation and for any section you dropped: name them.

## 🔴 Rule 5 — the internal mail may contradict the instruction you are executing

NDR's mail said *hold until the full set is ready*. The user then said *"that mail is
complete, forward it"*. Do not silently pick a side and do not stall:

1. Build exactly what the user asked for.
2. State the contradiction in one line, quoting the colleague's own words, and let the user
   decide — his written instruction is on record and he will remember it.

Never quietly drop a "hold" instruction, and never quietly obey it against the user's
explicit request either.

## 🔴 Rule 6 — carry the internal mail's caveats across

Caveats protect the sender and the firm. Re-organising content into a client-facing mail is
exactly when they get lost:

- `market values as assessed by us and are not a valuation report`
- `no khata or patta number is recorded in the file we hold; RTC and EC are being obtained`
- `title sub judice — Civil Appeal No. … next listed …`
- `the 8 June 2020 understanding operates only on the outcome of the appeal`

Dropping one turns a caveated statement into an assertion the firm now owns. When you
compress an internal mail, re-read the original once against your draft and tick each caveat
off.

## Rule 7 — every attachment the body lists must actually be attached

If the body says `Documents attached: 1 … 2 … 3 …`, all three must be in the MIME and named
**exactly** as listed. The native `gws_gmail_draft_create` / `gws_gmail_reply_draft` tools
carry no attachments — build the MIME by hand (see
`threaded-reply-pdf-attachments.md`) and read the payloads back afterwards.

Also check what you are attaching *contains*: a partnership deed carries the signatories'
Aadhaar numbers. Mask before shipping and say so in the body
(`pdf-redaction` → *Shipping only the necessary pages of a scanned deed*), then offer
unmasked copies on request rather than sending them unasked.

## Rule 8 — "just say nil" answers: scope them to what was asked, then flag the gap

A lender's checklist usually has an item the firm has nothing to say on, and the user will
dictate the answer flatly — *"for the borrowing parts you should mention nothing is there, no
borrowing is there"*. Write it, but:

1. **Scope the sentence to the entities the question was about** — "the three entities have not
   availed any term loan, cash credit, overdraft or other fund-based facility" — not a bare "the
   group has no borrowings". A group-level absolute is a false statement a banker will find.
2. **Flag what sits outside that scope, in one line, to the user** — in this case OD accounts
   elsewhere in the group and an APF under application with ICICI. He decides whether to widen
   the disclosure; you do not widen it silently, and you do not refuse to draft it either.
3. Same shape for "upcoming projects: nothing to report" — say the pipeline is captured in the
   land-bank particulars rather than asserting the firm has no pipeline, and tell the user that
   is the wording you chose so he can replace it with one line.

## Rule 9 — one live draft, always

This class often iterates: draft → "no, forward it" → "no, compile it". Each pivot means
**delete the superseded draft** and say which draft id is live. Two drafts of the same mail
in the folder is how the wrong one gets sent. Keep the user-facing report to the final shape
plus one line on what changed.

## Threading (same conversation, both shapes)

Reply into the recipient's own request thread so their tracker shows one conversation:

```python
orig = g.users().messages().get(userId='me', id=ASK_MSG_ID, format='full').execute()
h = {x['name'].lower(): x['value'] for x in orig['payload']['headers']}
msg['In-Reply-To'] = h['message-id']
msg['References']  = h['message-id']
# and pass threadId into drafts().create
g.users().drafts().create(userId='me',
    body={'message': {'raw': raw, 'threadId': h.get('thread-id-from-message')}}).execute()
```

- `drafts().get()` rejects `metadataHeaders` (`TypeError: Got an unexpected keyword
  argument metadataHeaders`) — use `format='full'` for headers or `format='raw'` for the raw
  message. `drafts().list()` requires `userId='me'`.
- Subject must be exactly `Re: <their subject>` — normalise away any existing `Re:`/`Fwd:`
  tokens rather than stacking them.
