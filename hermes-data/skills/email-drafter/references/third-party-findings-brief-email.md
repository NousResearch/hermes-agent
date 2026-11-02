# Findings brief to a THIRD PARTY — "send all of this in an email to <person>, explain what they need to do"

Ask shape (verbatim, 2026-10-26): *"Okay send all of this in an email to Mamta with a clear
explanation of what all she needs to do and all of the findings in the email and specific steps of
what she needs to do logging into the portal or whatever and finding out if the dues were paid or
not paid earlier."*

This is NOT a normal reply-to-counterparty mail. The recipient is a family member / shareholder /
client who has **no access to any of our systems**, no idea what we found, and is being asked to
perform a handful of physical/online actions. The deliverable is a self-contained briefing letter:
every finding, in plain language, plus numbered instructions they can follow alone at 8 pm with a
phone in one hand.

## 0. "all of this" — recover the findings FIRST

The dictation always refers back to a prior session ("all of this", "all of that", "the report").
Do not re-research from scratch and do not ask what "this" means.

1. `session_search(query="<person> <topic>", sort="newest")` — one hit gives you the closing
   assistant summary, which is usually a faithful digest of everything found.
2. If the discovery result comes back truncated (it does — 380 KB+ on a thread that ran all day),
   pull the tail directly from `/data/hermes/state.db`:
   `SELECT id, role, content FROM messages WHERE id > <anchor> ORDER BY id` and read the final
   assistant message plus the assistant turns that built the deliverable.
3. The prior session may also have produced an HTML report on disk (`/tmp/<name>/index.html`).
   Strip its tags to plain text and reuse its wording and its tables — it holds the precise
   figures (policy numbers, premiums, dates, DOB, branch) that must not be retyped from memory.
4. Carry the prior report's **caveats forward into the email**. Flags like "this number was read
   off a low-res photograph", "the register refreshes half-yearly", "branch code may have been
   re-designated" were true in the report and are still true in the email. Never quietly drop them
   to make the letter read cleaner — put them in a small-print "basis of this note" footer and,
   where the recipient can actually check, in the body.

## 1. Resolve the recipient's ACTIVE address (do not guess, do not use the sheet alone)

People like this have 3–4 addresses and the obvious one is often dead. Verified case 2026-10-26,
Mamta Rathod: `mamta.r@drahomes.in` (last inbound 2017), `mamta.r@draas.com` (0 inbound),
`mamatadr@gmail.com` (2005–2017), `maadi32@gmail.com` (**active — she was replying from it in
Aug 2025**).

Method — search each candidate as a *sender* across every mailbox and take the most recent inbound:

```python
for acct in ['google-draas', 'google-ahfl', 'google-gmail']:
    svc = build_service('gmail', 'v1', service_name=acct)
    for q in ['from:<addr1>', 'from:<addr2>', ...]:
        r = svc.users().messages().list(userId='me', q=q, maxResults=3).execute()
        # metadataHeaders From, Date, Subject on each hit
```

The address with the newest *inbound* message wins. Then:

- **To:** the active address (usually personal for a personal matter).
- **Cc:** the official/company address, so the record lands in the company mailbox too.
- Do NOT cc NDR — the draft already sits in his own Drafts folder.

## 2. Write it as a briefing letter, not a report

Fixed structure that worked (keep the order — it answers "why am I reading this", "what is the
answer", "what do I do" in that sequence):

1. One short opening paragraph: what documents this is about, when they sent them, and **how many
   things you need from them** ("three things, listed at the end"). Numbering the ask up front is
   what makes the letter feel finite instead of alarming.
2. **"The short answer"** — 3–5 bullets. Each bullet answers one question the recipient actually
   has, in their words: *is there anything to pay? is money owed to me? has it already been
   settled? what do I have to do?*
3. **The facts table** — the documents/transactions/policies in a plain HTML table with totals.
   Follow it with the one explanatory paragraph that stops them worrying it was mis-sold
   ("the ₹3,00,000 is death cover, not the payout — nothing is missing").
4. **What I checked and exactly what the authority said** — the queries run, the literal response
   string, and the control test that proves the answer is real rather than a broken page. Then two
   explicit paragraphs: **what it proves** and **what it does not prove**. That pairing is the
   single most important part of the letter — it is what stops a non-expert from over-reading a
   "not found".
5. **What you need to do** — numbered steps, each a sub-heading. Rules:
   - The exact URL as a link, plus a fallback navigation path ("licindia.in → Customer Services →
     Unclaimed Amounts") in case the deep link dies.
   - Say out loud what is **not** needed: "no login, no OTP, no registration".
   - Give the literal values to type ("policy number 364228553 and date of birth 20/10/1978").
   - Say what a good result looks like, quoted ("Records with respect to entered data not found").
   - Give the **stop-and-call-me branch**: "if any one comes back showing an amount, stop and call
     me — do not file any claim until we have spoken." Always include one.
   - Where multiple routes exist, list them cheapest-first (net banking → app → branch counter),
     with the cost of the expensive one (₹100–500).
   - Defer heavy or formal work: "in that event I will have the forms prepared and sit with you —
     you will not have to work it out at the counter."
6. **Cautions** — 2–4 items: name-spelling variance between the document and PAN/Aadhaar, where
   the originals physically are and what a lost original costs in time, and "nobody needs to be
   paid a percentage to recover this".
7. **What to send back** — a short numbered list of the exact artifacts (statement months, one-line
   confirmation, location of originals). This is the letter's close; it converts the letter into a
   two-way task.
8. **Small-print footer** — basis of the note: sources, dates, what was read off a photograph, what
   is refreshed only half-yearly.

## 3. Never link an internal Content artifact to an outsider

`content.ahfl.in/a/DRA-ART-…` sits behind Google login (`curl` returns 302). A family member
cannot open it, and a dead link in a personal letter reads as sloppiness. Reproduce the substance in
the body instead, and mention the internal report to NDR in the chat reply, not in the email.

## 4. Build, create, and VERIFY the draft

```python
msg = EmailMessage()
msg['From'] = 'Nishant Ranka <ndr@draas.com>'
msg['To']   = '...'
msg['Cc']   = '...'
msg['Subject'] = '...'
msg.set_content(plain_text)              # derive from the HTML: strip tags, convert </p>|</li>|</tr> to \n
msg.add_alternative(body_html, subtype='html')
raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
d = svc.users().drafts().create(userId='me', body={'message': {'raw': raw}}).execute()
```

Then read it back — never trust the create call alone:

```python
d = svc.users().drafts().get(userId='me', id=draft_id, format='full').execute()
# print To / Cc / Subject / From; assert parts == ['text/plain','text/html'];
# assert '<table' in html and the key URL is present in the html part
```

Run it through `terminal` with `HERMES_SESSION_USER_ID=<origin chat id> python3 …` so
`build_service` works in a job session; `sys.path.insert(0, "/opt/hermes")` first.

## 5. Report back to NDR

The chat reply is a changelog, not the letter: draft location and id, To/Cc, the subject line, a
one-line-per-section summary of what the email says, and every caveat you carried forward. Say
plainly that nothing was sent — draft only. If you chose the recipient address by activity, say
which address was stale and why.

## Pitfalls

- **Do not paraphrase the numbers.** Copy the policy/account/transaction figures out of the prior
  report or the source document; a letter to a family member is the worst place to mistype ₹4,00,000.
- **Do not soften "what it does not prove."** Removing it is what turns a helpful note into a
  false reassurance.
- **Do not address it to the company mailbox by default** just because it is the first address in
  the Gmail history — check which one they actually reply from.
- **Do not put the whole thing in a PDF attachment instead of the body.** The instruction is
  "all of the findings IN the email"; many recipients read on a phone and never open attachments.
- **Do not ask NDR to confirm the recipient's address** when the mail history answers it.
