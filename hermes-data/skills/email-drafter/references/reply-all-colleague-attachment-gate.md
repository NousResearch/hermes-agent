# Reply-all with colleague in CC + attachment approval gate (worked: DRA-Jiraaf / Sevaganapalli, Sep-2026)

Pattern: an external counterparty (e.g. Vishwas Rao, vishwas@vantagepointadvisors.in) emails a
thread asking questions (e.g. the 3 questions on DRA - Jiraaf: (1) split of land held between
Company and Firm, (2) guidance value of parcels, (3) estimated investment & return). NDR wants
the reply-all to ALSO bring a DRA colleague (e.g. Prakash Singh, psingh@draas.com) into the
chain, and to attach a financial snapshot — but ONLY after he approves the snapshot itself.

## Workflow

1. **Locate the thread** — Gmail search by entity names, not just the person:
   `q='Sevaganapalli'`, `q='Jiraaf'`, `q='Ranka Oasis'` etc. Voice transcription converts
   "Jiraaf Capital" → "Jiraffe" — search both spellings. The latest message in the thread is
   usually the counterparty's (e.g. `RE: DRA - Jiraaf` from Vishwas 03-Sep).
2. **Pull the full thread** (`threads().get(format='full')`) and record EVERY address that has
   ever appeared in To/Cc across messages — reply-all must keep all of them.
3. **Compose per NDR's instruction**: original thread addresses go in To (or CC per his exact
   instruction: "all the email addresses bring them to CC"), and the DRA colleague is added to
   the chain (usually CC). Keep subject `Re: <original>` and set In-Reply-To/References to the
   anchor message so Gmail threads it.
4. **If the email attaches or links Drive files — grant access FIRST, then verify:**
   - `drive.permissions().list(fileId=..., fields='permissions(id,emailAddress,role,type)')` —
     do not assume the recipient can open a file NDR owns.
   - If missing: `drive.permissions().create(fileId=..., body={'type':'user','role':'writer',
     'emailAddress':'psingh@draas.com'}, sendNotificationEmail=False)`.
   - Re-list permissions and print them in the reply to NDR as proof ("Prakash — Editor").
   - Editor (writer) on both PDF and HTML/worksheet copies = "make sure he has editor access".
5. **Attachment approval gate (NDR rule)**: when the deliverable is a snapshot cut from a
   financial model, DO NOT attach it to the draft before NDR approves. Deliver the PDF first
   (MEDIA: path in the reply) + a numbered list of what it shows + any flagged discrepancy,
   and only after his "approved" do you build/attach. Two questions commonly need his call:
   which number to show (model's actual vs narrative target) and which tab is trustworthy.
6. **Draft, never send** — Hermes never sends email autonomously. Build the final draft
   (thread preserved) and hand NDR the draft link
   `https://mail.google.com/mail/u/0/#drafts?compose=<draft_id>`.

## Worked example (DRA - Jiraaf, 04-Sep-2026)

- Thread `RE: DRA - Jiraaf`, anchor = Vishwas's reply asking the 3 questions.
- Recipients preserved: nishantprakash@theyelloweye.com, ndr@draas.com, ndr@drahomes.in,
  vineet@jiraaf.com, prakash.m@jiraaf.com, vishwas@vantagepointadvisors.in,
  kanishk.s@jiraaf.com. Colleague added: psingh@draas.com.
- Attachment: 1-page "Investor's Maths" snapshot from the Jiraffe IRR Summary tab.
- Drive: v13 Ren-Kaj files had ONLY ndr@draas.com on the ACL; granted `writer` to psingh@draas.com
  on both PDF and HTML and verified via permissions().list() before telling NDR access was OK.

## Pitfalls

- Voice transcriber: "Jiraffe" = Jiraaf Capital; "Sevaganapalli" also appears as
  "Seveganapalli"/"Saveganapalli"/"Saverganapali" in filenames — search all variants.
- `drafts().update()` in place requires MIMEText utf-8 parts (see
  gmail-draft-update-in-place.md); plain-ASCII MIME mojibakes ₹/–.
- Never hardcode a recipient address into the draft; resolve via the thread headers, then
  verify against contacts (see Email Recipients hard rule).