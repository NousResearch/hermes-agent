# Forwarding a RECEIVED mail + its photos to a new person, and diffing your prepared draft vs what NDR actually sent

Verified 2026-10-16 (Ranka Iris electrical: Bhavik's snag mail forwarded to Anbu with 5 site
photos plus a work-order-vs-delivered analysis; then a diff of NDR's edited reply against the
draft prepared for him).

Two distinct asks that arrive together. Handle both.

## PART A — Forward a RECEIVED mail with the sender's photos

Ask shape: *"I want to forward Bhavik's email to Anbu, with my points on the top, and all of
the photographs as well, please look at the photographs below for proof of deviations or snags."*

Differences from `forward-existing-sent-mail.md` (which carries a SENT message):
- The source is an INBOX message → fetch by its own message id, `format="full"`.
- The recipient is new so **there is no parent thread** — create a plain new draft with
  `drafts().create`; do NOT use `draft_reply_create`.
- **Forward from the mailbox that HOLDS the original**, not the skill's default `ndr@draas.com`.
  Verified 2026-10-18: a Medybiz Pharma mail sitting in `ndr@ahfl.in` was forwarded from
  `service_name='google-ahfl'`, and NDR was told which account holds the draft. Two reasons this
  matters: the `Message-ID` you thread on belongs to *that* mailbox, and a draft built in the wrong
  account is invisible to him where he expects it (see PART D — never theorise about the account,
  state the verified one). Passing `service_name` explicitly is fine and is NOT the "explicitly
  asked" case the main skill warns about — the source message itself is the instruction.
- The photos are the evidence. They must survive AND be legible.

### A1. Attachment bytes: `attachments().get()`, never `payload.body.data`
Walk `payload` recursively for `filename` + `body.attachmentId`; the inline `body.data` for an
attachment part is normally EMPTY. Download each with
`messages().attachments().get(userId="me", messageId=MID, id=aid)`.

Inline (cid) images from the sender arrive as ordinary parts with a filename — treat them as
attachments. Bhavik's mail carried 5 inline jpgs, all retrievable this way. If one of N fails,
retry that single fetch before reporting a shortfall — an earlier pass on this same mail
returned only 4 of 5 and the 5th was recoverable.

### A2. RENAME the files to self-describing names before attaching
`f16a253a-….jpg` tells the recipient nothing. Rename positionally to
`01_Cable-trench-no-sand-bed-no-bricks.jpg`, `02_Meter-panel-R-phase-busbar-burnout.jpg`, … so
each file reads as evidence on its own, and reference those exact names from the body
(`Proof: Photo 02`). Keep the mapping in an explicit ordered list — never `os.listdir`.

### A3. LOOK at every photo before writing a caption
Run `vision_analyze` on each downloaded image and caption only what is actually in the frame.
This is not optional — it is what lets you tell NDR which of the flagged points are **proven**
and which are not. Real output from this session:

- Photo 01 — open trench, cable lying on bare excavated earth, no sand cushion or bricks, depth
  reads ~0.5–0.8 m → proves the sand-bed item and the depth doubt.
- Photo 02 — close-up of the aluminium bus bar, bolted joint charred with a soot trail down the
  bar → proves the burn-out, and reads as a loose / high-resistance joint, i.e. workmanship.
- Photo 03 — panel door labelled ETV METER, viewing glass shattered with a shard left in the
  frame → proves the glass snag.
- Photo 04 — rust spots on the panel surface under the protective wrap → **not on the sender's
  list**; a new finding, say so.
- Photo 05 — trench under the boundary wall with an exposed conduit → relates to cable routing,
  NOT to the earthing items.

**Always state the gap explicitly:** "only 3 of the 8 flagged points have photographic proof —
water funnel, rod length, chamber, salt/charcoal and jelly have none." NDR acts on that (he
then asks for dated photos of the unproven items). Never let the forward imply that every claim
is evidenced.

### A4. Body shape — his points first, then the evidence, then the forwarded copy
1. His instruction to the new recipient, in his own words (who must act, that the items are NOT
   DONE, that this is work-order-vs-delivered, who will coordinate).
2. The itemised claim list — each with the source document's OWN clause reference (BOQ item
   number + quoted clause) and, where applicable, `Proof: Photo NN`.
3. Then `---------- Forwarded message ----------` with the sender's original text verbatim.
4. Then an attachment index (number, one-line description, `(Photo NN)`).

### A5. Build + verify
`MIMEMultipart("mixed")` → inner `MIMEMultipart("alternative")` carrying plain + HTML → then one
`MIMEBase("image","jpeg")` per photo with `encoders.encode_base64` and
`Content-Disposition: attachment; filename=<renamed>`.
`drafts().create(userId="me", body={"message":{"raw": raw}})`.

Read back `format="raw"` and assert: To, Cc, Subject, `len(attachments) == N` **with the RENAMED
filenames**, and keyword markers present in BOTH plain and html (e.g. `NOT DONE`, the quoted
clause, `SE44/2026-27`, `Proof: Photo 02`). Print the plain and html byte sizes — a healthy part
on one side with a stub on the other means one side silently dropped.

## PART B — "Analyze my reply — I made changes to the draft you prepared"

Ask shape: *"Please analyze my reply because I made a lot of changes to the draft you have
prepared."* He edited the draft in Gmail and sent it, so:

### B1. The Drafts folder is the WRONG place to look
After he edits-and-sends, the draft row is gone — `gws_gmail_draft_list` returned `count: 0`
while the message he sent existed. **The SENT copy is the only record.** Find it with `in:sent`
plus a distinctive subject phrase, then read it with `format="raw"`.

### B2. Report four buckets, in this order
1. **Resolved** — placeholders you had flagged that his edit settled. `Paplu` → *Sanju uncle*;
   `Vescom` → *Bescom*. Say the resolution and that the open question is closed.
2. **Rewritten by him** — sections he replaced with his own position (he added the QC-office
   history and the "precarious situation with him" line to the debits section). Summarise; do
   not re-litigate.
3. **Dropped** — content in your draft that is now in NEITHER email. This is the highest-value
   find: a temporary-power-meter deposit refund task vanished from both mails, so nobody owns
   it. Name it and say who should pick it up.
4. **Still open** — your flagged items he left untouched (unnamed debit heads, an unreconciled
   meter count). State plainly that no decision is recorded yet.

### B3. Flag the small defects he cannot see
Typos in the sent copy (`gets sold ny then`), and cross-email name inconsistencies
(`Sanjeev uncle` in one mail's Cc vs `Sanju uncle` in the other mail's body) — ask whether they
are one person or two rather than assuming.

## PART C — Cc is invisible unless you read raw

The native `gws_gmail_get` tool returns `To` but **not `Cc`**. To prove what actually went out,
decode `format="raw"` and read `msg.get("Cc")`. Verified this session: the wind-up mail carried
`Cc: Piyush Ranka, Manish Ranka, Sanjeev Ranka, Dharmesh Ranka`; the reply carried
`Cc: echamundeshwari@draas.com` alone.

## PART D — Never speculate about which account holds the draft

Report the account from a verified read only. Offering "you may be looking at another account"
as a hypothesis reads to NDR as an admission that you created it in the wrong one — he replied
*"Why ahfl.in account? This should be in the draas.com account."* State the verified account, the
draft id and where to click. If he says he cannot see it, search that account by subject and
report the draft id — do not theorise about his phone.
