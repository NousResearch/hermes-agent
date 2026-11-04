# NDR Email Formatting & Task-Splitting Rules (verified 2026-09-13)

These rules come from NDR's explicit corrections while briefing Sarthak on two tasks
(Dinesh Ranka tax follow-up + Ranka Oasis cash-flow models).

## 1. NO forced line breaks — paragraphs must flow at full client width

- Complaint (verbatim intent): drafts arriving with *"a lot of line termination mid-sentence…
  prepared and fitted for 8, 9, 10, 12 words per line and forced line termination"* were
  rejected — "that's not how I want my emails to be drafted" even though he has a wide screen.
- Rule: in the **text/plain** part, each paragraph = ONE logical line. Never hand-wrap at
  ~40-60 chars. The client wraps at its own width.
- For briefings/presentations, NDR explicitly asked: *"the email is redrafted using HTML, CSS,
  so the presentation is flawless."* Ship **multipart/alternative** = text/plain (flowed) +
  styled text/html.

## 2. Working recipe (used for the Sarthak briefing redrafts)

- Build `MIMEMultipart("alternative")`; attach `MIMEText(plain, "plain", "utf-8")` +
  `MIMEText(html, "html", "utf-8")`; `raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()`;
  `gmail.users().drafts().create(userId="me", body={"message": {"raw": raw}})`.
- HTML style that landed well: Arial/Helvetica 14px, line-height 1.65, colour #222, max-width
  860px wrapper, section headers as white-on-navy (#1F3864) rounded bars, h3 dark navy,
  ordered/unordered lists with margin. Inline `<style>` in `<head>` is fine for Gmail.
- Plain-part generation: take the HTML, replace `<br>` with newline, strip tags, collapse
  blank lines — but keep each paragraph as one logical line block.
- Verify after create: `drafts().get(format="full")` and assert BOTH mime parts
  (text/plain + text/html) exist plus subject/To.

## 3. One email per task (NDR rule)

- When NDR dictates ONE briefing covering multiple workstreams, he wants **separate draft
  emails — one per task** — not one combined email. His words: *"split it into two different
  emails. I would like one email per task… basically two different task emails I am sending
  out to him."*
- Subject names the single task (e.g. `Task: Dinesh Ranka (ABHPR8430M) — Income-Tax Demands —
  Own the Follow-up (dossier & tracker)` vs `Task: Ranka Oasis — Two Cash Flow Models
  (Actual-Cost & Jiraaf JV)`). Section numbering restarts per email; each email stands alone
  (own intro + own closing).
- Delete the superseded combined draft only AFTER the per-task drafts are created + verified.

## 4. User-edited drafts: RE-READ the current draft before reworking

- NDR frequently edits a Hermes-created draft in Gmail (removes links, rewrites sections),
  then says "take my updated draft". He may have removed things deliberately — e.g. he deleted
  the whole references block (project folder + engineering-team model + term-sheet links) so
  the assignee wouldn't be confused by an irrelevant/inconsistent model.
- ALWAYS `drafts().get(id=..., format="full")` and work from the CURRENT text/plain part —
  never from memory of what you originally wrote. Preserve his removals exactly; do not
  re-add what he deleted (offer to re-add later instead).
- Splitting recipe that worked: fetch draft → extract text/plain → split on the `=====`
  separator → locate section headers (`*1.`, `*2.`, …) → rebuild each email as header +
  content → adapt only the intro/closing so each reads standalone → create new drafts via API
  → delete the old combined draft → verify drafts list shows exactly the new drafts with
  correct subjects + mime parts.