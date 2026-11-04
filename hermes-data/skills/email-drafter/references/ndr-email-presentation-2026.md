# NDR Email Presentation & Multi-Task Drafting Preferences (verified 2026-09-13)

Worked example: Sarthak task-briefing batch (P-151..P-153 — Dinesh Ranka tax follow-up +
Ranka Oasis cash-flow models). NDR corrected the first combined brief TWICE.

## Rule 1 — HTML+CSS presentation, never hard-wrapped plain text

NDR: *"there is a lot of line termination mid-sentence… prepared and fitted for 8, 9, 10,
12 words per line and forced line termination, even if I have a very wide screen… that's not
how I want my emails to be drafted… the email is redrafted using HTML, CSS, so the
presentation is flawless."*

- Draft emails as **multipart/alternative**: a flowing `text/plain` fallback (paragraphs on
  single logical lines, NO manual wrapping — clients wrap at their own width) + a styled
  `text/html` part.
- HTML conventions NDR accepts (real-estate-financial-modeling styling family):
  - body: Arial/Helvetica 14px, line-height 1.65, color #222, max-width ~860px
  - section headers: white on navy #1F3864, rounded 4px, padding 8px 12px
  - sub-heads: navy #1F3864; lists with breathing room
- Gmail's composer wraps automatically; any mid-sentence `\n` we inject is a defect.

### Rule 1b — Multi-asset / inventory dossiers: HTML BODY, not a PDF attachment (corrected 2026-09-19)

When NDR asks to consolidate many assets, properties, or records into one pack for a
colleague, **put the whole document in the email BODY as HTML — do not attach a PDF.**
His correction after receiving a 14-page PDF: *"this entire HTML itself, let's send it to
him in an email. Instead of creating as a separate DRA liquidation inventory master HTML,
put a briefing opening note and let's send it in an email, the entire HTML content itself."*
He reads on a phone; an attachment is a download step, a body is the message.

Shape that landed (DRA Group Inventory for Liquidation → Prakash, 15 assets / 7 groups):

- **Branded wrapper, single self-contained HTML** — inline `<style>`, `max-width:860px`,
  gradient header block with title + subtitle + "Prepared <date> · For <name> · Confidential".
- **Opening briefing note box FIRST** — a tinted panel saying, in his words, "here is
  everything, in one place": what was consolidated from where, what each asset entry
  contains, and **what you need from the recipient**. This IS the "put a briefing opening
  note" instruction — it replaces a cover email and a cover page.
- **One `<h2>` per asset group, one `.asset` card per asset** — two-column key/value
  `<table>` for particulars (unit, owner, rate ask, market benchmark), then a `.lnk` block
  listing every collateral as inline `<a href>`.
- **Callout boxes for the honest findings** — amber `.flag` for caveats the recipient must
  read (title gaps, aggressive asks), red `.q` for questions that block a rate. NDR wants
  the pushback visible in the document, not only in the chat reply.
- **Closing "What we are asking X to do" numbered list** + the open questions restated.
- Send as a **single `text/html` part** (`msg.add_alternative(html, subtype="html")`); a
  short `set_content(...)` fallback is fine.

**Rate research belongs inside each asset card, beside the owner's ask.** Quote the live
portal benchmark next to our number and say plainly where we sit above market. He asked for
*"the correct rate"* — that means our ask **and** the market, for every asset.

**"Search the marketing collaterals and confirm from me" = build the link map, then ASK.**
When he says *"for every project, please search the marketing collaterals or relevant
sketches, maybe the land document folder… confirm all of that from me"*, he wants (a) the
real Drive collateral for every asset linked in the draft, and (b) a short list of what you
could not resolve, asked as questions. Never silently link a plausible-looking folder.

**Share-access sweep before sending** (his explicit ask: *"make sure that on all the
relevant links he has got viewer access"*): for every file AND folder linked, run
`permissions().list(fileId=…)` first; if the recipient is absent,
`permissions().create(type='user', role='reader', emailAddress=…)` with
`sendNotificationEmail=False`. Report the verified count in your reply. A linked-but-unshared
Drive URL is the #1 defect in this genre — the recipient hits "Request access" and the pack
is useless.
**Ownership trap:** a permission create returns `403` when the file isn't owned by the
active account (observed: a collateral video owned by `sales1.blr@draas.com`). That is the
signal to link the **parent folder** — which the recipient may already have — instead of the
file.

**Spell out money-losing blockers in the document.** Where an asset cannot be cleanly sold
(undelivered title, khata not in our name, live High Court matters), say so in a `.flag` box
at the top of that asset's card. Burying it in an appendix is the failure mode he corrects.

## Rule 2 — One email per task (split multi-workstream briefings)

When NDR dictates several workstreams for one reportee ("detailed task briefing email"),
the combined draft MANAGED to be split later: he will review, edit, then ask for
**one draft per task**. On the split:

1. Read his EDITED draft — he deletes links/files he deems irrelevant ("the entire Ranka
   Oasis folder with all legal everything is irrelevant", "the model put together by the
   engineering team is completely different — I don't want to confuse Sarthak"). He may also
   drop a whole references block. **Preserve his edits exactly — never re-add what he removed.**
2. Split on the `===...===` section headers; keep each section verbatim; adapt only the
   greeting/closing (intro must not say "three workstreams" if the email now carries one).
3. Delete the superseded combined draft (he will send the new ones; a leftover combined
   draft risks the wrong send).
4. Verify each new draft by keyword checks (must_have / must_not_have on the text/plain part)
   — e.g. Draft A must NOT contain the other task's section, Draft B must not contain the tax
   content.

Subjects in the worked example: `Task: Dinesh Ranka (ABHPR8430M) — Income-Tax Demands — Own
the Follow-up (dossier & tracker)` and `Task: Ranka Oasis — Two Cash Flow Models
(Actual-Cost & Jiraaf JV)` — subject names the single task in the email.

## Rule 3 — "Share the folders with him" = share matrix, then report

When NDR assigns a task owner and says share the folders:

- Working folders/docs the reportee must MAINTAIN → Editor (writer): dossier doc, tracker
  sheet, demand-letter folder, project records folder.
- Reference-only docs → Reader (term sheets, templates).
- If the reportee owns the file (his own model), no share needed — verify with
  `permissions().list` before assuming; a 403 "Sorry, you do not have permission to share"
  means the file isn't owned by ndr@draas.com (someone else owns it — leave it alone).
- Report the full matrix (file → role) in the reply so NDR can adjust.

## Rule 4 — Real-estate voice alias: "Giraffe Capital" = Jiraaf Capital

In Oasis/Sevaganapalli briefs, "Giraffe" (voice) = **Jiraaf Capital** (Ranka Oasis × Jiraaf
JV, ~6.2 ac at Sevaganapalli, Hosur — term sheet dated 25-Aug-2026). Use both names in the
brief ("Jiraaf Capital ('Giraffe' in briefs)") so the reportee recognises the deal from
either name.