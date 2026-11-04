# Long personal / advisory letter → formatted HTML draft

Verified 2026-10-16 (NDR → Kishan Murjani Nair, "Flameback - an honest note, and one number I keep
coming back to", ~1,400 words, 30 paragraphs). Use when NDR says *"well formatted so it's easy to
read, bolding whatever is appropriate"* about a long prose letter he has already approved.

**Library note (2026-10-16):** `email-drafter/SKILL.md` is **at the 100,000-char ceiling** (~100.8 KB,
112 reference files), so `skill_manage(action='patch', file_path='SKILL.md')` is **REFUSED** with
`content is 100,140 characters (limit: 100,000)`. Put new guidance in `references/` — a new reference
file is still discoverable without a SKILL.md pointer, because it shows up in `linked_files` when the
skill is loaded. SKILL.md needs trimming before it can take edits again.

## The hard rule: format only, never reword

When the text has been **approved by a third party** (a spouse, a co-investor, a lawyer), the wording
is frozen. Formatting is a presentation change; any word change invalidates the approval. Rebuild the
body **verbatim** from the copy that was actually sent for review — read it back with `gws_gmail_get`
or the raw API, never from memory. Then prove it: keyword-assert the distinctive phrases and numbers
on **both** the plain and HTML parts.

## Never reply into the review thread

If the letter went to a reviewer first (e.g. NDR → Roshini → Kishan), that thread contains the private
review correspondence. Sending to the final recipient as a **reply** leaks it. Always create a **new
draft, new thread**, even though the subject line is the same. Do not pass `threadId` to
`drafts().create`.

## HTML shape that survives Gmail

- **Inline styles only.** Gmail strips `<style>` blocks and `<head>` CSS. Every rule goes on the element.
- Body: `Georgia,'Times New Roman',serif`, `16.5px`, `line-height:1.72`, `color:#222`, wrapped in a
  `max-width:660px; margin:0 auto` div. Serif at generous leading reads as a personal letter, not a memo.
- Paragraphs: `margin:0 0 20px 0` — margin, never `<br><br>`, so it reflows on mobile.
- Callout block for the emotional/pivotal paragraphs (the reviewer's paragraph, the actual ask):
  `background:#FBF7F0;border-left:3px solid #B08968;padding:18px 22px;border-radius:0 5px 5px 0`.
  Two at most — one at the top carrying the shared framing, one at the ask. More than two and none land.
- Bullets: real `<ul>`/`<li>` with `padding-left:24px`, `margin:0 0 12px 0` on the `li`. Never hand-typed
  dashes for a formatted build.
- Entities (`&rsquo; &ndash; &ldquo; &rdquo; &amp;`) are ASCII and safe. Raw non-ASCII in the **Subject**
  is the trap (forces RFC2047 encoded-words, which Gmail's draft renderer mangles) — keep the subject
  pure ASCII and write `Rs`, not `₹`.

## Bold discipline

Bold the load-bearing claims, not the prose: the single reassurance ("**I am not worried about the
money.**"), the thesis sentence ("**PMS in India is not a startup that goes from 1 to 100.**"), each
hard number in the model, the reviewer's key clause, the ask, and the closing line. 20-30 `<strong>`
tags over ~1,400 words is right. Bold every paragraph and the emphasis stops meaning anything.

## Verification that actually proves it (all cheap)

1. `labelIds == ['DRAFT']` — proves unsent. Re-check *after* any subsequent tool call; a draft can be
   reported created and then not persist. **Read it back via `drafts().get(id=draft_id, format='raw')`,
   never `messages().get(id=message_id)`** — Gmail re-saves a draft the user opens in the web UI and
   mints a *new message id*, so the old one 404s and looks exactly like a deletion. The **draft id is
   stable, the message id is not.**
2. Read `To` / `Cc` / `Subject` off the **raw** message with `decode_header` — Cc is invisible in the
   JSON draft summaries.
3. Walk the parts; assert **both** `text/plain` and `text/html` exist.
4. Keyword-assert the same phrase list against both bodies (strip `'` vs `&rsquo;` when comparing).
5. **Paragraph-integrity check** on the plain part: split on `\n\n`; flag any line inside a multi-line
   paragraph that ends without terminal punctuation — that is a mid-sentence wrap.
6. Count `<strong>`, callout blocks and `<li>` in the HTML as a structural sanity check.
7. Optional visual: write the HTML to disk and `browser_navigate` to `file:///...`, then `browser_vision`
   to confirm typography/spacing and catch defects. The browser stack reads local files fine.

## If the user opens / edits the draft before you finish

A user-editable draft is a moving target. After creating it, **re-read and DIFF the live body against
the text you authored** — never assume your build is what is sitting there. A word-level diff
(tokenise both, `difflib.SequenceMatcher`) surfaced in this session that NDR had made two edits, the
second an **unfinished sentence** ending `"...on the most honest & ."` — a dangling ampersand that
would have gone out broken. **Only the diff catches that**: a keyword check passes it happily, because
every keyword is still present.

Rules that follow:
- **Never silently clobber a user's in-progress edit.** If they are mid-edit, flag the defect, quote the
  exact broken text, and offer 2-3 concrete completions. Apply only once they pick one.
- Normalise before diffing: strip `\r`, unify curly quotes/dashes, drop `*` (Gmail's plain part renders
  bold as `*…*`), and **rejoin hard-wrapped lines** in the plain part — Gmail rewraps it, so a naive
  line-by-line compare reports the entire body as changed.
- **Hex colours get lowercased by Gmail's editor** (`#FBF7F0` → `#fbf7f0`), so a case-sensitive search
  for a callout colour reports **0** and looks like the formatting was stripped. Count the *structural*
  markers instead (`border-left`, `background`, `<strong>`, `<li>`).

Good news, and worth knowing: the editor **preserves** the formatting. `<strong>`, `<ul>/<li>`, the
serif `font-family`, the `max-width` column and the styled callout `<div>`s all survived a
create → open-in-UI → re-save round trip intact.

## Plain part

Same words, one logical line per paragraph, `- ` for bullets. No wrapping — the integrity check in (5)
depends on it.

## See also

- `new-email-html-attachment-nonascii.md` — the `EmailMessage` + `drafts().create` recipe this builds on.
- `flameback-kishan-case.md` — the worked example (approved text, reviewers, and the two-thread rule).
