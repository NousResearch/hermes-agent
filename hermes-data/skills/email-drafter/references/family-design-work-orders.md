# Family / Kids' Design Work Orders (Ruhaan & Rivaan)

Class: NDR issues personal **design work orders** to his sons for the **914, Embassy Habitat** flat
(bathroom interiors, maker space, home office, etc.). He expects the EXACT same HTML styling every
time ("exact same HTML formatting use") and treats each email as a contract + work-order document
with deliverables, process, and money stated.

## Where the pieces live

- Template (known-good, 2026-10-12 Rivaan bathroom WO): `templates/kids-work-order-email.html`
  — copy it, substitute the `{{PLACEHOLDERS}}`, reuse unchanged.
- First example: Ruhaan "Maker Space & Home Office" WO, sent 10-Oct-2026 from ndr@draas.com
  (message `1a12499bf3481ce2`, thread `1a1249147d0f3f97`). Second: Rivaan "Ground Floor Bathroom" WO
  draft `r-6090176748499275154` in nishantranka@gmail.com.

## HTML style (replicate exactly)

- Outer: `font-family:Arial,Helvetica,sans-serif;font-size:14px;line-height:1.55;color:#222222;max-width:680px` inside `<div dir="auto">`
- Summary card: `background:#1d1d1f;border-radius:6px;padding:16px 18px` with
  `color:#f5a623;font-size:12px;letter-spacing:1.5px;text-transform:uppercase` label,
  white bold 16px title, `#cccccc` 13px deliverable line
- Section headers: `h3` `font-size:15px;color:#1d1d1f;border-bottom:2px solid #f5a623;padding-bottom:4px`
- Bullets: `ul` `padding-left:20px`, `<li><b>…</b> …</li>`
- Signature: "Best," / **Nishant Ranka** / grey 13px "914, Embassy Habitat · Bengaluru"
- Escape properly: `&#8212;` em dash, `&#8594;` arrow, `&#8377;` ₹, `&#39;` apostrophe, `&amp;`, `&quot;`
- The draprojects.in logo `<img>` was in the original template; DROP it for personal-account sends (NDR agreed)

## Account routing

- Personal job → `google-gmail` (nishantranka@gmail.com) — NDR says "send it for my personal email because it's a personal job"
- Company job → `google-draas` (ndr@draas.com)
- Verify WHOAMI on the service before creating the draft (`.users().getProfile()`); assert the expected email.

## Replicating an earlier WO ("same formatting")

1. Find the sent message: `gws_gmail_search` with project keyword + "work order" (e.g. `914 work order`); check BOTH google-draas and google-gmail.
2. Fetch raw/full: `svc.users().messages().get(id=..., format='full')`, walk `payload.parts`, take the `text/html` part, base64url-decode.
3. Adapt, don't rewrite — keep the section numbering, card, and voice.
4. Subject convention: Ruhaan's WO had literal asterisks (`*914 Embassy Habitat — Work Order: …*`); NDR dropped them in review for Rivaan's. Leave them OFF unless asked.

## Recipients & voice traps (IMPORTANT)

- **"Mark Pebley" (also "Pebli" / "Pebbley") = Ruhaan** — "Pebby" is Ruhaan's handle (pebblyshark69@gmail.com).
  First encounter: resolve failed everywhere (contacts sheet, Gmail, sessions); NDR corrected the voice garble.
  When a dictated recipient names no known person → ASK NDR rather than guessing; a family nickname is the answer.
- Ruhaan's Gmail display name is literally `tonaebjfk eh dojf lkw` — never match him by display name.
- Roshini = rnr@draas.com (business convention even for family CC).
- **Rivaan has TWO email spellings in the wild:** `rankarivaan@gmail.com` (NDR's own sent mail, e.g. FIFA World Cup mail; also the Ruhaan WO draft CC history) vs `rivaanranka@gmail.com` (Roshni's calendar invites).
  The 2026-10-12 draft went to `rivaanranka@gmail.com` after NDR confirmed "send it to Rivaan, Ranka" without objection — but treat the address as unconfirmed until NDR has seen either spelling; if in doubt, state which one you used when reporting.

## Drafting with attachment (native tools have NO attachment param)

- Build `MIMEMultipart("mixed")`; `MIMEText(html, "html", "utf-8")`; attachment as **`MIMEImage(bytes, _subtype="jpeg")`** — NOT `MIMEApplication`, which yields `application/jpeg` instead of `image/jpeg`.
- `Content-Disposition: attachment; filename="914-Embassy-Habitat-....jpg"` (rename the image_cache snapshot to a meaningful name).
- `drafts().create(userId='me', body={'message': {'raw': base64url(msg.as_bytes())}})` — DRAFT ONLY, never send.
- Verify: re-`get` the draft `format='raw'`, print To/Cc/Subject, list parts, then sha256(attachment payload) vs sha256(local file) — byte-identical required.
- Report with the compose link (`https://mail.google.com/mail/u/0/#drafts?compose=<draft_id>`) and a summary of money/terms stated.

## Standard WO content (NDR's spec, seen across both examples)

- Section 2 functional requirements FIRST, with an explicit **approval gate**: "get the functional requirement approved by me before proceeding" — written into the work order.
- Deliverables D1–D9: FRD → 2–3 options with mood boards → selected design → Blender model (engineering-grade measurements) → full drawing set (plan, sections, elevations, renders, dimensioned) → material palette + spec → **BOQ** → engineering drawings (plumbing/drainage/electrical/waterproofing) → iteration support.
- Budget discipline: US$-denominated credit budget; cheap models (Gemini-class) for initial renders first; GPT Astra/Opus + Higgsfield ONLY at the end for the final Blender model, sections, elevations and material palette.
- Payment: professional fee with a "normally ₹X, but first job → ₹Y to encourage you" line.