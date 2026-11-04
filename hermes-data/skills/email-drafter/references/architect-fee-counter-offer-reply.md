# Architect fee counter-offer reply (worked 2026-09-05, Arvind Jain / A.J. Architects, Palya row houses)

## Trigger
NDR replies to an architect/consultant's per-sft quotation with a polite rebuttal + lump-sum counter. This is DIFFERENT from the fee-revision *confirmation* email (agreed terms, color-coded boxes) — this is the earlier negotiation step, rejecting a fresh quote and proposing a fixed fee.

## Inputs to gather first
1. **The quotation email** — Gmail search `from:<architect> <project>` → get threadId + attachment (quote is usually a PDF). Extract with `pdftotext -layout` (pymupdf may not be installed).
2. **A prior-rate benchmark from the SAME architect** — search Gmail for the previous project's work-order/fee email (`Ranka North Star (Allalsandra) – Work Order & Fee Basis Confirmation`). The benchmark is the architect's OWN earlier rate on a HARDER scope, which NDR treats as the strongest persuasion.
3. **The complexity contrast** — the new scope's simplicity vs the benchmark's difficulty. Must be argued explicitly, not asserted:
   - Benchmark (hard): full apartment floor plate — constrained, FAR maximisation, privacy, Vastu, multiple iterations.
   - New scope (easy): villa/row-house blocks — 2–3 block designs, then block placement; a few orientation/Vastu-compliant variants. Far less involved.

## Reply structure (NDR's dictation → email)
1. **Candid opener** — thank for quote; say the per-sft fee feels high / hard to justify (NDR's word: "candid").
2. **Perspective paragraph** — "for [prior project], the rate for the entire design proposal was Rs X per sq ft" — spell out WHY that was harder.
3. **Contrast for current scope** — blocks vs floor plate; enumerate the actual design work (2–3 types, placement, Vastu orientation).
4. **Lump-sum proposal** — "since this is one or two options with conceptual drawings, meant for ideas to flow before we decide whom to give the project to, please consider ~₹3,00,000 for the entire exercise."
5. **Explicit deliverable list** (numbered) so the fee is anchored to scope — the counter loses power if deliverables stay vague:
   - 2–3 floor plan options for the villa types (e.g. 2,750 / 3,000 / 3,250 / 3,500 sq ft, capped at 3 types)
   - 1–2 master plan options (walkways, clubhouse position, 1–2 elevator cores for elderly/ground-level access since row villas have no lifts)
   - a few renders / visual references (cladding, balcony railings, facade lighting, overall concept)
6. **Invite a counter, keep the door open** — "if ₹3L is too low, happy to hear a counter — but I couldn't reconcile it with Rs 5/sft for something far more involved. Wait to hear back to finalize."

## Delivery mechanics (validated)
- Threaded reply via `gws_skill_bridge.call('draft_reply_create', service_name='google-draas', message_id=<original msg id>, to=<architect email>, body=<html>, html=True)` — threadId auto-links to the original.
- **HTML with highlighted figures** (`<span style="background:#fff3cd;font-weight:bold;padding:1px 5px">`) on every rupee amount — ₹8/sft quote, ₹5/sft benchmark, ₹3,00,000 offer. Same preference as NDR's fee-revision confirmation emails.
- Verify after create: `drafts().list()` → labelIds contains DRAFT, payload.mimeType == text/html, from ndr@draas.com, threadId matches original.
- Recipient address comes from the email headers (arch_arvind2000@yahoo.co.in) — long-standing partners like Arvind have NO contacts-sheet row; headers are authoritative.
- Deliver the Gmail link `https://mail.google.com/mail/u/0/#all/<message_id>` (drafts view per USER profile preference).