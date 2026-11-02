# Ranka Udaya — Lead Outreach Message Bank (HTML)

Verified 2026-10-01. Trigger: user shares project marketing copy (Google Doc site-summary, RG/brochure insert PDF) and asks for "utility messages", "messages to the leads", "generate in html format", "add maximum messages to select the relevant to send to the leads".

## Workflow

1. **Mine the source doc for product facts.** Google Doc link → `curl -sL "<link>/export?format=txt"` (the /edit URL works with /export?format=txt appended). PDF insert → `pdftotext -layout`. Pull every number, approval, and contact string verbatim.
2. **Rebase ALL figures to the CURRENT collateral, not the first doc read.** Here the website-style summary (branded "Serenity Estate") carried the same core numbers (₹1.2 Cr total, 8×1-BHK, 2,400 sq.ft BUA, ₹60,000/mo, 6%) but the RG insert (branded "Ranka Udaya") is the current objective — flag contact drift: website listed Prakash 99000 93816, RG insert lists **99000 29200**. User said "rename Serenity Estate to Ranka Udaya" — the two brands share the offering. **NDR directed 2026-10-03: the outreach contact number is 99000 93816 (not the insert's 99000 29200).**
3. **Know the target segment before drafting.** This session: "target customers are who looking for investing for rental income and long term asset appreciation" → angle every message at guaranteed income + asset appreciation + comparison + urgency. The user later narrowed to "short utility messages only about the rental guarantee offering and comparable to other investment options... limited period offer (urgency)" → build a Guarantee / Compare / Income / Urgency / Trust set, 3–5 lines each.
4. **Build as a Python builder script** (data array of {cat, title, when, text} → render cards) so a one-line ask ("add more messages", "only these categories") is one rebuild. Pattern: write `build_<project>_message_bank.py`, emit HTML with `<pre class="msg">` per message, then browser-test filters + copy before delivering via `MEDIA:`.
5. **Verify interactivity before shipping** — `browser_navigate file://...` then `browser_console` expression: count cards, click a chip, count visible, confirm copy buttons exist. Deliver the HTML with `MEDIA:<path>` (user is on Telegram; a saved-but-undelivered /tmp file is useless).

## HTML message-bank format (the user-preferred deliverable)

Self-contained single HTML, no external deps:
- **Header band** (DRA dark `#1d1f22` + gold `#FDB913`) — project, count, one-line disclaimer of the figures source.
- **Search box + category chips** — chips filter `data-cat`; search filters a lowercased title+text `data-q` attribute.
- **One card per message**: category pill, title, "when to use" line (⭐ useful — the salesperson needs to know which lead gets which message), message in a `<pre class="msg">` (white-space:pre-wrap so WhatsApp `*bold*` and line breaks survive copy), 📋 Copy button.
- **Copy mechanics**: `navigator.clipboard.writeText` with `document.execCommand('copy')` textarea fallback (file:// contexts).
- **Contact block — copy rule (NDR, 2026-10-03).** Every message carries exactly ONE contact number, `99000 93816`, and NO sender name. Never add a copy-button sign-off (`— Prakash · <number>`) on top of a number already inside the message text: the copied paste then shows TWO numbers. Pitfalls: (a) `copyText(m.text + '\n\n· 93816')` duplicates the number for every message that already ends with one — keep the append handler as `copyText(m.text, b)`; (b) a message with no number (e.g. "Site visit invite") must get its own single contact line added to the data array, not the sign-off append; (c) verify with a per-message JSON sweep — `sorted({m["text"].count("<number>") for m in arr})` must equal `[1]` across all messages, plus zero hits for the sender's name.
- **Select-to-send bar**: per-card checkbox → sticky bottom bar "N selected / Copy selected / Clear" → concatenates selected messages with `— Title —` separators. Plus a "Copy all visible" button for broadcast passes.
- **Escape discipline**: build via Python `html.escape()` on text + `json.dumps` for the JS array; never hand-embed `&`/`"`/em-dashes.

## Product facts snapshot — Ranka Udaya RG (Sep-2026 insert, current)

- ₹1.2 Cr* total investment (plot + 2,400 sq.ft BUA), *T&C apply
- 8 × 1-BHK rental homes per plot · 2,400 sq.ft BUA
- **₹60,000/month rental guarantee · 6% p.a. for 5 years on ₹1.2 Cr**
- HNTDA approved · RERA **TNRERA/30/LO/0642/2026** · ready to build · infrastructure ready
- 38 plots · near Exide Industries, opposite Clover Greens Golf Course, Infosys Sarjapur corridor
- DRA Group (40+ yrs, 10M+ sq.ft delivered, 12,000+ customers); construction Trubuild
- Contact: 99000 93816 (outreach number per NDR 2026-10-03; RG insert's 99000 29200 is NOT used in lead messages) · rankaudaya.in · tagline "Homes that earn for you" / "You own. DRA builds. You earn."
- Site: SY No 240/3A, Seveganapalli Village, Chichuraganapalli, Hosur Taluk, Krishnagiri District
- Locality anchors for messages: Clover Greens opposite · Exide next door · Infosys Sarjapur · 5-star hotel + elite villas upcoming · schools (Indus/TISB/Oakridge) & hospitals 10–30 min · Dmart/Forum Prestige City

## Message-category recipe (investor leads)

- **Utility** — intro, guarantee pitch, self-paying asset, tangible+appreciation, location, limited-period offer, site-visit invite, DRA trust, compliance.
- **Rental Income** — zero-vacancy (guarantee removes vacancy fear — strongest hook), passive monthly income, retirement/pension income, hands-off, 8-units-one-title diversification, rent-vs-FD cashflow.
- **Appreciation** — land+building dual growth, corridor growth story, launch-pricing advantage, leverage (control ₹1.2 Cr asset, ~25% deployed), 10-year wealth view.
- **Compare** — vs FD 5.5%, vs Bengaluru apartments 3–4%, vs Grade-A commercial 6–8% (big ticket + vacancy), vs equity/MFs (market risk) — always one per answer, never a wall.
- **Objections** — "no tenants?" → guarantee; "construction risk?" → DRA/Trubuild/ready-to-build/RERA; "I'll wait" → launch pricing moves; "₹1.2 Cr too big" → EMI structure (rent covers EMI); "is it legal?" → HNTDA+RERA numbers.
- **Follow-up** — fresh lead (personalised [Name]), nudge after no reply, re-engage long-silent (new hook: plots left / pricing round), post-site-visit recap.
- **Closing** — hold-a-plot (48h), final availability push, call/WhatsApp ask.
- **Referral** — warm-lead referral ask, thank-you check-in for booked investors.

Personalised lead messages use `[Name]` placeholders; utility/broadcast messages never do. Sign-off consistent: `— Prakash · 99000 29200`.