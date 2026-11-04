# Ranka Udaya RG — Rental-Guarantee Product Facts & Lead Outreach Message Bank

## RG product facts (the NEW sales framing, Oct 2026)

The RG insert ("Homes that earn for you", `RankaUdaya_RG_Insert_revised.pdf`) is now the lead-generation tool for Ranka Udaya — and it is a DIFFERENT product story from the raw-plots framing (see the Ranka Udaya outreach kit section in SKILL.md).

Verified insert numbers:
- **₹1.2 Cr*** total investment (plot + 2,400 sq.ft BUA)
- **8 × 1-BHK rental homes** · 2,400 sq.ft BUA
- **₹60,000/month rental guarantee for 5 years**
- **6% p.a. for 5 years on ₹1.2 Cr**
- Ready to build · Infrastructure ready · **HNTDA approved**
- **RERA TNRERA/30/LO/0642/2026**
- Contact **99000 29200** · site **rankaudaya.in**
- "A DRA GROUP INITIATIVE | YOU OWN. DRA BUILDS. YOU EARN."

NOTE: the older Serenity-Estate web copy carried the same numbers (₹1.2 Cr, 8×1-BHK, ₹60k/mo, 6%, 38 plots) under the Serenity Estate name — the project is the SAME, now branded Ranka Udaya. When drafting for the RG product, use the insert's phone 99000 29200, not the older plot-sales contact.

## Lead Outreach Message Bank — HTML selector pattern

**Trigger:** user asks to "generate messages for leads" / "utility messages to attract investors for <product>" — especially when they want a reusable library to pick from per lead, or specify an audience (rental-income investors, appreciation investors) and an angle (limited-period offer, guarantee).

**Deliverable shape — self-contained HTML selector page:**
- Search box + category chips (Guarantee / Compare / Income / Appreciation / Objections / Follow-up / Closing / Referral) that filter cards in JS.
- One card per message: category pill, short title, "when to use" line, the WhatsApp-ready text in a `<pre>`, and a per-card **📋 Copy** button (navigator.clipboard with a textarea `execCommand('copy')` fallback — works on file:// pages).
- A sticky select bar: each card has a checkbox; "Copy selected" concatenates chosen messages (title + text + sign-off); "Copy all visible" copies the filtered set.
- Messages authored for WhatsApp: short (3–6 lines), emoji-led, `*bold*` emphasis, one CTA per message, same sign-off block appended on copy (e.g. "— Prakash Singh · Ranka Udaya · 99000 29200").
- Build from a Python script with `MESSAGES = [{cat,title,when,text}]` rendered to cards and `MSGS = json.dumps(...)` embedded in `<script>` — keep the builder script on disk so a one-line content change = one rebuild.

**Audience-targeting rule (rental-income + appreciation investors):** keep every message anchored on the *guarantee* (income certainty, ₹60,000/mo for 5 yrs), the *comparison* (vs FD ~5.5%, vs apartment yield 3–4%, vs commercial 6–8% with vacancy risk, vs equity — "when markets behave"), and the *urgency* (limited-period launch offer, 38 plots, "today's price is the cheapest it will ever be"). Every message carries the "⏳ limited-period offer" hook — reads natural, not pushy.

**Delivery:** send the HTML file via `MEDIA:` immediately (user is on Telegram); verify in browser (`cards.length`, chip-filter counts, copy-button existence) before shipping.