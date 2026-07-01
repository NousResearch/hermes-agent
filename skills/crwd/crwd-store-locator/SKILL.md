---
name: crwd-store-locator
description: "Find the store for a CRWD live gig — nearest Walmart/Target/etc, its address, phone/store number, and hours, so a member can confirm stock before going. Use when a member asks where to go, where's the nearest store, store hours, or whether a store has the product in stock."
version: 1.0.0
metadata:
  hermes:
    tags: [crwd, store, walmart, target, location, nearest, hours, stock, in-store]
    related_skills: [crwd-gig-execution, crwd-reminders-followups]
    requires_toolsets: [web]
---

# CRWD Store Locator

Help a member find the physical store for a **live (in-store) gig** — the nearest one, its
address, phone/store number, and hours — so they can head over (or call to confirm stock).

## When to Use

- "Where's the nearest Walmart/Target?"
- "Where do I go for this gig?"
- "What are the store hours?" / "Are they open now?"
- "Do they have it in stock?"

## Procedure

1. **Never assume location.** If you don't already know the member's city/ZIP (from the
   conversation or their profile), **ask for it first** — one short question:
   *"What city or ZIP are you in? I'll find the closest one."* Do not guess a location or
   pick a random store.
2. Figure out which retailer the gig uses. If the member is on a specific gig, confirm the
   store chain with `crwd_db` (`get_gig_details` → the gig's `gig_stores` / store_name) so
   you point them at the right retailer.
3. Find the store with `web_search` (and `web_extract` on the store page if needed), e.g.
   *"Walmart near 90210"* or *"Target store 78701 hours phone number"*.
4. Give them, in a tight reply:
   - **Store name + full address**
   - **Phone / store number**
   - **Hours** (and whether it's open now, if you can tell)
5. Suggest they **call ahead to confirm stock** before driving over — stock isn't something
   you can see, so don't claim it's in stock.
6. Offer a deadline reminder if the gig is time-sensitive (see `crwd-reminders-followups`).

## Pitfalls

- **Don't invent a store, address, or phone number.** If search is ambiguous (e.g. a bare
  ZIP matches several stores), give the top match and note there are others nearby.
- Store hours online can be stale — if it matters ("open now?"), tell them to confirm by phone.
- You cannot see live inventory. Never say "they have it in stock" — say "call to confirm."
- Keep it short: this is a phone chat widget. Name + address + phone + hours, not a wall of text.

## Verification

- You asked for the member's location before searching (unless you already had it).
- The reply includes a real store name, address, phone/store number, and hours.
- You pointed them at the retailer the gig actually uses, not just any big-box store.
