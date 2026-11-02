# Verified EXTERNAL counterparty addresses (Ranka North Star / Bangalore)

Resolve these from Gmail `From:` headers, not the contacts sheet — the sheet has no usable row
for several of them (e.g. `contact_resolver("Arvind Architect")` returns nothing relevant;
`entity_resolver` was erroring `_handle() got an unexpected keyword argument 'task_id'` on
2026-10-29, so fall back to `contact_resolver` + `gws_gmail_search`).

| Person | Role | Address | Evidence |
|---|---|---|---|
| **Arvind Jain** (a.k.a. "Arvind", "AJ", "Arvind Architect") | Principal, **AJ Architects** — architect for **Ranka North Star** (Allalasandra), floor plans, area statement, pre-DCR drawings | `arch_arvind2000@yahoo.co.in` | NDR replies to this address on the North Star plan-review threads; Cc'd with Shrinidhi GV |
| **Shrinidhi GV** | AJ Architects — drawings / area statement | `shrinidhi.a.j.architects@gmail.com` | Cc on North Star plan review |
| **Jayaram Gowda** ("Jayaram", "Jairam", "Jayram sir") | **Vardhan Ventures** — licensing / approval / TDR-DRC consultant for Ranka North Star, Allalasandra (Sy 14/1); phone +91 99802 99080 | `info@vardhanventures.in` | He replies as `Info Vardhan <info@vardhanventures.in>`; NDR's "Dear Mr. Jayaram" licensing-mandate mail went to this address. ⚠️ **NOT** "Vardhan Constructions" (`Vardhan Constructions` is a separate RCC contractor on BuxRanka Hudson) and not `Vardhan Ventures` the invoice/PO counterparty on Ranka Amber — same mailbox, different matter |
| **Sinchana Gowda** | DRA — North Star coordination, plan feedback | `sgowda@draas.com` | internal; never `sinchana@kelsa.io` |
| **Vinod Kumar Das (Rahul)** | DRA — TDR/DRC application pack | `vkdas@draas.com` | internal |

Two same-surname traps seen in October 2026 mail:

- **"Vardhan"** in a Ranka Amber subject line = **Vardhan Ventures' invoice/PO 739 dispute** (drainage,
  Anbu thread) — not the licensing consultant. Read the thread, then pick the address.
- **"Arvind"** is never `Arvind Ankalikar` (L&T Infotech) nor `Aravinda Jyothi` (eng5.blr@draas.com,
  voice-misspelling alias "Arvind AHFL"). Both surface in `contact_resolver`; neither is the architect.

Gazette / town-planning matter (RMP 2015 amendments): the filed final notification is Drive id
`14W5-Eg-7vNNK7MM33PVzrujbC_jqKEjD` in the RD > Bangalore > town-planning folder, and the full
analysis is DRA Content artifact `DRA-ART-2026-000099`.
