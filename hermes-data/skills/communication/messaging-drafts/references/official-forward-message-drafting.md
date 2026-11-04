# Official Forward Message — Format Recipe & Tool-Recovery (verified 2026-09-21)

Session: MP→DC letter for GBJT Serenity Hillview (Sy 93/2 Hurulagurki CLU cancellation),
delivered to Manohar Singh as WhatsApp deep link. User dictated the build preferences
explicitly and asked for a redo — these are now the standing recipe for this class.

## Format recipe for "Respected Sir/Madam" official forward messages

1. **Salutation addresses the SENIOR official only** ("Respected Sir," / "Respected Madam,"),
   never the final authority. Chain: us → intermediary (Manohar) → senior official (MP) →
   deciding authority (DC). Message is addressed to the MP; the hand-off is phrased
   *"I request you to kindly forward this to the Deputy Commissioner for favourable
   consideration"* — do NOT name the DC inside the body.

2. **Key facts in bold, laid out as short labeled lines**, not prose paragraphs:
   - *Subject:* (bold) — e.g. "Cancellation of land conversion order — Sy. No. 93/2,
     Huralagurki Village, Vijayapura Hobli, Devanahalli Taluk"
   - *Land:* extent
   - *Conversion Order:* number + date
   - *Application pending:* File No. + Comp. No.
   Then 1–2 short narrative paragraphs for the trust/benevolent background and the request.
   WhatsApp `*bold*` renders correctly inside the deep-link text.

3. **Word count: 135–150 max.** User: "max 150 words"; 121–130 was accepted as fine.
   Bolding + labeled lines do NOT count as padding — user liked them.

4. **Source authority on conflicting dates/numbers: use the authority's own letter.**
   The internal Drive copy of a request letter may carry a different date than the DC's
   letter citing it — use the DC's date. User verbatim: *"Let us use the date from the DC
   letter… forget what we have on drive."* The government document the recipient will
   recognise wins over our internal copy.

## Hygiene: two-source date verification when drafting from a govt letter

Pulling facts from a scanned govt letter: read the DC's reference list to reconstruct the
chain (applicant request date, Conversion OM number+date, file/comp numbers). Watch for a
date mismatch between the authority's citation and the Drive copy — flag it to the user
before drafting (they will tell you which wins).

## whatsapp_link handler when the tool is missing from the live toolset

If `whatsapp_link` is not in the current tool list, do NOT hand-encode the URL. Import and
call the sanctioned encoder directly — it is the same module the tool wraps:

```python
import sys, json
sys.path.insert(0, '/opt/hermes')
from tools.whatsapp_link_tool import whatsapp_link_tool
r = json.loads(whatsapp_link_tool({"phone": "+919845890316", "text": msg, "platform": "telegram"}))
print(r["url"])   # or r["display_link"] for MarkdownV2 link
```

This produces the tool's exact output — api.whatsapp.com/send, `%EF%BC%86`/`%EF%BC%83`
escapes, parts/split handling, MarkdownV2 display_link — it is NOT manual URL construction.
Verified 2026-09-21: tool absent from live toolset, module present at
/opt/hermes/tools/whatsapp_link_tool.py. Resolve the phone via live contact lookup first
(People API + sheet both agreed on +91 98458 90316 for Manohar Singh).

## Companion learnings (same session)

- DC letter source: `20260709_SerenityHillView_LandConversionCancellation_EnglishTranslation.pdf`
  (Drive id 1zQ3iI07Tv-GgPkQr-T2KUeQfchBZBbk9) / HTML 1LQ56kmAMv_8n0u0CWyqmX3caYs6xpoNf —
  DC letter File No. BRDC-DVH0ALN1/1/2026, Comp No. 1054046, dated 07-07-2026, to Tahsildar
  for inspection report; Conversion OM No. 239145 dated 28-07-2021; Trust request cited as
  26-12-2025 (Drive copy of Trust letter is 20-11-2025 — use DC's date per user).
- Fact-checking trap: BIAAPA *granted* the CLU (order BIAAPA/TP.03/BuVuBa/47/2019-20/387
  dated 12-06-2021) — so "BIAAPA confirmed no approval" must mean no layout/building
  sanction, NOT no CLU. Flag this nuance to the user rather than writing what they dictated
  verbatim.