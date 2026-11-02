# Message drafted FOR a senior official, forwarded by an intermediary — worked pattern (2026-09-21)

Class: message body is written in the voice of a person two hops away (MP, regulator,
senior official). Chain: us → intermediary (e.g. Manohar Singh) → the official → the
decision-maker (e.g. Deputy Commissioner). The official forwards the text **as-is**.

## Worked example — Serenity Hillview CLU cancellation → MP → DC

- Purpose: senior MP forwards a request to the DC (Bengaluru Rural) to favourably
  consider the Trust's application for cancellation of the residential land conversion
  and restoration to agricultural status.
- Recipient of the link: Manohar Singh (forwarder), not the MP, not the DC.
- User confirmations that shaped the final message:
  - Address it **"Respected Sir"** — the person the message is being drafted FOR is the
    MP; he will forward it. Do NOT name the end decision-maker (DC Anuradha) in the body.
  - **Bold the key facts** (WhatsApp `*...*`): subject line, survey number, area,
    application/file numbers, status facts — makes the message scannable.
  - Use **sections/labels** for readability: `*Subject:*`, `*Land:*`, `*Conversion Order:*`,
    `*Application pending:*` as separate bolded lines.
  - Word budget: **135–150 words** (user said "135 is fine" then "135 to 150").
  - When dates conflict between an archived internal letter and the authority's own
    more recent letter, **the authority's letter wins** — use its file number, comp
    number and dates verbatim.

## Fact base used (Serenity Hillview / Godwad Bhawan Jain Trust, Sy.93/2)

- DC letter (English translation on Drive, folder Serenity Hill View > Legal):
  File No. **BRDC-DVH0ALN1/1/2026**, Comp. No. **1054046**, dated 07-07-2026 — DC's
  request to Tahsildar, Devanahalli for a report on whether any development occurred
  and whether the conversion order can be cancelled.
- Conversion Order: Official Memorandum No. **239145** dated **28-07-2021** (DC Bengaluru
  Rural) — residential conversion of 6A 26G in Sy. No. 93/2, Huralagurki Village,
  Vijayapura Hobli, Devanahalli Taluk.
- Trust's cancellation request letter to DC: dated **26-12-2025**, by **Sri Kumar Pal
  Sisodiya** on behalf of Godwad Bhawan Jain Trust (this is the date the DC letter cites;
  an older archived letter dated 20-11-2025 exists on Drive — use the DC-cited date).
- BIAAPA land-use change order BIAAPA/TP.03/BuVuBa/47/2019-20/387 dated 12-06-2021
  (prior grant of residential CLU; do not claim BIAAPA "granted nothing" — the
  residential CLU WAS granted in 2021; the no-approval claim applies to layout/building
  approval, or to a fresh BIAAPA no-development letter if one exists).

## Link generation

- The `whatsapp_link` tool may not be registered in the live session toolset. The
  sanctioned fallback (NOT hand-encoding, NOT wa.me) is to invoke the tool's own handler:
  ```python
  import sys, json
  sys.path.insert(0, '/opt/hermes')
  from tools.whatsapp_link_tool import whatsapp_link_tool
  r = json.loads(whatsapp_link_tool({"phone": "+91XXXXXXXXXX", "text": msg, "platform": "telegram"}))
  print(r["url"])  # and r["display_link"] for Telegram-safe markdown link
  ```
- Same encoder as the tool (api.whatsapp.com/send, fullwidth ＆/＃ escapes, split logic,
  telegram markdown escaping) — never hand-build the URL with urllib.
- Verify the number against live stores first (contact_resolver → People API ×3 →
  sheet → Gmail), per contact-phone-lookup skill.