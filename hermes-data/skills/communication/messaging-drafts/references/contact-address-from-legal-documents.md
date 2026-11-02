# Contact Details from Legal / Professional Documents

When the contacts sheet/resolver doesn't have a person's residential address OR email, legal documents (partnership deeds, sale deeds, MOUs, agreements, term sheets) always contain the full address of every named party in the "residing at" clause. **Professional certificates submitted to regulators (RERA CA fund certificates, ICAI membership docs, UDIN-attested statements) also carry the professional's email, membership number and firm details** — see the certificate-recovery workflow below.

## Sources to check (in order)

1. **Drive for legal agreements** — search by person name + document type keywords:
   - Partnership deeds (`partnership deed`, `deed of partnership`)
   - Sale deeds, agreements of sale
   - Term sheets, LOIs, MOUs
   - Board resolutions, incorporation documents
   - Settlement agreements, cancellation agreements

2. **RERA / regulator certificates (for professionals — CAs, lawyers, engineers)** — a CA's certificate attached to a RERA fund-statement submission or a project approval contains a rich contact block in the signature: full name, ICAI Membership No., email, UDIN, firm name, firm address, firm phone. Worked example 2026-10-07 (Saurabh T Jain, TSJ & Associates LLP): user supplied `http://www.tsjca.com` + a PDF `CA Certificate Fund Required.pdf`; OCR of the certificate (`vision_analyze` per page; tesseract fallback when the vision router has no keys) yielded **ICAI Membership 221686, email stjain@gmail.com, UDIN 21221686AAAAMA1354, firm TSJ & Associates LLP, #40 4th Floor Lakshmi Complex, KR Road, Bangalore-560002, ph 080-26705995, info@tsjca.com**. The website phone (+91 98864 89873) matched the existing contacts sheet row ("Saurabh CA Ref VK" phone-only) → identity confirmed. Firm website research (About/Contact pages + email pattern probe) can surface `info@<firmdomain>` when only the domain is given.

3. **Google Contacts (People API)** — NDR's personal Google Contacts may have residential addresses saved alongside phone numbers. Fallback: the user may say "check my Google contact."

4. **Email signatures** — Look at forwarded emails from the person; some people include their residential address (though usually it's their office address).

## Enrichment after recovery (dual-source, 2026-10-07)

When an email/membership is recovered from a certificate, update BOTH stores so future lookups succeed:
- **Online contact sheet** ('NDR DRAAS Google contacts', id `1XbSRAXxPLY4cXMTm2rmvKh11Nx3x0aKUxxuWualoV9g') — write email into the row's email column (rows are phone-only for many CA/adviser contacts), fix the name spelling (e.g. "Nithin" → "Nitin; full name Nitin Osthwal Kalpesh, email Kalpeshnitin@gmail.com").
- **Google Contacts (People API)** on the account(s) where the contact exists (`people().get` + `people().updateContact` with the person's etag).
- Resolver-side: `contact_resolver` + `noun_learner` actions (add_alias, update_associations, append_history, increment_score). NOTE: noun_learner's vocab-corrections action has a known code bug (`VOCAB_CORRECTIONS_TAB` referenced but never defined) — the documented fallback is a direct sheet write into the existing vocab_corrections tab.

## Legal document address format

Indian legal documents use the standard format:
```
Mr. FULL NAME, S/o Father's Name, aged about XX years, residing at [FULL ADDRESS]
```

The address includes: flat/villa number, building/society name, survey number (if relevant), locality, city, pincode.

## Worked example: Manohar Singh

**Task:** Find Manohar Singh's home address (MIMS Espacio, villa number)

**Search:** Searched Drive for "Redsol" OR "Farmers Collective" — found `Red Sol Farmers Collective Partnership Deed`.

**Document:** `20260311 Redsol Farmers Collective Partnership_Deed_11Mar2026` (Google Doc, ID: `1-H6ISOicLqXTtM-qqHbKOz_CzGVfx0CJlcRvZXxMBzY`)

**Extracted address:**
> "Mr. MANJUNATH MANOHAR SINGH, S/o Mr. Manjunath Singh, aged about 53 years, residing at Villa H, MIMS Espacio, Survey No. 105/8 & 106/1, Jakkur, Bangalore – 560064"

**Result:** Villa H, MIMS Espacio, Jakkur, Bangalore - 560064

## Tips

- Partnership deeds list ALL partners with full addresses — the most reliable single source
- The principal place of business is often at one partner's residential address
- For company directors, incorporation documents (MOA, Form DIR-12) list residential addresses
- Google Docs can be read directly via the Docs API (`build_service('docs', 'v1')`) — PDFs require OCR or text extraction