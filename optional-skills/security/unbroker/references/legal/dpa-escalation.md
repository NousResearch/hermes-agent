# DPA escalation — the EU/EEA/UK Article 77 complaint path

The killer feature of the GDPR pipeline: when a data broker fails to honour an Article 17
erasure request within the 30-day Article 12(3) window, the subject can file an Article 77
complaint with their national supervisory authority. CCPA has no equivalent.

This guide covers the operational steps an EU/EEA/UK subject (or their agent) follows to escalate.

## The decision: when to escalate

`pdd.py next` surfaces the escalation automatically when **all** of these hold:

- The subject's residency is an EU member state, an EEA state, or the UK
  (`dossier.is_eu_residency()` returns true).
- The broker has `gdpr_scope: true` (brokers flagged as unlikely to honour Art. 17 are not
  escalated — `escalate` against them is still legal but rarely productive).
- The case is in `submitted` or `awaiting_processing` state (the broker was notified and
  has had its chance).
- 35+ days have elapsed since the Art. 17 was filed (Art. 12(3)'s 30 days + a 5-day grace
  for time-zone and clock-skew tolerance). The constant is `DPA_ESCALATION_THRESHOLD_DAYS`
  in `scripts/autopilot.py`; it can be raised to 90 if you want to wait for the full
  maximum extension window (see below).
- The subject has not already filed a DPA complaint for this broker
  (`dossier.preferences.dpa_complaint_filed_<broker_id>` is unset).

If all five hold, `next_actions` adds a `dpa_escalate` (or `dpa_escalate_generic`) action
to the output queue with the rendered command. The agent executes the command, which
produces a complaint file at `subjects/<subject_id>/drafts/dpa_complaint_<dpa>_<broker>.txt`.

### Note on the 35 vs 90 day threshold

Article 12(3) GDPR gives the controller:

- **30 days minimum** (one month from receipt — the standard response window).
- **+60 days optional extension** (two further months for complex requests, but the
  broker MUST notify the extension within the first month).

So the realistic response window is **30 to 90 days**. The skill surfaces escalation at
**35 days** — that's the common case where the broker does not exercise the extension
clause. If the broker explicitly invoked the 2-month extension, you can raise the
threshold to 90 (`DPA_ESCALATION_THRESHOLD_DAYS = 90` in `scripts/autopilot.py`) to wait
for the full maximum window before filing. The skill's `dpa_escalate` action `why` text
reminds you of the 90-day ceiling when it surfaces the escalation.

## The procedure (per step)

### 1. Render the complaint

```bash
python3 scripts/pdd.py escalate <subject_id> <broker_id> \
    --request-date 2026-05-27 \
    --request-channel PEC
```

- `--request-date`: the date the Art. 17 was sent (default: lookup in dossier preferences).
- `--request-channel`: how the Art. 17 was sent (email / PEC / web form / post). The complaint
  cites this so the DPA knows how the broker received the prior request.
- The complaint is rendered with a DPA-specific template for every shipped adapter: localized
  templates where already represented (for example Garante, CNIL, BfDI, AEPD, AP/GBA, UODO,
  IMY, Datatilsynet DK, and Finland), and authority-specific English packages for the remaining
  EU/EEA adapters.
- The output JSON includes the web-form URL, email, and (for Italy) the PEC address.

### 2. Review the draft

Open `subjects/<subject_id>/drafts/dpa_complaint_<dpa>_<broker>.txt`. Check for:

- Correct spelling of your name and contact email.
- Correct address (rendered from `identity.current_address`).
- Correct broker name and the date you sent the prior Art. 17.
- The subject-matter is the right broker (the render takes the first `broker` argument).

### 3. Gather attachments

The DPA will require:

1. **A copy of the Art. 17 request you sent to the broker.** Screenshot of the sent email
   (including headers), or a copy of the PEC receipt, or a postal receipt, depending on the
   channel you used.
2. **Proof of receipt.** Most brokers acknowledge Art. 17 receipts; if not, the postmark /
   PEC delivery receipt is the next-best evidence.
3. **Any broker response.** If they replied (refusal, partial erasure, "we don't have your
   data"), include it. If they did not respond, the complaint notes this and asks the DPA to
   treat the silence as non-compliance.
4. **A copy of your photo-bearing ID.** Most DPAs require this for the first complaint
   (anti-fraud). Send a **copy**, never the original.

### 4. File with the DPA

| DPA | Residency | Language | Channel | Web form |
|---|---|---|---|---|
| Garante | EU-IT | it | PEC preferred (`protocollo@pec.gpdp.it`) or web form | https://www.garanteprivacy.it/diritti/come-agire-per-tutelare-i-tuoi-dati-personali/reclamo |
| CNIL | EU-FR | fr | Online form or post | https://www.cnil.fr/fr/adresser-une-plainte |
| BfDI / Land DPA route | EU-DE | de | BfDI page for federal/telecom; private-sector complaints normally go to the Land DPA | https://www.bfdi.bund.de/DE/Buerger/Inhalte/Allgemein/Datenschutz/BeschwerdeBeiDatenschutzbehoereden.html |
| ICO | UK | en | Online form | https://ico.org.uk/make-a-complaint/ |
| AEPD | EU-ES | es | Electronic office complaint models | https://sedeaepd.gob.es/sede-electronica-web/vistas/infoSede/tramitesCiudadanoReclamaciones.jsf |
| Autoriteit Persoonsgegevens | EU-NL | nl | Complaint/tip form | https://klachten.autoriteitpersoonsgegevens.nl/ |
| APD-GBA | EU-BE | nl/fr | Citizen portal complaint | https://www.dataprotectionauthority.be/citizen/actions/lodge-a-complaint |
| DSB | EU-AT | de | Email/post with bilingual templates | https://data-protection-authority.gv.at/data-protection-in-austria/right-to-lodge-a-complaint |
| DPC | EU-IE | en | Online form; phone is guidance-only | https://forms.dataprotection.ie/contact |
| CPDP Bulgaria | EU-BG | bg | Complaint/alert route; Bulgarian translation expected | https://cpdp.bg/en/lodging-complaints-and-alerts/ |
| AZOP | EU-HR | hr | Rights-violation request to the Croatian DPA | https://azop.hr/rights-of-individuals/ |
| Cyprus Commissioner | EU-CY | el | Complaint to the Commissioner | https://www.dataprotection.gov.cy/dataprotection/dataprotection.nsf/page1i_en/page1i_en?opendocument= |
| UOOU Czechia | EU-CZ | cs | Complaint against a controller/processor | https://uoou.gov.cz/verejnost/stiznost-na-spravce-nebo-zpracovatele |
| AKI | EU-EE | et | Contact AKI for privacy-rights violations | https://www.aki.ee/en/guidelines-legislation/how-can-we-help-foreign-persons-and-authorities |
| HDPA | EU-GR | el | Complaint to the Hellenic DPA | https://www.dpa.gr/en/individuals/complaint-to-the-hellenic-dpa |
| NAIH | EU-HU | hu | Authority procedure from a data-subject complaint | https://www.naih.hu/data-protection/international-affairs |
| DVI | EU-LV | lv | Complaint concerning personal-data processing | https://www.dvi.gov.lv/en/services/complaint-concerning-processing-personal-data |
| VDAI | EU-LT | lt | Recommended complaint form / e-service | https://vdai.lrv.lt/en/services |
| CNPD Luxembourg | EU-LU | fr | Online complaint form | https://cnpd.public.lu/en/particuliers/faire-valoir/formulaire-plainte.html |
| Malta IDPC | EU-MT | en | Data Protection Complaint route | https://idpc.org.mt/contact/ |
| CNPD | EU-PT | pt | Complaint/participation forms | https://www.cnpd.pt/cidadaos/participacoes/ |
| ANSPDCP | EU-RO | ro | GDPR complaint page and written submission routes | https://www.dataprotection.ro/?lang=en&page=Plangeri_RGPD |
| UOOU Slovakia | EU-SK | sk | Proceedings on personal-data protection; Slovak template expected | https://dataprotection.gov.sk/en/office/proceedings-on-protection-personal-data/ |
| Information Commissioner Slovenia | EU-SI | sl | Complaint route via Information Commissioner | https://www.ip-rs.si/en/ |
| UODO | EU-PL | pl | Traditional or electronic complaint | https://uodo.gov.pl/en/680/1402 |
| IMY | EU-SE | sv | E-service; email/letter fallback for protected identity | https://www.imy.se/en/individuals/forms-and-e-services/file-a-gdpr-complaint/ |
| Datatilsynet | EU-DK | da | MitID complaint form; alternative contact routes available | https://www.datatilsynet.dk/english/file-a-complaint |
| Data Protection Ombudsman | EU-FI | fi | Secure report-of-fault form | https://tietosuoja.fi/en/report-of-fault-in-personal-data-processing |
| Datatilsynet Norway | EEA-NO | no | ID-porten complaint form; written fallback | https://www.datatilsynet.no/om-datatilsynet/kontakt-oss/klage-til-datatilsynet/ |
| Personuvernd | EEA-IS | is | Icelandic DPA public site; use EDPB list as routing fallback if needed | https://www.personuvernd.is/ |
| Datenschutzstelle Liechtenstein | EEA-LI | de | Free complaint right and published forms | https://www.datenschutzstelle.li/datenschutz/themen-z/beschwerderecht |
| (generic) | EU / EU-EEA | en | Choose the competent DPA from the EDPB member list | https://www.edpb.europa.eu/about-edpb/our-members_en |

Run `python3 scripts/pdd.py dpas` to list the shipped adapters, or
`python3 scripts/pdd.py dpas --residency EU-ES` to resolve a subject's filing channel.

### 5. Record the filing

Once the complaint is filed with the DPA:

```bash
python3 scripts/pdd.py escalate <subject_id> <broker_id> --file
```

This:
- Records the filing timestamp in `dossier.preferences.dpa_complaint_filed_<broker_id>`.
- Records which DPA the complaint went to in `dossier.preferences.dpa_complaint_dpa_<broker_id>`.
- Transitions the case to `human_task_queued` with `human_task_reason="DPA complaint filed with <DPA name>"`.
- Adds a `dpa` field to the case for downstream reporting.

After this, `pdd.py next` will **stop** surfacing the escalation action for this broker — the
loop runs until the DPA responds or the next re-check window arrives.

### 6. Wait for the DPA

- **Garante**: typical response 3-6 months for the first contact, longer for substantive
  decisions. The PEC filing is timestamped; the Garante is bound by Art. 78(2) to respond
  within a reasonable time.
- **CNIL**: faster initial triage (typically 4-8 weeks); substantive decisions take months.
- **BfDI / Land DPAs**: variable; private-sector Land DPAs are typically faster than BfDI.
- **ICO**: typically 4-8 weeks for the first acknowledgement; substantive decisions take
  longer. The ICO's complaint procedure is well-documented.

If the DPA finds the complaint substantiated, it can order the controller to erase the data
under Article 58(2)(c) and impose administrative fines under Article 83. The subject is not
a party to the enforcement action but receives the outcome.

## When escalation does NOT make sense

- **The broker is in a non-EU/EEA/UK jurisdiction and refuses to engage.** DPA escalation is a
  lever against controllers subject to GDPR. For a US-domiciled broker with no EU presence,
  the DPA can still try (GDPR has extraterritorial scope under Art. 3(2)), but the chances of
  success drop substantially.
- **The broker has already erased the data.** `pdd.py next` will surface the broker as
  `confirmed_removed` once the re-scan re-check confirms the erasure; no escalation needed.
- **The subject wants faster action than a DPA timeline allows.** DPAs are slow. For
  time-sensitive removal, the right path is a private right of action under Article 79 in
  national court (which the unbroker skill does not currently automate).

## Operational notes

- **Audit trail.** Every transition through `pdd.py record` and `pdd.py escalate` writes to
  `audit.jsonl` with the timestamp, event, broker, and any disclosed fields. The DPA
  complaint and the prior Art. 17 request are both audit-logged.
- **Idempotency.** Running `pdd.py escalate --file` twice for the same broker is safe; it
  overwrites the timestamp with the most recent filing.
- **Re-filing after rejection.** If the DPA rejects the complaint (e.g. "you didn't exhaust
  the controller's complaint process"), the subject can re-record the Art. 17 request to the
  broker with a fresh timestamp, then re-run escalation after another 35 days.
- **Cumulative complaints.** Nothing prevents the subject from filing multiple DPA complaints
  about different brokers in parallel. Each one is independent.

## What this guide does NOT cover

- **Pre-litigation posture.** A privacy lawyer should review the complaint before filing if
  the exposure is non-trivial (sensitive data, ongoing harassment, defamation risk).
- **Class-action / collective complaints.** The skill is single-subject. Collective complaints
  under Article 80 (representation) are a different procedural path.
- **Cross-border complaints.** If the subject is in Italy and the broker is in Ireland (often
  the case for Big Tech), the "one-stop-shop" mechanism under Article 56 designates the
  Irish DPC as the lead authority. The skill's generic fallback covers this case but the
  optimal filing strategy is to contact both DPAs.

## Disclaimer

This is not legal advice. The procedures above are grounded in the GDPR text and the
published procedural guidance of each named DPA. Subject-specific strategy (especially
where sensitive data or pre-litigation considerations apply) should be reviewed by a privacy
lawyer in the subject's jurisdiction before filing.
