# GDPR / UK-GDPR (Art. 17 erasure + Art. 21 objection + Charter Art. 8)

For EU/EEA/UK subjects. Citations ground every request the skill sends.

## Rights invoked

The skill files under three EU Charter / GDPR articles that compose a complete legal basis:

- **Article 17 GDPR — Right to erasure** ("right to be forgotten"). The subject's primary lever.
  Erasure must be carried out without undue delay and in any event within one month (Article 12(3)).
- **Article 21 GDPR — Right to object**. Sends in parallel with Art. 17. Forces the broker to
  demonstrate "compelling legitimate grounds that override" the subject's interests — a bar
  commercial data-broking of named individuals rarely clears (see EDPB Guidelines on legitimate
  interest, three-part test).
- **Article 8, Charter of Fundamental Rights of the European Union — Protection of personal
  data**. The constitutional anchor. Quoting it in the request signals to the broker's legal
  team that this is not a routine suppression request — it is a fundamental-rights claim.

For UK residents post-Brexit, the same rights are retained verbatim under the **UK GDPR**
(Data Protection Act 2018, retained EU law). The Article 12(3) deadline, the Article 83 fine
ceiling, and the Article 77 complaint procedure all apply via the ICO.

## The cite ladder (what each template invokes)

| Template                       | Art. 17 | Art. 21 | Charter Art. 8 | Art. 12(3) | Art. 77 | Article 83 |
|--------------------------------|:-------:|:-------:|:--------------:|:----------:|:-------:|:----------:|
| `emails/gdpr-erasure.txt`      | ✓       | ✓       | ✓              | ✓          | (mention) | (mention) |
| `emails/gdpr-art21-only.txt`   | (fallback) | ✓    | ✓              | ✓          | (mention) | (mention) |
| `emails/gdpr-indirect-deletion.txt` | ✓ | ✓       | ✓              | ✓          | (mention) | (mention) |
| `dpa-complaints/garante.txt`   | ✓ (cited as prior request) | — | ✓             | ✓          | ✓        | ✓          |
| `dpa-complaints/cnil.txt`      | ✓ (cited as prior request) | — | ✓             | ✓          | ✓        | ✓          |
| `dpa-complaints/bfdi.txt`      | ✓ (cited as prior request) | — | ✓             | ✓          | ✓        | ✓          |
| `dpa-complaints/ico.txt`       | ✓ (cited as prior request) | — | ✓             | ✓          | ✓        | ✓          |
| `dpa-complaints/generic.txt`   | ✓ (cited as prior request) | — | ✓             | ✓          | ✓        | ✓          |

The Art. 17 / 21 request templates lead with the rights and ask for erasure. The Art. 77 DPA
complaint templates cite those as the prior request and ask the supervisory authority to enforce.

## When to use which request kind

The skill picks via `legal.render_request(kind, broker, fields)`:

- **`gdpr`** (default) — the Art. 17 + 21 combined request, for any broker holding the subject's
  personal data with a privacy contact addressable by email or web form.
- **`gdpr_art21_only`** — for brokers where the legitimate-interest claim is strong (public
  records, phone directories with strong publisher-side grounds). Leads with Art. 21 objection
  and only mentions erasure as a fallback.
- **`gdpr_indirect`** — for cases where the subject's PII sits on a third party's record (the
  "subject appears on someone else's profile" case). Mirrors the CCPA `ccpa_indirect` shape.
- **`generic`** — fallback when the broker's `optout.deletion.kinds` does not include `gdpr`
  (rare; the `request_kind()` dispatcher in `autopilot.py` enforces this restriction).

## Per-broker routing

The `brokers.gdpr_scope()` filter returns the set of brokers an EU subject can reasonably
expect to honour Art. 17 (US brokers with a track record + all EU-native brokers under
`references/brokers/eu/`). Brokers with `gdpr_scope: false` are skipped from the DPA-escalation
planner — the subject still files an Art. 17 against them (and the broker may still honour it),
but `pdd.py next` will not surface escalation actions for brokers unlikely to respond to a DPA.

Every curated US broker record now carries a `gdpr_scope` boolean and `jurisdictions` includes
`"EU"` alongside `"US"`. This means an EU subject sees their native brokers (Pagine Bianche,
118000, etc.) AND the US brokers with documented Art. 17 paths (Spokeo, Whitepages,
etc.) in their `pdd.py plan` output.

## Request content

The rendered email always contains:

1. The Article(s) being invoked (17 / 21 / Charter Art. 8).
2. The data subject's name + contact email + (where required) address — **least-disclosure**;
   SSN and ID numbers are never volunteered.
3. The listing URL(s) the subject's data appears at.
4. A clear statement of what is required: written confirmation of erasure within 30 days,
   confirmation that data will not be re-acquired and re-processed, and (where applicable)
   disclosure of any third parties the data has been shared with (Article 15(1)(c)).
5. An escalation notice: if no satisfactory response is received within 30 days, the subject
   will lodge a complaint with their national supervisory authority under Article 77.

## Statutory deadlines

- **Article 12(3) GDPR**: controller must respond within **one month** of receipt, with an
  optional **two-month extension** for complex requests (the broker must notify the
  extension within the first month). Total possible window: **30 to 90 days**.
- **Article 78(2)**: supervisory authority must act on a complaint within a reasonable time,
  typically 3 months.
- **Article 83(5)**: administrative fines up to **EUR 20 million or 4% of total worldwide
  annual turnover**, whichever is higher, for infringements of the basic principles
  (including Art. 5, 6, 7, 9, 22, 44, 45, 46).

## The Article 77 escalation path (DPA complaints)

When an Art. 17 request is filed and the broker does not respond within the 30-day Article 12(3)
window, the subject can file an Article 77 complaint with their national supervisory authority.
This is the legal step CCPA has no equivalent for — and the killer feature of the EU pipeline.

`pdd.py escalate <subject> <broker>` renders a complaint package for the subject's national DPA.
Custom DPA-specific templates ship for every registered adapter: localized/native templates
where already represented, plus authority-specific English packages for the remaining EU/EEA
adapters.
The complaint cites the prior Art. 17 request date and asks the DPA to enforce under Article 58(2).

`pdd.py next` surfaces this **automatically** 35+ days after an Art. 17 was filed with no
response (35 = 30-day Art. 12(3) clock + 5-day grace for time-zone and clock-skew). The
subject reviews the rendered draft, attaches the prior Art. 17 evidence + a copy of their ID,
files with the DPA via PEC (Italy), national web form, email, or post as advertised by the
adapter, then runs
`pdd.py escalate <subject> <broker> --file` to stop the autonomous queue from re-surfacing.

See `references/legal/dpa-escalation.md` for the full per-DPA procedure.

## Notes on specific jurisdictions

- **Germany (federal)**: BfDI handles federal bodies and telecom. Most data-broker complaints
  against private-sector controllers go to the **Land** DPA of the subject's residence
  (16 Land DPAs). The BfDI adapter ships a German complaint that notes this and asks the
  authority to forward to the correct Land DPA.
- **Italy**: filing via **PEC** (protocollo@pec.gpdp.it) is the strongest paper trail and
  preferred by the Garante for any complaint that may later escalate to court. The web form
  is the no-PEC fallback.
- **France (CNIL)**: CNIL is one of the most enforcement-active DPAs in the EU (Clearview AI
  fine, etc.). Their complaint procedure is well-documented and accessible.
- **All named EU member states plus Norway, Iceland, and Liechtenstein**: shipped adapters route
  `pdd.py next` to the national filing channel instead of the generic EDPB member-list fallback.
  Run `pdd.py dpas --residency <code>` to inspect the exact portal.
- **UK (ICO)**: ICO retains Article 17 / Article 21 rights verbatim under the UK GDPR.
  Post-Brexit case law: the April 2024 Upper Tribunal ruling in the Experian case narrowed some
  of ICO's powers; the current ICO guidance reflects the post-ruling position.

## What is out of scope (call out in PR body)

- Brazil (LGPD), Switzerland (nDSG/revFADP), and other non-EU/EEA/UK jurisdictions.
- Automated web-form submission of Garante/CNIL/ICO/BfDI complaint forms (first version
  renders + saves the complaint text; browser-form automation is a follow-up).
- Exact reproduction of every DPA's official PDF/form fields. The shipped templates are
  DPA-specific complaint packages intended for the portals; browser-form automation and
  per-field official form filling are follow-up work.
- Article 20 (data portability) and Article 15 (right of access) request templates — same
  engine, follow-up work.

## Disclaimer

This is not legal advice. The cite ladder is grounded in the published text of the GDPR,
the EDPB Guidelines, and the EU Charter. A subject with non-trivial exposure (sensitive
data, ongoing harassment, pre-litigation posture) should consult a privacy lawyer in their
jurisdiction before filing.
