# Kelsa Support Team — Contacts & Roles (NDR's Kelsa issues channel)

NDR reports Kelsa bugs/issues via the **Kelsa dev/support WhatsApp group** (group-paste pattern,
no pre-set number). Messages are addressed by first name to whoever he names (usually Ashwin,
with Pawan looped in to confirm/restore). Do NOT resolve the group to a single person.

## Who's who (from find_contact.py "Kelsa" — sheet row numbers 2026-09-09)

| Person | Role | Phone | Email |
|---|---|---|---|
| **Ashwin Hegde** | Co-Founder, Kelsa.io (O3 Infotech) — architecture/UI/support decisions, "should be resolved" fixes | +91 99728 03083 | ashwin.hegde12@gmail.com |
| **Pavan Kumar O3** | O3 Infotech tech — Excel imports, audit/timestamp checks, data fixes | +91 98496 44710 (O3 no: +91 90350 16100) | pavan@kelsa.io |
| **Guna Kelsa** | Process/support — business-process clarifications, SPOC checks | +91 99723 12600 | — |
| Ajay Haridas | Customer Advocacy Executive | +91 94493 38566 | ajay@kelsa.io |
| Arnav Singh | Kelsa | +91 80500 75520 | arnav@kelsa.io |
| Apoorv Gupta | O3 Infotech | +91 99687 87240 | apoorv@kelsa.io |
| Shilpam Rastogi Jain | Kelsa | +91 78290 77400 | shilpam@kelsa.io |
| Kuntal Kumar | O3 Infotech | +91 79082 95742 | kuntal@kelsa.io |
| Vikramaditya H | Customer Success Advocate | +91 77605 03837 | vikramaditya@kelsa.io |
| Rupsa Das | CSA | +91 62905 80625 (Wapp 83359 18795) | rupsa@kelsa.io |

Other Kelsa/O3 contacts: Anuj Kelsa +91 88056 46679, Kiran Kelsa +91 97390 23876,
Saurabh Kelsa +91 98102 95216, Sumukha Kelsa +91 63600 01783, Prajjwal Kelsa O3 +91 90791 23050,
Chakresh Jha O3 Kelsa +91 82407 54900, Rahul Mishra Kelsa +91 93043 95338, Sailesh Arya Kelsa +91 74160 11450.

## Division of labour observed (Sep 2026)

- **Ashwin** answers support pings directly ("Should be resolved" after UI change; "Not related to UI changes — please file a ticket"; diagnoses public-form vs audit issues; fixes forward-looking timestamp handling).
- **Pavan** confirms technical root causes from data/audits ("as per the audit it is excel import sir" — 7 Sep Excel import rewrote lead timestamps).
- **Guna** handles business-process side (public-form usage policy, SPOC checks, "will check").

## Lookup rule

- `contact_resolver` for anything Kelsa → FALSE POSITIVES (single-char "K", unrelated Ashwin Bafna). Skip it.
- Use `scripts/find_contact.py "Kelsa"` — returns the whole cluster in one shot.