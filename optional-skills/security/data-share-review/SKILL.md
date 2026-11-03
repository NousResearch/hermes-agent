---
name: data-share-review
description: Use when data leaves the team; minimize, mask, scan for PII.
version: 0.1.0
author: rlaope (github.com/rlaope), Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [privacy, pii, redaction, pseudonymization, de-identification, data-sharing]
    category: security
    related_skills: [security-event-response, nemo-curator]
---

# Data Share Review Skill

Prepare a dataset, log bundle, ticket/CRM export or transcript excerpt before it leaves the team
(vendor, contractor, public issue, model provider, training corpus): classify every field, drop what
the destination does not need, pseudonymize with the key kept local, and call the file redacted only
after a residual scan of the exact file that leaves. It does not decide whether the data may lawfully
leave at all (DPA, consent, cross-border transfer) — that is a question for counsel.

Ported from [rlaope/oh-my-hermes](https://github.com/rlaope/oh-my-hermes) (MIT), `omh-data-share-review`.

## When to Use

- "Send last month's ticket export to the analytics vendor, strip the customer details first."
- "Redact / de-identify / pseudonymize / anonymize this dataset (or these logs) before I share it."
- Attaching a log or a data sample to a public issue, a support case, or a provider's bug tracker.
- Building a training or eval corpus from real user data.

Don't use for: analysing the dataset itself, a secret already committed or pushed
(`security-event-response`), or GPU-scale corpus PII filtering (`nemo-curator`).

## Prerequisites

Python 3 (stdlib only) for `scripts/data_share.py`. It reads `.csv`/`.tsv` by header, `.jsonl`/`.ndjson`
by top-level keys, and anything else as one free-text blob. Every subcommand prints JSON counts and
metadata, never a value from the file.

## How to Run

```bash
python3 ${HERMES_SKILL_DIR}/scripts/data_share.py inventory export.csv
python3 ${HERMES_SKILL_DIR}/scripts/data_share.py pseudonymize export.csv out/share.csv --columns customer_id,email --drop birth_date,phone --key-file ~/share-keys/vendor-2026-10.key --lowercase
python3 ${HERMES_SKILL_DIR}/scripts/data_share.py scan out/share.csv --originals export.csv --columns customer_id,email --quasi postcode,plan --k 5
python3 ${HERMES_SKILL_DIR}/scripts/data_share.py record out/share.csv --destination analytics-vendor
```

Run each through `terminal`. Exit 2 with `{"error": ...}` means the step did not happen.

## Quick Reference

| Subcommand | Prints | Use for |
|---|---|---|
| `inventory FILE` | per column: non-empty, distinct, mean length, free-text flag, shape counts; header shape counts | classifying fields without reading rows |
| `pseudonymize IN OUT --columns a,b --key-file K [--drop c] [--lowercase]` | rows, columns kept/dropped/pseudonymized | HMAC-SHA256 pseudonyms (`p_<20 hex>`); creates a 0600 key if absent; refuses a key inside OUT's folder |
| `scan FILE [--originals SRC --columns a,b] [--quasi x,y --k N]` | totals and per-column counts for email, phone, card (Luhn), IP, secret (column names included); original-value hits; k-anonymity groups; `all_zero` | the residual scan |
| `record FILE --destination D` | destination, date, fields, rows, sha256 | the share record |

A column whose name is itself identifier- or secret-shaped prints as `#<n> (name redacted)`: the
header is data (a pivoted export) and counts like any other cell.

## Procedure

1. **Destination and need.** Ask where the file goes and what the destination needs it for. Done
   when every field the destination asked for is named with its purpose.
2. **Inventory.** Run `inventory`; read the header with `read_file` (small `limit`) only if the
   counts leave a column's class unclear and no column name is redacted. Never page through rows —
   every row read enters the session and the provider. Classify each field as direct identifier, quasi-identifier, secret,
   free text or not personal, using `references/data-share-method.md` §1 (class comes from content,
   not the column name). Done when every field has a class and the evidence for it.
3. **Minimization plan.** Give each field exactly one action — drop, generalize, mask or pseudonymize
   — in the order of §2: drop first, since a masked field that ships still links rows to people.
   Done when every kept field names the destination's need for it.
4. **Transform.** `pseudonymize` direct identifiers that must stay joinable, with a new key per
   destination stored outside the shipped folder (never committed, attached or uploaded). Generalize
   quasi-identifiers (date → month/year, postcode → region, age → band) with a short script via
   `terminal` or `execute_code`. Done when the output file exists and the key path is recorded.
5. **Residual scan.** Run `scan` on the exact file that leaves, after the last transform, with
   `--originals` for every dropped or pseudonymized identifier column and `--quasi` for the kept
   quasi-identifiers. Free text counts as unscanned until this step counts it. Done when the JSON
   shows a count for every class.
6. **Verdict.** `redacted` only when every class counts zero or each remaining hit is named and
   accepted by the user; otherwise `not redacted` with the counts; `not_observed` if no scan ran.
   Fix hits (drop the free-text column, re-mask) and re-scan; never ship on an explained-but-unaccepted hit.
7. **Share record.** Run `record` and give the user the metadata: destination, date, fields, row
   count, sha256, scan result. No value from the file, no key.

## Pitfalls

- **Hermes redaction is not this scan.** `privacy.redact_pii` (off by default) hashes user IDs and
  strips phone numbers in the gateway's LLM context on WhatsApp, Signal, Telegram and BlueBubbles;
  `security.redact_secrets` (on by default) masks secret-shaped strings in tool output. Neither
  changes a file on disk or a file that leaves.
- **A plain hash is reversible** by hashing candidate emails or phone numbers. Use the keyed HMAC.
- **Never put an original value in a `search_files` pattern or a command line** — that copies it
  into the session. `--originals` reads the source column itself and prints one count.
- **Pattern counts over-match.** Long order numbers can count as phones, and `--originals` matches
  every source value as a whole word, so a short one (`Al`, `7`) also counts unrelated text; that is
  a reason to inspect per-column counts, not to skip the class. Names and street addresses have no
  reliable shape: only `--originals` and dropping the free-text column catch them.
- **The key file is the mapping.** Losing it breaks joinability; leaking it reverses every pseudonym.
- A secret found in the data is also a leak to contain — hand it to `security-event-response`.

## Verification

- `scan` on the shipped file prints `"all_zero": true`, or every non-zero count is accepted by name.
- The sha256 in the share record equals the sha256 the scan reported (same file, no later edit).
- The key file is outside the shipped folder: `pseudonymize` refuses otherwise.
