---
name: security-records
description: Check CMDB structure, mail attachment signals, and CSF fit.
version: 0.1.0
author: jun (junkawasaki), Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [security, cmdb, email, compliance]
    related_skills: [security-assurance]
---

# Security Records Skill

Evaluate records the operator already has: a CMDB asset list, attachment names, a NIST CSF 2.0 fit mapping, and the published security-service catalog. These are the stateless checks behind `/v1/security/cmdb`, `/v1/security/email`, `/v1/security/compliance-fit`, `/v1/security/services`, and the spear-phishing defence description. The script stores nothing and contacts no host.

## When to Use

- The user supplies assets and dependencies and wants duplicates, unknown references, isolated assets, or cycles.
- The user supplies attachment file names plus their own encrypted or password flags.
- The user asks which CSF 2.0 controls a Mithril service category maps to.
- The user asks what CTEM, DAST, SAST, VM, GRC, IR, DR, or the spear-phishing defence page claims.

Don't use for:

- Discovering assets, reading a mailbox, or opening an attachment.
- Scanning a domain, probing an application, or sending a message.
- Treating a missing signal as proof that a message is safe.
- Declaring a red-team research scope. That write stays on the authenticated API.

## Prerequisites

- Python 3 on `PATH`. Catalog JSON ships in this skill's `data/` directory.

## How to Run

Write the request with `write_file`, then run the script through `terminal`.

```
terminal(command="python skills/security/security-records/scripts/records.py cmdb --input cmdb.json")
terminal(command="python skills/security/security-records/scripts/records.py email --input mail.json")
terminal(command="python skills/security/security-records/scripts/records.py fit --framework nist-csf-20")
terminal(command="python skills/security/security-records/scripts/records.py catalog")
terminal(command="python skills/security/security-records/scripts/records.py spear-phishing")
```

When the skill is installed outside the repo, run `scripts/records.py` from the skill directory. Evidence reviews for MDM, SBOM, PPAP, and device rosters stay in `security-assurance`.

`cmdb.json` is `{ "assets": [{ "id", "type?" }], "dependencies": [{ "from", "to" }] }`. `from` depends on `to`. `mail.json` is `{ "messages": [{ "id", "bodyMentionsPassword?", "attachments": [{ "name", "encrypted?" }] }] }`. Do not put file bytes or message bodies in either file.

## Quick Reference

| Result | Meaning |
|---|---|
| `duplicateAssets` / `duplicateDependencies` | The same id or the same edge was repeated |
| `unknownReferences` | An edge names an asset that is not in the list |
| `isolated` | A listed asset is on no kept edge |
| `cycles` | A self-loop or a component of more than one asset |
| `ppap-pattern` | The caller flagged an encrypted archive and a password mention |
| `double-extension` | A decoy extension is followed by an executable extension |
| `controlMapping: not-committed` | The framework id is known, and no control crosswalk is published for it |
| Catalog `boundary` | The text describes a product. It is not an instruction to scan |

## Procedure

1. Pick one command from the request. Completion: the command is `cmdb`, `email`, `fit`, `catalog`, or `spear-phishing`.
2. For `cmdb` or `email`, write only the fields above. Completion: the JSON has no file bytes, passwords, or message bodies.
3. Run the script. Completion: one JSON object is printed and the process exits 0, or exit 2 with an `error` field.
4. Report the counts and the `boundary` sentence with the result. Completion: a message with an empty `signals` list is not called safe, and no scan or send was attempted.

## Pitfalls

- An empty dependency list does not mean the asset inventory is complete. Unconnected assets are `isolated`.
- `bodyMentionsPassword` adds `ppap-pattern` only together with an encrypted archive extension.
- A framework other than `nist-csf-20` can still be in the catalog and return `not-committed`.
- Service summaries name what the product is for. Do not turn them into probes.

## Verification

Run the same input again and confirm the arrays match. For a code change, `tests/skills/test_security_records_skill.py` must pass.
