# Report Contract

## Allowed classifications

- `READY TO EXPORT`
- `READY TO IMPORT`
- `BLOCKED`
- `COMPLETE WITH WARNINGS`

## Required headings

- Hermes Environment Migration
- Phase Verdict
- Source Inventory
- Target Inventory
- Classification
- Integrity Evidence
- Import Plan
- Secrets Plan
- Path Rewrites
- Blocked Items
- Rollback Plan
- Verification

Do not invent an intermediate classification. Put nuance in confidence, warnings, blockers, and Not Verified.

Do not invent inventory details. Every named profile, skill, path, version, provider, job, gateway, plugin, archive member, or dependency must be present in the evidence. When only a count or category is known, report only that count or category and append `(names not provided)` or the equivalent.

Treat evidence labels narrowly. A stated `clean SHA-256 manifest` supports only the stated hash result. Do not infer manifest scope, missing-file checks, orphan-file checks, signatures, provenance, archive safety, or tamper analysis unless each was explicitly verified.

Use the required headings exactly as level-2 headings and in the listed order. Do not rename, number, omit, or insert substitute level-2 sections.
