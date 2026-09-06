<!--
Copy to: logs/ledger/audits/AUDIT-YYYY-MM-DD-NNN-<kebab-slug>.md
Then add a row to logs/ledger/INDEX.md.
Naming + versioning rules: logs/ledger/README.md
-->

# AUDIT-YYYY-MM-DD-NNN — <Title>

| | |
| --- | --- |
| **Audit ID** | AUDIT-YYYY-MM-DD-NNN |
| **Date** | YYYY-MM-DD (America/Los_Angeles) |
| **Author** | <name / Claude session URL> |
| **Scope** | <what was and was not examined> |
| **Upstream base** | hermes@<sha> (<N> behind upstream/main) |
| **North-Forge version** | NF-vX.Y.Z |
| **Ledger schema** | v1 |
| **Supersedes** | <prior AUDIT id, or —> |
| **Method** | <commands / tools used> |

## 1. Summary

<3–6 sentences: overall state, the headline findings, whether action is required.>

## 2. Repository state

- Identity / provenance:
- Fork vs upstream (`git rev-list --left-right --count upstream/main...origin/main`):
- Local checkout vs `origin/main`:
- Working tree (`git status --porcelain`):

## 3. Findings

Severity: CRITICAL · HIGH · MEDIUM · LOW · INFO.

### F-01 — <SEVERITY> — <short title>
- **Observed:** <fact + evidence: paths, command output>
- **Impact:** <why it matters>
- **Status:** RESOLVED (this session) · OPEN · INFO / no action
- **Linked:** CHG-… / ERR-…

### F-02 — …

## 4. Remediation performed this session

- CHG-… — <what was done>

## 5. Open items / recommendations

- ERR-… — <what remains, who must act>

## 6. Notes

<README review, upstream-merge risks, follow-up audit triggers, etc.>

## Appendix — raw data

```
<key command output>
```
