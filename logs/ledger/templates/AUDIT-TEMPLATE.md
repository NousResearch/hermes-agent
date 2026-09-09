<!--
Copy to: logs/ledger/audits/AUDIT-YYYY-MM-DD-NNN-<kebab-slug>.md
Then add a row to logs/ledger/INDEX.md.
Naming + versioning rules: logs/ledger/README.md

Write section 1 "Since last handoff" FIRST, before the rest of the body — it
frames everything else. Every finding needs a Confidence tag (Confirmed Fact /
Field-Reasoned / Unverified — see README). Set the header Run: id before starting.
-->

# AUDIT-YYYY-MM-DD-NNN — <Title>

| | |
| --- | --- |
| **Audit ID** | AUDIT-YYYY-MM-DD-NNN |
| **Date** | YYYY-MM-DD (America/Los_Angeles) |
| **Run** | RUN-YYYY-MM-DD-NNN |
| **Author** | <name / Claude session URL> |
| **Scope** | <what was and was not examined> |
| **Upstream base** | hermes@<sha> (<N> behind upstream/main) |
| **North-Forge version** | NF-vX.Y.Z |
| **Ledger schema** | v2 |
| **Supersedes** | <prior AUDIT id, or —> |
| **Method** | <commands / tools used> |

## 1. Since last handoff

Diff this audit against the previous one's open items. Not optional — write it first.

| | Items |
| --- | --- |
| **Closed since <prev AUDIT id>** | <ERR-/DECISION- ids resolved or decided, + the CHG- that did it. Or "none".> |
| **New since <prev AUDIT id>** | <ERR-/DECISION- ids opened since. Or "none".> |
| **Unchanged / still open** | <ERR-/DECISION- ids carried forward untouched. Or "none".> |

<One or two sentences: is the trend toward closure or accumulation? Anything that
has been "still open" across multiple audits and needs a push?>

## 2. Summary

<3–6 sentences: overall state, the headline findings, whether action is required.>

## 3. Repository state

- Identity / provenance:
- Fork vs upstream (`git rev-list --left-right --count upstream/main...origin/main`):
- Local checkout vs `origin/main`:
- Working tree (`git status --porcelain`):

## 4. Findings

Severity: CRITICAL · HIGH · MEDIUM · LOW · INFO.
Confidence (required, one of): Confirmed Fact · Field-Reasoned · Unverified.

### F-01 — <SEVERITY> — <short title>
- **Observed:** <fact + evidence: paths, command output>
- **Confidence:** <Confirmed Fact | Field-Reasoned | Unverified> — <why this tag>
- **Impact:** <why it matters>
- **Status:** RESOLVED (this session) · OPEN · INFO / no action
- **Linked:** CHG-… / ERR-… / DECISION-…

### F-02 — …

## 5. Remediation performed this session

- CHG-… — <what was done>

## 6. Open items / recommendations

- ERR-… — <fault that remains, who must act>
- DECISION-… — <choice that remains open, who decides>

## 7. Notes

<README review, upstream-merge risks, follow-up audit triggers, etc.>

## Appendix — raw data

```
<key command output>
```
