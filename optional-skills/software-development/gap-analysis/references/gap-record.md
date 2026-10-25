# Gap Record — candidate worksheet, verdict rules, priority rubric

One record per candidate. Fill every field; use `[A DEFINIR]` only for genuinely unknown data. This file is the working evidence trail — keep it inside the run report (`.claude/memory/gap-analysis-{YYYYMMDD}.md`).

## Candidate record

```markdown
### GAP-<category>-<kebab-scope>

- **Key:** GAP-<category>-<kebab-scope>   # stable across runs; never reuse for a different gap
- **Category:** requirements | architecture | implementation | tests | security | observability | documentation | automation | operation
- **AS-IS:** <what exists today> — evidence: <path:Lx-Ly>, <symbol>, <command + output>, <commit>
- **TO-BE:** <what is documented/desired> — source: <.specs/...>, <docs/...>, <ADR>, <rule>
- **Difference:** <observed delta between AS-IS and TO-BE>
- **Impact:** <what breaks, degrades, or is missing if unresolved>
- **Coverage check:** <compiler / linter / test / CI / hook / review / doc / automation that might already cover it — and the evidence that it does or does not>
- **Dedup check:** <terms searched; matching .specs/, Issues, mechanisms — or "none">
- **Verdict:** CONFIRMADO | REJEITADO | DUPLICADO | INCONCLUSIVO
- **Justification:** <why this verdict, citing the evidence above>
```

## Verdict rules

| Verdict | When | Required output |
| --- | --- | --- |
| `CONFIRMADO` | TO-BE documented, AS-IS demonstrably different, no covering mechanism | Gap record + priority card → spec candidate |
| `REJEITADO` | A compiler, analyzer, test, CI step, hook, review gate, or doc already covers it — or the difference is not proven | Cite the covering evidence; no artifact created |
| `DUPLICADO` | An equivalent `.specs/` entry, GitHub Issue, or existing mechanism already tracks it | Link the existing artifact; no artifact created |
| `INCONCLUSIVO` | Sources contradict each other or evidence is insufficient | Question to the user (pt-BR); no artifact created |

**Never** promote a candidate on suspicion alone: a gap requires TO-BE + AS-IS + difference + impact.

## Priority card (CONFIRMADO gaps only)

Qualitative scales — justify each value in one line. Do not fabricate metrics.

| Field | Scale | Justification |
| --- | --- | --- |
| Impact | `high / medium / low` | blast radius if unresolved |
| Urgency | `high / medium / low` | release blocker? user-facing? |
| Risk | `high / medium / low` | security, data loss, regression potential |
| Scope | `file / module / system` | how much of the repo it touches |
| Effort | `S / M / L` | rough size only — never person-hours |
| Confidence | `high / medium / low` | how solid the evidence is |

Sort order for output: `category → impact → urgency → key` (deterministic, stable across runs).

## Decomposition

A gap is too broad when a single SPEC could not cover it (multiple modules, mixed categories, or > ~5 requirements). Split it into child keys `GAP-<category>-<kebab-scope>-<n>` and produce one record each before invoking `write-specs`.
