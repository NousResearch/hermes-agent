# Agent Work Loop — Evidence Model (Hermes adaptation)

The judging model behind `better-harness`. Five dimensions × three checks =
fifteen stable review identities. This file owns check definitions, evidence
states, scoring ceilings, and Learning Capture rules. Evidence collection
belongs to `evidence-passes.md`; reader-facing gates belong to
`quality-gates.md`.

## Review unit: Task Episode

One user goal with one observable acceptance boundary. Merge refinements of the
same goal; **split unrelated goals even when they share a session**. Never turn
aggregate counts (tool calls, file reads, session length) into task behavior.

Close a changed Episode only when its final validation set is linked to its
change set. Treat `failure → edit → pass` as a repair candidate until evidence
also retains reproduction and diagnosis.

## Evidence states (not pass/fail)

| State | Meaning |
| --- | --- |
| `Present` | An owned mechanism or review contract exists |
| `Wired` | The relevant task, trigger, or owner route can reach it |
| `Exercised` | A linked episode or inspection used it and retained a result |
| `Outcome-supported` | A comparable later result supports the claimed effect |
| `Missing` | Inspected evidence confirms a required mechanism/result is absent |
| `Unobserved` | The available observation boundary cannot decide |
| `Not applicable` | Inspected task+project evidence proves it does not apply |

An exercised operation may expose a defect; a safe denial may be correct
behavior; an unavailable external boundary is `Unobserved`, not automatically
`Missing`. **Static configuration proves at most the mechanism it contains.**

## Score ceilings (absolute, per dimension)

| Highest supported evidence | Absolute score ceiling |
| --- | --- |
| `Missing` / `Unobserved` / `Not applicable` | 59 |
| `Present` | 74 |
| `Wired` | 84 |
| `Exercised` | 94 |
| `Outcome-supported` | 100 |

Ceilings, not a formula. A required or triggered check that is missing,
unresolved, or blocked keeps its dimension at 59 or lower. Scores above 75
additionally require inspected source/test ownership plus an executed or
explicitly provided validation route. Score each dimension independently; never
derive a score from finding counts. **A score never creates or suppresses a
finding.**

## The five dimensions and fifteen checks

### 1. Task Understanding — does the agent know the goal and what "done" means?

| Check | Question it answers |
| --- | --- |
| Intent and Acceptance (`goal-understanding`) | Task preserves intended outcome, definition of done, exclusions, corrections, and unresolved questions as one recoverable acceptance boundary |
| Relevant Context (`relevant-context`) | Decisions use the authoritative instructions, architecture/domain owner, canonical source, and material dependent contracts instead of broad or incidental context |
| Scope Boundary (`scope-boundary`) | Intended files, modules, generated artifacts, visible/external effects, risk, exclusions, and approved expansions remain explicit and traceable |

Typical finding: "The task has no recoverable acceptance boundary" (emit only
when conflicting or missing completion criteria materially affected the
result); "The change bypasses the canonical owner"; "The repair silently
expands beyond the requested scope."

### 2. Controlled Execution — is work on supported, repeatable paths?

| Check | Question it answers |
| --- | --- |
| Reproducible Startup (`instruction-led-start`) | A clean or declared starting state becomes usable through the project-owned, non-interactive setup/startup route |
| Supported Operation (`supported-operation`) | Target behavior is discoverable and invocable through a supported command/skill/CLI route with usable inputs, outputs, failures, cleanup |
| Permission Boundary (`permission-boundary`) | Filesystem, network, tools, credentials, external writes, shared state, protected actions stay inside enforced boundaries |

Typical finding: "The documented startup route cannot reproduce a usable
workspace"; "The external tool has access but no supported task workflow";
"The task can cross a protected boundary without a valid decision."

### 3. Change Validation — is there evidence the change actually works?

| Check | Question it answers |
| --- | --- |
| Relevant Verification (`relevant-check`) | The final material change is mapped to and exercised by the smallest project-owned check covering its behavior, invariant, risk |
| Failure Diagnosis and Repair (`failure-repair`) | An observed failure is reproduced, localized with attributable diagnostics, explained by a causal hypothesis, repaired at the smallest correct owner |
| Post-repair Revalidation (`validate-again`) | The same failed check (or justified equivalent with same scope) runs again on the repaired final state |

Require the ordered chain `failure → reproduction → diagnosis → bounded
repair`. A retry pass without diagnosis is not repair evidence. Valid branches:
no relevant failure → Relevant Verification passes, repair/revalidation are
N/A; relevant failure → all three checks must pass on the repaired final state.

Typical finding: "The final change has no check that exercises its affected
behavior"; "The failed check was retried without locating the cause"; "The
repaired state was never checked at the original scope."

### 4. Reliable Delivery — does AI speed bypass quality checks or acceptance?

| Check | Question it answers |
| --- | --- |
| Delivery Acceptance (`acceptance-evidence`) | The current result reaches the real review/CI/merge/release boundary with revision-bound decision evidence |
| High-risk Approval (`high-risk-approval`) | Every destructive, privileged, external, irreversible, shared-state, credential, release, or production action gets its required decision before the effect |
| Rollback or Recovery (`rollback-recovery`) | The actual side effect has a risk-proportionate rollback/restore/retry/compensation/safe-abort path with an owned postcondition |

Local tests and an agent "done" message belong to Change Validation, not
delivery. If the external boundary cannot be opened, use `Unobserved` rather
than inventing a missing PR or deployment. Delivery Acceptance is required;
approval and recovery are conditional on inspected risk.

### 5. Learning Capture — does the next task benefit from this one?

| Check | Question it answers |
| --- | --- |
| Lifecycle Opportunity Detection (`lifecycle-repeat-detection`) | A bounded review distinguishes current capability gaps, supported repeated opportunities, entropy-backed maintenance opportunities, adequate clean windows, inadequate evidence |
| Loop Engineering (`loop-engineering`) | A supported opportunity routes through coverage inspection (observed → built-in → configured → extend → create) into the smallest durable owner with a repeatable trigger, artifact, verifier, state, safety boundary, stop rule |
| Longitudinal Validation (`later-validation`) | A reusable improvement stays accountable through a comparable later-outcome evaluation or a recurring maintenance/freshness inspection against canonical truth |

## Learning Capture special rules

- **One Agent-authored score (35–100)**, not three subscores. `null` belongs to
  an unresolved review slot; never project zero. The 35 floor only means a
  bounded review was completed — it awards no points for findings, states,
  counts, or configured mechanisms.
- **Memory/Skill value chain** (positive evidence only with the full chain):
  `exists → retrieved → relevant → applied → later outcome improved`, without
  guardrail regression. Count, absence, installation, or configuration earns no
  credit.
- **Opportunity classes:** current capability gap (one handoff, current-
  dimension finding only) · repeated opportunity (≥2 distinct comparable Task
  Episodes + repeated friction or successful procedure) · entropy-backed
  opportunity (named asset/invariant + stable trigger + canonical truth +
  runnable inspection returning clean/gap/needs-more-evidence) · adequate
  no-candidate window (2 eligible Episodes + explicit clean decision).
- **Ceilings:** uncovered applicable demand ≤ 59; wired owner ≤ 84; current-task
  exercise without later comparison ≤ 74; later exercised comparison ≤ 94; only
  a later comparable **improved** outcome permits 100.
- File age, churn, unresolved markers, and counts are inspection triggers, not
  drift. Confirm drift against an opened mismatch or executed result before
  making it a finding.
- A confirmed Memory/Skill integrity problem is a pending Asset Health finding;
  its repair can advance Repair Progress but never raise any dimension score in
  the same observation window.

## Project Harness lens (feedforward + feedback)

Two complementary views of the project's agent-readiness:

- **Feedforward guides** steer the agent before it acts: AGENTS.md/Rules, specs,
  skills, acceptance criteria.
- **Feedback sensors** observe results and enable self-correction: linters,
  tests, hooks, evaluation agents.

Judge these five capabilities (do not emit a second score set; map observations
into the five dimensions above): **Context Map** (can an agent reach the right
context/boundary/risk/next step?) · **Environment Readiness** (versioned,
bounded, diagnosable setup without guessing?) · **Fast Feedback** (affected
checks return timely, actionable evidence?) · **Quality Gates** (rules
mechanically enforced and repairable?) · **Change Safety** (changes bounded,
accepted through evidence, recoverable?).

Strong harness evidence is executable, repeatable, observable, auditable, and
portable across agents. "Many docs" or "many CI files" is not strength unless
the artifacts connect to actionable feedback and mechanical guardrails.

## Reference ownership

| Concern | Owner |
| --- | --- |
| Evidence collection, per-lane boundaries | `evidence-passes.md` |
| Finding eligibility, privacy, repair prompts | `quality-gates.md` |
| Report structure and support tracks | `SKILL.md` Steps 4–5, `templates/harness-report.md` |
| Definition/research rationale | [QoderAI/better-harness models/agent-work-loop.md](https://github.com/QoderAI/better-harness/blob/main/models/agent-work-loop.md) |
