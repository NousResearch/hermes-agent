# Comparative Analysis: `startup-skill` vs `cyber-vc-analyst`

Date: 2026-07-03

## Scope

This note compares the external repository
[`ferdinandobons/startup-skill`](https://github.com/ferdinandobons/startup-skill)
against the current local Hermes skill setup centered on
`skills/research/cyber-vc-analyst/`.

The goal is not to copy its surface area blindly. The goal is to identify
where that repo is structurally stronger, where the current cyber workflow is
already ahead, and what should be productized next.

## Current Local Baseline

The local `cyber-vc-analyst` setup already has several strengths that are more
advanced than a generic startup-research skill:

- strong domain specialization for early-stage cybersecurity investing
- explicit company, theme, compare, and triage modes
- private-vault-first workflow for founder and company context
- Return on Security MCP integration model
- Slack-oriented invocation and follow-up flow
- schemas for company, theme, compare, triage, and evidence-ledger outputs
- fixtures covering the main runtime-facing paths
- release-note scaffolding and roadmap documentation
- company output rooted at `4.Resources/Companies`
- theme output rooted at `3.Areas/cyber futures frontier`

In short: the current skill is already better aligned to the actual investment
workflow and data sources you use.

## What The External Repo Does Better

The external `startup-skill` repo is stronger in workflow packaging and
operational structure.

### 1. It is a suite, not a single primary skill

The repo is organized as a set of adjacent skills:

- `startup-design`
- `startup-competitors`
- `startup-positioning`
- `startup-pitch`

This creates a clearer mental model for users. Each skill owns a narrower job,
and together they form a repeatable system.

### 2. Its main workflow is phased and resumable

The `startup-design` skill is built around a multi-phase process with explicit
progress persistence via `PROGRESS.md`. That is operationally useful because
long research runs can be resumed rather than restarted.

This is better than a purely prompt-driven model when a workflow spans multiple
rounds, requires tool usage, or depends on incremental research quality.

### 3. It has stronger artifact choreography

The repo appears designed around output files that feed later stages. The
`startup-competitors` skill explicitly reuses artifacts from prior work where
present.

That pattern is valuable because it turns one-off responses into a compounding
research system.

### 4. It encodes research discipline directly into the skill

The competitor workflow includes ideas that are worth carrying over:

- research-depth tiering
- sequential research waves
- explicit verification pass
- honesty protocol for uncertainty

These are good controls for reducing shallow synthesis and false confidence.

### 5. It presents as a product, not just a prompt

The repo has a clearer packaging layer:

- install/update options
- modular README framing
- release/changelog orientation
- direct invocation conventions

This is mostly product discipline, not model quality, but it materially affects
adoption and maintainability.

## Where The Current Cyber Skill Is Stronger

The local `cyber-vc-analyst` setup is materially stronger in areas that matter
for your actual use case.

### 1. The domain model is much deeper

The current skill is not generic startup analysis. It is already structured
around cybersecurity venture analysis, taxonomy mapping, thematic analysis, and
investment-committee style outputs.

That specialization is a major advantage and should not be diluted.

### 2. The evidence model is better

The current skill explicitly separates:

- private vault context
- ROS MCP market intelligence
- public evidence

That is stronger than a generic startup research flow because it matches how
real investment judgment gets formed and audited.

### 3. The output contract is more durable

The current skill already has schemas, fixtures, release notes, and
Slack-runtime behavior. That gives it a better path toward regression
discipline than most prompt-only repositories.

### 4. It is closer to a real operating workflow

The current setup already spans:

- Slack invocation
- vault write-back
- comparison workflows
- thematic memos
- triage mode

That is closer to a private investment operating system than the external repo,
which is broader but less tightly coupled to your actual environment.

## Gap Analysis

The main gap is not analytical quality. The main gap is workflow architecture.

The external repo suggests that the current cyber setup should evolve from
"one broad skill with multiple modes" toward "a small suite of related skills
sharing common schemas, fixtures, and references."

The second gap is resumability. The current cyber workflow is good at producing
finished memos, but it is not yet designed as an explicitly checkpointed
research program.

The third gap is quality gates. The current skill has structure, but it would
benefit from hard-coded verification and uncertainty passes before final output.

## Recommended Enhancements

### High-value near-term changes

1. Split the workflow into a skill family.

Recommended shape:

- `cyber-vc-company`
- `cyber-vc-theme`
- `cyber-vc-compare`
- `cyber-vc-triage`
- `cyber-vc-competitors`

Keep `cyber-vc-analyst` as the top-level orchestrator or compatibility entry
point, but let the narrower skills own their specific workflows.

2. Add progress persistence.

Introduce a lightweight progress artifact for longer runs, for example:

- `research-state/<slug>.md`
- status by phase
- evidence gathered
- unanswered questions
- next recommended step

This matters most for theme memos, competitive landscapes, and deeper company
workups.

3. Add a mandatory verification pass before final output.

Before finalizing any memo, require a short internal check that confirms:

- which claims are direct facts
- which are ROS-derived market intelligence
- which are inferred
- which confidence ratings are weak due to missing data

4. Add research-depth modes.

Recommended levels:

- `quick`
- `standard`
- `deep`

This is especially useful in Slack, where a user may want either a fast answer
or a more expensive memo run.

5. Add cross-artifact reuse.

Theme runs should automatically look for:

- existing company memos
- compare notes
- prior triage notes
- market-theme notes

Company runs should likewise reuse prior theme memos where relevant.

### Medium-term repo/productization changes

1. Create a clearer private workflow repo structure.

Recommended top-level shape:

- `skills/`
- `schemas/`
- `fixtures/`
- `prompts/`
- `docs/`
- `release-notes/`
- `examples/`

2. Separate runtime skill docs from authoring docs.

Keep Hermes-facing `SKILL.md` concise and operational.
Move design rationale, schema decisions, and workflow guidance into `docs/`.

3. Add fixture classes for degraded-source scenarios.

Useful fixture types:

- vault + ROS + public evidence
- vault + ROS only
- vault only
- ROS unavailable
- sparse company evidence

4. Add a reusable comparison template.

The compare path is valuable enough to deserve its own stronger artifact model:

- normalized comparison table
- winner-by-dimension summary
- adjacency vs direct competition classification
- investment preference rationale

5. Add semantic versioning discipline across the whole skill family.

The external repo signals product packaging well. The current skill should keep
that momentum and make versioned changes visible across prompts, schemas,
fixtures, and Slack behavior.

## Recommended Direction

Do not rework the current cyber skill into a generic startup framework.

Instead:

- keep the cyber-specific analytical depth
- adopt the external repo's modularity
- adopt its progress/resume model
- adopt stronger verification gates
- package the whole thing as a private cyber investment workflow system

That preserves the current domain advantage while improving usability,
repeatability, and maintainability.

## Proposed Next Implementation Steps

1. Carve `cyber-vc-compare` and `cyber-vc-theme` into explicit standalone
skills while retaining `cyber-vc-analyst` as the compatibility wrapper.
2. Add a shared progress-state convention for long-running research.
3. Add a verification checklist section to the skill workflow and fixtures.
4. Introduce `quick` / `standard` / `deep` research modes for Slack and CLI
invocations.
5. Promote the private repo structure so prompts, schemas, fixtures, and
release notes become the system of record rather than accumulating inside one
skill directory.

## Bottom Line

The external `startup-skill` repo is a better model for modular packaging and
resumable research orchestration.

The current `cyber-vc-analyst` setup is a better model for domain depth,
evidence handling, and real workflow integration.

The right move is to combine those advantages: keep the cyber-specific
investment logic, but evolve the implementation into a small, versioned,
multi-skill private workflow repo with checkpoints, verification passes, and
artifact reuse.
