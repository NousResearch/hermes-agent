---
template_id: sec-vuln-remediation
version: "1.0.0"
spec_file: hermes_cli/kanban_templates/sec-vuln-remediation.md
status: normative

input:
  role: scout-issue
  severity_tag: SEC
  severities: [CRITICAL, HIGH, MEDIUM, LOW]
  required_fields:
    - vuln_id
    - severity
    - component
    - body
    - evidence_links
    - source_card_id
  done_when_section_required: false   # warn-only when absent

steps:
  - key: corpus-recon
    title: "{{VULN_ID}} child 1/4: corpus-recon"
    assignee: default
    assignee_fallback: default
    gate: auto
    parents: [source]
  - key: repro-patch
    title: "{{VULN_ID}} child 2/4: reproduce + patch"
    assignee: calcifer
    assignee_fallback: default
    gate: auto
    parents: [source, corpus-recon]
  - key: regression-test
    title: "{{VULN_ID}} child 3/4: regression test"
    assignee: default
    assignee_fallback: default
    gate: auto
    parents: [source, repro-patch]
  - key: ship-pr
    title: "{{VULN_ID}} child 4/4: ship + open PR"
    assignee: calcifer
    assignee_fallback: default
    gate: approval
    parents: [source, regression-test]

gates:
  auto_dispatch_steps: [corpus-recon, repro-patch, regression-test]
  approval_gated_steps: [ship-pr]
  ship_block_kind: needs_input
  forbid:
    - auto_merge
    - auto_deploy

idempotency:
  key_pattern: "sec-vuln-remediation:{VULN_ID}:{STEP_KEY}"
  preflight: "children-of-source + workflow_template_id"
  no_rewrite: true
  force_flag: "--force"

placeholders:
  - "{{VULN_ID}}"
  - "{{SEVERITY}}"
  - "{{COMPONENT}}"
  - "{{SOURCE_CARD_ID}}"
  - "{{SOURCE_TITLE}}"
  - "{{EVIDENCE_LINKS}}"
  - "{{ISSUE_BODY}}"
  - "{{DONE_WHEN_VERBATIM}}"
  - "{{DONE_WHEN_MAPPED}}"
  - "{{PREV_STEP_ID}}"
  - "{{CHAIN_IDS}}"
  - "{{TEMPLATE_ID}}"
---

# Workflow template spec: `sec-vuln-remediation`

**Status: NORMATIVE.** This file is the single source of truth for the
`hermes kanban workflow sec-vuln-remediation` template. The implementation
(t_0ddc29ce) parses the YAML frontmatter above and follows the prose below.
The end-to-end gate review (t_548066d6) verifies behaviour against this
document. Any change to template behaviour MUST change this file first, and
vice versa — the two never diverge.

**Provenance.** Extracted from the 8 VULN-* chains / 24 done cards on the ops
board (scout cards t_a8678357, t_16011ffd, t_afc4e9f3, t_bdf3243f,
t_7d133338, t_4bb5c0b4, t_87922f45, t_4c028632; root proposal t_c5df2ff8).
The historical shape — scout/recon → reproduce → patch → regression test →
ship/open PR — becomes a registered template with automatic child spawning.

---

## 1. Purpose

Running a SEC severity-tagged scout issue card through this template
instantiates the standard 4-step remediation chain:

1. **corpus-recon** — corpus-first recon, report only, no code.
2. **repro-patch** — reproduce the failure mode and apply a minimal fix.
3. **regression-test** — red-green regression tests on the existing suite.
4. **ship-pr** — APPROVAL-GATED: open a scoped PR against origin/main.

Steps 1–3 are auto-dispatchable. Step 4 is parked at a human approval gate
from the moment the chain is created and can only advance via an explicit
Sab unblock. Auto-merge and auto-deploy are FORBIDDEN everywhere in the
chain (section 5.4).

## 2. Template identity

| Field | Value |
|---|---|
| `template_id` | `sec-vuln-remediation` |
| Spec/template file | `hermes_cli/kanban_templates/sec-vuln-remediation.md` (this file) |
| Stored on every spawned card | `workflow_template_id = 'sec-vuln-remediation'` |
| Stored on every spawned card | `current_step_key = <step key>` (one of `corpus-recon`, `repro-patch`, `regression-test`, `ship-pr`) |
| Discovery | `hermes kanban workflow list` |

Both columns already exist on `tasks` (`kanban_db.py`); the dispatcher
records `current_step_key` into `task_runs.step_key` at run start. The
columns are informational for routing in v1 — gating comes from the
parent links and the sticky block described in section 5.

## 3. Input contract (normative)

The template takes exactly **one** source card reference: a card id
(`t_…`) or a VULN-id (`VULN-XX-000`). The card must satisfy ALL of:

### 3.1 Scout-issue predicate

The referenced card is a scout issue card if ANY holds:

- `created_by == 'scout'`, or
- `idempotency_key` starts with `scout:`, or
- title starts with `[gh]` (the scout convention for GitHub-derived issues).

A synthetic card used in tests MUST set `created_by=scout` or a `scout:`
idempotency key.

### 3.2 SEC severity-tagged predicate

The card must carry the SEC tag AND a parseable severity:

- SEC tag: `[SEC]` in the title (case-insensitive), or `[SEC]` anywhere in
  the body.
- Severity: `[SEC][<SEVERITY>]` in the title, or `severity <SEVERITY>` in
  the body (case-insensitive). `<SEVERITY>` ∈ {CRITICAL, HIGH, MEDIUM, LOW}.
  Unknown severity values are a validation error, not a pass-through.

Cards tagged otherwise (e.g. `[P1]`, `[P2]`, `[auto]`, un-tagged) are
rejected with a clear error naming the tag found and the expected `[SEC]`
form.

### 3.3 Required fields

| Field | Extraction rule (first match wins) | Required |
|---|---|---|
| `vuln_id` | Structured block `VULN-id:` line; else regex `VULN-[A-Z]{2}-\d{3}` in title or body | YES |
| `severity` | Structured block `Severity:` line; else §3.2 severity rule | YES |
| `component` | Structured block `Component:` line; else the noun phrase of the FULL CONTEXT clause after `(VULN-…)` (the affected service/path) | YES |
| `body` | The card's own `body` column, non-empty | YES |
| `evidence_links` | Structured block `Evidence:` line; else the `Evidence: <url>` line; else the source card's own id | YES (≥1 entry) |
| `source_card_id` | The card id of the referenced card | YES (implied) |
| `done_when` section | `Done when:` paragraph in the body (see §4.4) | NO (warn-only) |

### 3.4 Canonical structured block

New scout cards SHOULD carry a machine-readable block (the implementation
must parse it when present, and fall back to the regex rules when absent):

```markdown
## sec-vuln fields
VULN-id: VULN-EM-003
Severity: HIGH
Component: communication/email (email service)
Evidence: https://github.com/veroscale/veroscale-services/issues/6
```

`Evidence:` accepts a comma-separated list of URLs and/or card ids.

### 3.5 Validation errors

Any failed predicate or missing required field aborts BEFORE any card is
created. The error (exit code 2, see §7.2) enumerates exactly what failed,
in this order: unknown card → not a scout issue → not SEC-tagged → missing
severity → missing required field(s) (each named, with the line format that
would fix it). No partial writes may occur on validation failure, and
`--dry-run` performs the same validation.

## 4. Child chain (normative)

### 4.1 Chain shape and ordering

Four children, created in order (1 → 2 → 3 → 4) so each body can embed the
real id of its predecessor. Parent links:

- step 1: `parents = [source_card]`
- steps 2–4: `parents = [source_card, <previous step child id>]`

This gives automatic sequential promotion through `recompute_ready`: a
step becomes `ready` only when the source card AND the previous step are
`done`. All four link the source card so `kanban show <source>` shows the
full chain, and so the idempotency pre-flight (§6.1) can find existing
children cheaply.

Every child references the parent VULN-id and the source issue in its body
(§4.3 provenance block) — the link row alone is not sufficient.

### 4.2 Step table

| # | key | title pattern | assignee (fallback) | gate | workspace |
|---|---|---|---|---|---|
| 1 | `corpus-recon` | `<VULN_ID> child 1/4: corpus-recon` | `default` | auto | scratch |
| 2 | `repro-patch` | `<VULN_ID> child 2/4: reproduce + patch` | `calcifer` (`default`) | auto | scratch |
| 3 | `regression-test` | `<VULN_ID> child 3/4: regression test` | `default` | auto | scratch |
| 4 | `ship-pr` | `<VULN_ID> child 4/4: ship + open PR` | `calcifer` (`default`) | **approval** | scratch |

Priority from severity: CRITICAL=100, HIGH=80, MEDIUM=60, LOW=40 (all four
children inherit it).

**Assignee resolution.** The template declares a primary assignee and a
fallback per step. The command resolves each step at run time against
`hermes profile list`:

- primary exists → use it; else fallback exists → use it (and print the
  substitution); else → validation error (exit 2) listing valid profiles.
- The dispatcher SILENTLY drops cards with unknown assignees — the command
  must therefore verify assignees before creating anything, and `--dry-run`
  prints the resolved assignee per step.
- Optional `--assignee <KEY>=<PROFILE>` (repeatable) overrides per step;
  the profile is validated the same way.

### 4.3 Body templates (verbatim)

Common provenance block, prepended to every child body (placeholders
replaced; `{{PREV_STEP_ID}}` empty for step 1; `{{CHAIN_IDS}}` only
populated on the ship card, listing ids of steps 1–3):

```markdown
<!-- AUTO-SPAWNED by hermes kanban workflow {{TEMPLATE_ID}} v1.0.0 — do not edit the provenance block -->
## Source issue
- VULN-id: {{VULN_ID}}
- Severity: {{SEVERITY}}
- Affected component: {{COMPONENT}}
- Source card: {{SOURCE_CARD_ID}} — "{{SOURCE_TITLE}}"
- Evidence: {{EVIDENCE_LINKS}}
- Chain step: {{STEP_KEY}} ({{STEP_N}}/4); previous step: {{PREV_STEP_ID}}
```

Step-specific bodies (each ends with the verbatim issue body and the
issue's done-when block):

**Step 1 — corpus-recon**

```markdown
## Scope
CORPUS-FIRST: search the corpus (cfd signals, repo tests/specs/docs, sibling
worktrees/branches) BEFORE any code. Survey the affected component named in
the issue. Extend existing patterns — never introduce a parallel layer.

## Done when
1. Recon report published as a comment on THIS card (or attached artifact):
   file paths + line refs, trust model, exposure verdict
   (confirmed / not-reproducible / already-fixed), patterns to extend,
   signals cited.
2. No code changes in this step (report only).
3. Any human-only actions discovered (DNS, Workspace/admin UI, irreversible
   ops) are listed for the patch step — they become their own blocked
   [Sab action] cards per gate G2, never folded into an auto step.
4. {{DONE_WHEN_MAPPED}}

## Issue body (verbatim)
{{ISSUE_BODY}}

## Issue done-when (verbatim)
{{DONE_WHEN_VERBATIM}}
```

**Step 2 — repro-patch**

```markdown
## Scope
Reproduce the failure mode from the issue (or cite recon's repro evidence),
then apply the minimal fix. Branch from origin/main; the diff must stay
inside this VULN-id's scope (gate G4). Extend existing helpers — do not bolt
on parallel machinery. If the fix requires a design decision: STOP, ship a
design-doc PR with numbered Sab gates, and block for approval (gate G1).

## Done when
1. Failure mode reproduced against the fixed code, or recon's repro
   evidence cited.
2. Minimal-diff fix applied; tests added/extended (red-green proof lands in
   step 3).
3. No human-only action performed here — any discovered [Sab action] was
   split to its own blocked card (gate G2).
4. {{DONE_WHEN_MAPPED}}

## Issue body (verbatim)
{{ISSUE_BODY}}

## Issue done-when (verbatim)
{{DONE_WHEN_VERBATIM}}
```

**Step 3 — regression-test**

```markdown
## Scope
Regression tests for the fix. Corpus-first: extend the existing test surface
(no new test framework, no parallel suite). Prove the tests are load-bearing.

## Done when
1. Tests added to the existing suite; actual output pasted in the card.
2. Red-green proven: reverting the fix fails the new tests; fix applied →
   full suite passes.
3. {{DONE_WHEN_MAPPED}}

## Issue body (verbatim)
{{ISSUE_BODY}}

## Issue done-when (verbatim)
{{DONE_WHEN_VERBATIM}}
```

**Step 4 — ship-pr**

```markdown
## Scope
APPROVAL-GATED STEP. You are dispatched only after Sab approved via
`hermes kanban unblock <this-card>`. Open ONE PR against origin/main with
the fix + regression tests; verify mergeability and CI; post evidence.
Do NOT merge. Do NOT deploy. Merge and deploy are Sab actions.

## Sab gates (all must hold; post evidence for each)
- gate_1: PR open against origin/main, NOT merged, NOT auto-merged
- gate_2: PR diff scoped to {{VULN_ID}} only — no sibling/unrelated work
- gate_3: test output pasted verbatim; typecheck/lint clean
- gate_4: no auto-deploy artifact (no deploy hook fired, no Kamal/CF push)

## Done when
1. Gates 1–4 verified and evidenced as comments on this card.
2. {{DONE_WHEN_MAPPED}} (items mentioning PR-merged/coord-with-Sab are
   verified AFTER Sab merges — see below).
3. After Sab merges: re-verify merged state via the gh API, comment the
   merge commit + close/comment the issue, then complete this card.

## Chain
{{CHAIN_IDS}}

## Issue body (verbatim)
{{ISSUE_BODY}}

## Issue done-when (verbatim)
{{DONE_WHEN_VERBATIM}}
```

### 4.4 done_when derivation from the issue body

The scout card's `Done when:` paragraph is the source of the chain's
definition of done. Algorithm (implement exactly):

1. Extract the `Done when:` section: from the line beginning `Done when:`
   (case-insensitive) through the next blank line or the `Idempotency:`
   line, whichever comes first. Parse items as `(\d+)\)\s*(.+)` or `-\s*(.+)`.
2. Classify each item into exactly one step, first matching rule wins
   (case-insensitive regex):

   | Step | Item matches |
   |---|---|
   | `ship-pr` | `\b(PR|pull request)\b` … `\b(open|merged|merge)\b`; `\bmerged? to (main|master)\b`; `\b(coord|coordinate).*\bSab\b`; `\birreversible\b`; `\bdeploy\b` |
   | `regression-test` | `\bregression test\b`; `\btest asserts\b`; `\btests? (pass|fail|added|written)\b`; `\bfailure mode.*cannot be reproduced\b`; `\b(reproduce|reproduction).*\bfixed\b` |
   | `corpus-recon` | `\b(recon|scoping|corpus|survey|investigate|map)\b`; `\bcite\b`; `\bscope check\b`; `\bconfirm which\b` |
   | `repro-patch` | everything else (default bucket) |

3. `{{DONE_WHEN_MAPPED}}` per child = the quoted items classified to that
   child (e.g. `- "(2) regression test asserts each role/credential
   combination"`), or the sentence `No issue done-when items map to this
   step; static acceptance above applies.` when empty.
4. `{{DONE_WHEN_VERBATIM}}` = the full extracted section, verbatim, in
   EVERY child — nothing is lost.
5. If the issue body has no `Done when:` section: warn on stderr, use the
   static acceptance only (the field is not in the required list).

## 5. Gates (normative)

### 5.1 Auto-dispatch (steps 1–3)

Steps 1–3 are created as ordinary gated cards. Once their parents are
`done` they promote to `ready` and the dispatcher runs them with no human
intervention. No approval, no review request, no block.

### 5.2 Approval gate (step 4) — mechanics

The ship card is **parked at the approval gate from creation**. Verified
kernel behaviour that the implementation MUST respect (kanban_db.py):

- A card created with `initial_status=blocked` alone is NOT sticky:
  `recompute_ready` promotes blocked cards whose parents are done unless a
  `blocked` event row exists (`_has_sticky_block`). Creating the ship card
  as plain `blocked` would therefore auto-dispatch it once the chain
  completes — a gate bypass.
- `block_task` only fires from `running`/`ready` (returns False otherwise),
  and a card created with parents lands in `todo`.

**Required implementation sequence (single `write_txn`):**

1. `create_task(... initial_status="running", parents=[])` → status `ready`.
2. `block_task(conn, id, kind="needs_input", reason=<gate text below>)` →
   status `blocked` + `blocked` event → sticky.
3. Insert `task_links` rows for `(source_card → ship)` and
   `(regression-test child → ship)` directly in the same transaction.

Gate text (verbatim):

```
APPROVAL GATE (sec-vuln-remediation) — {VULN_ID}. Steps 1-3 of the chain
must complete before this card may run. Approving (hermes kanban unblock
<id>) authorizes ONLY: opening a scoped PR against origin/main with
evidence (gates 1-4 in the body). Auto-merge and auto-deploy are FORBIDDEN;
merge and deploy remain Sab actions outside this card.
```

Release semantics (kernel-verified):

- Sab unblock before the chain completes → `_landing_status_after_parents`
  parks the card in `todo` (dependency-gated) → auto-promotes when parents
  are done → dispatches. The human approval still happened; nothing runs
  without it.
- Sab unblock after the chain completes → `ready` → dispatches to the
  ship assignee.
- There is NO path to dispatch without Sab's unblock: sticky-blocked cards
  are skipped by `recompute_ready`, and the dispatcher only picks `ready`.

### 5.3 Borrowed Sab-gate steps (from t_1f959c11, t_49de4413, t_3ed43201)

These human checkpoints remain explicit even on auto-spawned chains. They
are inherited patterns, not new policy:

- **G1 — design-first.** When a fix needs a design decision (approach
  choice, new dependency, policy number), the worker does NOT implement
  blind. It ships a design-doc PR with a numbered "Sab sign-off gates"
  list (t_1f959c11 pattern) and blocks the patch card
  (`kanban_block kind=needs_input`, reason listing the numbered gates).
  Sab answers via chat approval or `kanban unblock`; no code lands before.
- **G2 — [Sab action] split.** Any human-only action discovered during the
  chain (Workspace/DNS/admin UI, credential rotations, irreversible ops)
  becomes its OWN blocked card titled `[Sab action] <what> (<VULN_ID>)`
  with a numbered "What Sab needs to do" list + a verification checklist
  the worker re-runs after Sab signals done (t_49de4413 pattern). It is
  never folded into an auto-dispatched step, and auto steps never perform
  the action themselves.
- **G3 — phased/irreversible sign-off.** Rollouts touching external state
  (DNS, email auth, quotas, data destruction) advance one phase at a time;
  each phase is a separate Sab sign-off, with live-state verification
  BEFORE the mutation and evidence comment AFTER (t_3ed43201 pattern).
  30-day monitoring windows apply where the change is policy-carrying.
- **G4 — PR hygiene at ship.** PR base must be `origin/main`; the diff must
  be scoped to the VULN-id (the t_1f959c11 assessment-gate lesson: a stale
  base re-shows merged sibling work and accumulates unrelated changes —
  both are ship-gate failures); tests re-run on the corrected base with
  pasted output. Merge is a Sab action (sole-committer repos: Sab merges or
  explicitly delegates).

### 5.4 Hard prohibitions

- **No auto-merge.** No worker in any step may merge a PR, invoke `gh pr
  merge`, or complete the chain on a merged state it caused. The ship step
  opens the PR and stops; the issue's `PR merged` done-when item is
  verified after Sab merges.
- **No auto-deploy.** No step may trigger a deployment (Kamal/CF push,
  deploy hooks, restart of live services) as part of the chain. A vuln
  whose remediation includes a deploy gets a separate `[Sab action]` deploy
  card (G2).
- These prohibitions are written into every child body (step 2 and 4
  templates above) so the worker sees them even if the card is dispatched
  out of context.

## 6. Idempotency and re-spawn rules (normative)

### 6.1 Keys and pre-flight

- Every child is created with `idempotency_key =
  sec-vuln-remediation:<VULN_ID>:<STEP_KEY>` (the `create_task`
  idempotency layer returns the existing card instead of duplicating).
- BEFORE creating anything, the command pre-flights:
  `SELECT id, current_step_key FROM tasks WHERE workflow_template_id =
  'sec-vuln-remediation' AND id IN (children of the source card)` — i.e.
  tasks linked from the source card with the template set. If any exist,
  print `chain already exists: <ids>` and stop (exit 0), UNLESS `--force`.

### 6.2 Re-run behaviour (no rewrite)

- Re-running against a VULN-id that already has children is a no-op that
  reports the existing chain: existing cards are returned by their
  idempotency keys; their titles, bodies, and status are NEVER rewritten
  (the same no-rewrite rule as the back-link migration).
- **Partial chains**: a pre-existing step card is reused; missing steps are
  created; the ship card's sticky block is applied ONLY if the reused ship
  card is not already blocked. Never re-fire `block_task` on an
  already-sticky card (would trip block-recurrence accounting).
- **`--force`**: bypasses the pre-flight stop ONLY when the caller intends
  a genuinely new chain (old chain archived/abandoned). `--force` still
  honours per-step idempotency keys and the no-rewrite rule. `--force` is
  NOT a gate bypass: the ship card is still created sticky-blocked.

## 7. CLI contract (normative for the implementer)

### 7.1 Commands

```
hermes kanban workflow list
hermes kanban workflow <template> <card-ref> [--dry-run] [--force]
    [--assignee KEY=PROFILE]... [--json]
```

- `<template>` = template id (currently `sec-vuln-remediation`).
  Resolution: `$HERMES_KANBAN_TEMPLATES_DIR` (tests use this) →
  packaged `hermes_cli/kanban_templates/<template>.md`; filename must
  equal the frontmatter `template_id`.
- `<card-ref>` = card id (`t_…`) or VULN-id. VULN-id resolution (in
  order, over non-archived tasks): (1) tasks whose TITLE contains the
  VULN-id — among those, prefer `created_by=scout`/`[gh]`-titled cards;
  (2) if exactly one candidate remains, use it; (3) if multiple remain,
  error listing candidate ids (a VULN-id alone is ambiguous once a chain
  exists — children carry it in their bodies). A card whose title carries
  the VULN-id but fails the §3.1 scout predicate is a validation error.
- `--dry-run`: full validation + assignee resolution, prints the would-be
  children (title, assignee, parents, gate state) without writing. Exit 0
  on success, 2 on validation failure.
- `--json`: machine-readable output (shape below).

### 7.2 Output and exit codes

- Exit 0: chain created (or no-op existing chain, or dry-run success).
- Exit 2: usage/validation (unknown template, unknown card, not a scout
  issue, not SEC-tagged, missing severity/fields, unknown assignee, bad
  flag). Error text enumerates failures (§3.5).
- Exit 1: runtime/DB failure (partial write impossible: creation happens in
  one transaction).
- Human output: `Created chain for <VULN_ID> (<severity>):` then per child
  `t_…  <title>  assignee=<p>  parents=<ids>  gate=auto|APPROVAL` and a
  final line naming the ship card + its unblock command
  (`hermes kanban unblock t_…`).
- JSON shape: `{"template_id", "source_card", "validated": {…},
  "children": [{"step", "id"|null, "title", "assignee", "parents",
  "gate"}], "ship_gate": {"card_id", "unblock": "hermes kanban unblock …"}}`.

### 7.3 DB write path

- Add `workflow_template_id` + `current_step_key` params to
  `create_task` (columns exist; write path does not) and wire
  `--workflow-template-id` / `--current-step-key` onto `hermes kanban
  create` for parity with the existing list filters.
- The ship-card transaction (§5.2) needs create + block_task + link
  insertion in one `write_txn` — expose a small helper rather than three
  separate calls.

### 7.4 Docs and help

- `hermes kanban --help` + `workflow` subcommand help text.
- `website/docs/user-guide/features/kanban.md` (and the zh-Hans mirror if
  present) gains a "Workflow templates" section documenting the command,
  the input contract, and the ship gate.

## 8. Acceptance criteria (testable)

The implementation (t_0ddc29ce) is complete when, against a synthetic SEC
scout card (fixture in §10):

1. `workflow list` shows `sec-vuln-remediation`.
2. `workflow sec-vuln-remediation <card> --dry-run --json` validates and
   prints 4 children with correct titles/assignees/parents, ship gate
   `approval`, exit 0.
3. Real run creates exactly 4 cards: titles match §4.2 patterns; bodies
   contain the provenance block, verbatim issue body, done-when mapping,
   and gate text; `workflow_template_id` and `current_step_key` set;
   parent links = [source] / [source, prev].
4. The ship card is `blocked` with `block_kind=needs_input`, a `blocked`
   event row exists (sticky), and `hermes kanban list --workflow-template-id
   sec-vuln-remediation` finds all 4.
5. `recompute_ready` does NOT promote the ship card after its parents are
   done; `hermes kanban unblock t_ship` → `ready` only then.
6. Re-running the same command no-ops with the existing ids (idempotency
   keys + pre-flight), zero rewrites.
7. Non-scout / non-SEC / missing-field cards are rejected with exit 2 and
   field-level messages, no cards created.
8. Unknown assignee (profile absent) → exit 2 before any create.
9. No path exists in the codebase for auto-merge or auto-deploy: grep for
   `gh pr merge` / merge calls / deploy invocations inside the template
   execution path returns nothing (or only explicit user-flag paths that
   are not part of the template).
10. E2E (t_548066d6): Sab walks one synthetic chain end-to-end; sign-off
    recorded on the scout card; VULN-MS-003 (t_f7c53ef0) either folded in
    or explicitly noted why not.

## 9. Back-link migration contract (for t_10e2b69c)

The existing 8 VULN-* chains get the template reference WITHOUT rewriting
titles, bodies, or status:

```sql
UPDATE tasks
   SET workflow_template_id = 'sec-vuln-remediation'
 WHERE workflow_template_id IS NULL
   AND id IN (SELECT DISTINCT child_id FROM task_links
              WHERE parent_id IN (…8 scout card ids…))
   OR (workflow_template_id IS NULL AND (title LIKE 'VULN-%' OR title LIKE '%(VULN-%' OR body LIKE '%VULN-%'))
```

(`workflow_template_id` is the "equivalent" of `template_id` — the column
already exists.) Current_step_key is NOT back-filled (historical cards
never had step keys; inventing them would be a rewrite of meaning). The
migration report must list per VULN-id: card count, back-link status, and
anomalies — including VULN-MS-003 (t_f7c53ef0, currently blocked awaiting
follow-on cards) which is back-linked but its chain is NOT re-spawned.

## 10. Synthetic fixture (for implementation + e2e tests)

```markdown
[gh] veroscale-services: [SEC][HIGH] Example vuln in the widget service (VULN-TST-001) — #999 unaddressed

FULL CONTEXT: veroscale-services#999 (VULN-TST-001) — example vulnerability
in the widget service, severity HIGH. Test fixture — do not act on.

Evidence: https://github.com/veroscale/veroscale-services/issues/999

CORPUS-FIRST: search signals for 'VULN-TST-001' before scoping.

Done when: (1) widget service validates input, (2) regression test asserts
the failure mode, (3) PR merged to main.

Idempotency: scout:gh:veroscale-services:999
```

Expected classification: (1)→repro-patch, (2)→regression-test,
(3)→ship-pr. No items map to corpus-recon (static acceptance only).

## 11. Out of scope (v1)

- Dispatcher routing on `current_step_key` (columns are informational in
  v1; gating is parent links + the sticky block).
- Per-step model pinning (steps run on the assignee profile's model).
- Template chaining (a template spawning another template).
- General-purpose templating language beyond the placeholder list above.
