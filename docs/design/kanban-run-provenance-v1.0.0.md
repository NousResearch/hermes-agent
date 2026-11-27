# Contract: Structured Run Provenance for Hermes Kanban

- **Version:** 1.0.0-draft
- **Status:** Proposed — awaiting `factory-director` routing, then security review
- **Author:** `solution-architect`, card `t_8c86298c`
- **Date:** 2026-08-20
- **Supersedes:** free-text subject/final SHA in `task_runs.summary` / `task_runs.metadata` prose

---

## §0 — Problem

The provenance broker (research `t_5247914a`) needs, per gated candidate:

```
{run_id, task_id, authenticated profile, outcome, implementation subject SHA,
 final candidate SHA, artifact paths + sha256, completed_at}
```

Today subject and final SHA commonly live in comments, `summary`, or `metadata` prose. **Parsing
prose is a tamper and ambiguity surface.** A worker writes the text; a worker can therefore choose
what the exporter reads. The canonical authority is the local Kanban SQLite database, so the fix is
to make these fields *structured columns the kernel owns*, not text the worker composes.

This document decides the schema, who may write each field, mutability, and the trust seam to the
DSSE attestation design. It is a contract decision, not an implementation.

---

## §1 — Source anchors (verified, not recalled)

Read-only inspection of the live board DB (`immutable=1` open, no writes):

```
db: ~/.hermes/kanban/boards/hermes-agent/kanban.db

task_runs (16 cols)
  id PK, task_id NOT NULL, profile, step_key, status NOT NULL, claim_lock,
  claim_expires, worker_pid, max_runtime_seconds, last_heartbeat_at,
  started_at NOT NULL, ended_at, outcome, summary, metadata, error
  indexes: idx_runs_status, idx_runs_task

tasks (37 cols)
  ... branch_name, project_id, current_run_id, idempotency_key, session_id ...

task_events (6 cols)
  id PK, task_id NOT NULL, run_id, kind NOT NULL, payload, created_at
  indexes: idx_events_run, idx_events_task
```

**Two facts from the schema shape the decision:**

1. `task_runs.profile` and `task_runs.outcome` **already exist as first-class columns.** The
   additive-columns-vs-envelope question is therefore not symmetric — half the required fields are
   already structured. Only the SHAs and artifact hashes live in prose.
2. `task_runs.metadata` is a TEXT blob. It is the surface being eliminated, so the contract must not
   solve the problem by putting a new JSON schema *inside* it.

---

## §2 — Decision

**Additive typed columns on `task_runs`, plus one child table for artifacts. No envelope blob.**

### §2.1 `task_runs` — new columns

| Column | Type | Null | Writer | Notes |
|---|---|---|---|---|
| `subject_sha` | TEXT | yes until claim | **kernel only** | Full 40-char lowercase hex. Captured at claim from the typed task target. Never writable via the worker path. |
| `verified_head_sha` | TEXT | yes | **kernel only** | Full 40-char lowercase hex. The head the run actually verified, stamped at terminalization. For implementation runs it equals `subject_sha`. **This is the run-level SHA pair; it is *not* `final_candidate_sha`.** See §4 and the normative resolution in v1.1.0 §A.4. |
| `provenance_version` | INTEGER | no | kernel | Starts at `1`. Lets a future schema change be detected rather than silently reinterpreted. |
| `corrects_run_id` | INTEGER | yes | kernel | FK → `task_runs(id)`. Non-null only on correction rows (§5). |

> **⚠ Corrected by v1.1.0 §A.4 (B3).** An earlier revision of this table listed a
> `final_candidate_sha` column and a stored `role` column. **Neither is part of the normative
> schema.** `final_candidate_sha` is an attestation-layer field the broker derives (§4, v1.1.0
> §A.4); the run stores `verified_head_sha` instead. `role` is derived at query time from
> `profile` (§2.3) and is not persisted. The rows above are the complete, normative column list.

`profile`, `outcome`, `ended_at` are **existing** columns and are reused as-is. `completed_at` for
export purposes is `ended_at`; no new column.

**CHECK constraints (fail-closed at write time):**
```sql
CHECK (subject_sha IS NULL OR (length(subject_sha) = 40
       AND subject_sha NOT GLOB '*[^0-9a-f]*'))
CHECK (verified_head_sha IS NULL OR (length(verified_head_sha) = 40
       AND verified_head_sha NOT GLOB '*[^0-9a-f]*'))
CHECK (provenance_version >= 1)
```

> **⚠ Corrected (B2). The pattern `sha GLOB '[0-9a-f]*'` is WRONG and must not be implemented.**
> A leading `[0-9a-f]` character class followed by `*` anchors only the **first** character; the
> `*` then matches any remaining 39 characters, including uppercase and non-hex. Demonstrated
> against real SQLite (`verify_adr0007_mechanisms.py`, table `legacy_sha_check`): `'a' + 'Z'*39`
> and `'a'*39 + 'g'` are both **accepted** by the old pattern and both **rejected** by the
> `NOT GLOB '*[^0-9a-f]*'` form above. The negated form is normative: it rejects any character
> outside `[0-9a-f]` at any position. Length is still checked separately because GLOB alone
> cannot express "exactly 40". Hostile cases are executable — see A5 and §10.


### §2.2 New table `run_artifacts`

```sql
CREATE TABLE run_artifacts (
  id            INTEGER PRIMARY KEY,
  run_id        INTEGER NOT NULL REFERENCES task_runs(id),
  artifact_path TEXT    NOT NULL,
  sha256        TEXT    NOT NULL,
  created_at    INTEGER NOT NULL,
  UNIQUE(run_id, artifact_path),
  CHECK (length(sha256) = 64 AND sha256 NOT GLOB '*[^0-9a-f]*')
);
CREATE INDEX idx_run_artifacts_run ON run_artifacts(run_id);
```

The `sha256` CHECK uses the same negated-GLOB form as §2.1 and for the same reason (B2):
`sha256 GLOB '[0-9a-f]*'` would validate only the first of the 64 characters.

One-to-many, per `platform-engineer`'s constraint. `UNIQUE(run_id, artifact_path)` is deliberate:
double-recording an artifact becomes an error rather than a silent duplicate, closing one of the
ambiguity surfaces this card exists to eliminate.

---

## §3 — Trust semantics

### §3.1 Who may write what

| Field | Worker | Kernel | Broker |
|---|---|---|---|
| `artifact_path`, `sha256` | **supplies** (pre-terminal) | validates + stamps `created_at` | reads |
| `subject_sha` | never | **stamps at claim** | reads |
| `role`, `profile` | never | **stamps at dispatch** | reads |
| `outcome` | never | **derives from transition** | reads |
| `ended_at` | never | stamps at terminalization | reads |
| final candidate SHA | never | never | **derives** (§4) |

**Invariant:** a caller-supplied role, outcome, or identity is a *claim*, never authentication. The
worker's only provenance input is `{artifact_path, sha256}` pairs, and even those are hashes over
content the broker can re-verify independently.

### §3.2 Role independence is per-RUN, not per-commit

Corrected from an earlier draft after `platform-engineer` review. **Commit count is transport; run
identity is provenance.** One run can emit three commits; three commits can be assembled by one
actor. Neither demonstrates that review, QA and security were performed by distinct principals.

The evaluator asserts, over **terminal `task_runs` rows**:

1. For each required role in `{code_review, qa, security}` (+ `implementation` when represented),
   at least one terminal run exists with that `role`.
2. The `profile` values across those required roles are **pairwise distinct**.

> The same profile satisfying two required roles is a **FAIL**, not a warning. Self-review is the
> exact failure this table exists to prevent.

This is enforceable *only* because `profile` is kernel-stamped: a worker cannot write another
profile's name into its own run row.

### §3.3 Repository and PR identity

- The run stores **locators**, not authority.
- The run does **not** carry a PR number. A PR may not exist when the run ends; requiring it would
  force either a nullable field everyone treats as optional, or a write after terminalization —
  which §5 prohibits.
- The broker independently **resolves and validates** `owner/repo` and PR number to immutable
  numeric GitHub IDs at attestation time, and binds those.
- **Branch name is never an authority.** It is worker-influenced and mutable; trusting it is the
  same defect class as trusting a role string in candidate JSON. It may serve as a resolution
  *hint* only.

`repo_numeric_id` is deliberately **not** added to `task_runs` in v1.0.0. Whether it must be
persisted belongs to the DSSE trust design (§7 seam), not to this schema.

---

### §2.3 — `role` is derived, not stored

`role` is **not** a persisted column. It is resolved at query/export time through a static
`profile → role` map pinned in protected-base policy:

```
code-reviewer     -> code_review
qa-verifier       -> qa
security-reviewer -> security
<implementer>     -> implementation
```

**Rationale (`research-scout`, `t_5247914a`):** persisting a coarser `role` alongside `profile`
forces a data migration every time a profile is reclassified, and creates a second place where role
can disagree with the dispatcher record. `profile` is the kernel-stamped fact; `role` is a view over
it. Recorded as an opinion adopted, not a research finding — the memo is explicit that it only ever
modelled `profile` / `authenticated_profile`.

Consequence for §3.2: independence is asserted over **distinct `profile` values**, with the map
applied to check role coverage. The map lives in protected-base policy so a candidate cannot
reclassify a profile into a role it did not perform.

---

## §4 — Subject SHA vs verified head SHA vs final candidate SHA

> **⚠ Normative resolution of the v1.0.0 `final_candidate_sha` contradiction (B3).**
> An earlier revision of this section was internally inconsistent: §2.1 listed
> `final_candidate_sha` as a run column, §4.1 said to carry both SHAs on every run, and the
> closing paragraph of §4 said it is *not* a run field. **The single normative model is the
> three-SHA model below**, stated in full in v1.1.0 §A.4. Exact superseded text is enumerated
> in v1.1.0 §A.4.1. Where any older wording survives elsewhere, §A.4 wins.

**Three distinct SHAs. Two live on the run; the third is an attestation-layer field.**

| | Subject SHA | Verified head SHA | Final candidate SHA |
|---|---|---|---|
| Meaning | the commit the run's work was performed against | the head the run actually verified when it terminated | the head the **attestation** binds its verdict to |
| Known at | claim time | run-completion time | attestation-assembly time (after the run) |
| Stored | `task_runs.subject_sha` | `task_runs.verified_head_sha` | **not a Kanban column** — broker-derived |
| Written by | kernel, at claim | kernel, at terminalization | broker, at attestation time |

### §4.1 — Both *witnessed* SHAs on every run; the final candidate stays broker-side

Kanban records only what the kernel can **honestly witness**:

- Carry **both** `subject_sha` and `verified_head_sha` on **every** run, uniformly. Do **not**
  make it conditional on run type.
- For implementation runs the two collapse to the same value. That redundancy is harmless —
  special-casing it costs more than it saves and creates a branch where a field can be
  legitimately absent.
- For review / QA / security runs they can **legitimately diverge**: the branch may advance
  between the commit a run reviewed and the head an attestation is later requested against.

**That divergence is the tamper check, not a defect.** It is what makes the broker's fail-closed
condition *"PR head ≠ requested `expected_head_sha`"* (`t_5247914a` §7) evaluable at all. A schema
that stored only one SHA would have made stale-evidence rebinding undetectable at the run layer.

This does **not** conflict with §5 immutability: `verified_head_sha` is written by the kernel at
terminalization, in the same transition that sets `ended_at`. It is never a post-terminal write.

**Why the final candidate SHA is *not* a run column.** It is knowable only after assembly — i.e.
after the run is terminal — and §5 makes terminal runs immutable. Storing it on the run would
require exactly the mutation this contract prohibits, and a Kanban column named
`final_candidate_sha` / `final_sha` would invite a false equivalence between *"the head this run
verified"* and *"the head the attestation binds"*. The broker derives and binds it (§3.3); the run
field is *provenance*, the attestation field is *authority*. They must agree on `verified_head_sha`,
and the broker fails closed if they do not.

**Why subject SHA must come from a kernel-captured typed task target** (not worker JSON, not the
evidence branch HEAD): if it is read from branch HEAD, the worker chooses *what was reviewed* after
review happened. Capturing at claim makes the subject an **input** to the work rather than an
**output** of it.

---

## §5 — Mutability and corrections

**Terminal runs are immutable.** Once `status` is terminal and `ended_at` is set, no field on that
row may be UPDATEd.

Corrections are **append-only**: a new `task_runs` row with `corrects_run_id` pointing at the row
being corrected.

Broker obligations — all **fail-closed**:

1. A correction **may not** alter provenance already consumed by an emitted attestation.
2. The broker **MUST reject ambiguous or unresolved correction chains**: two corrections of the same
   row, a cycle, or a dangling `corrects_run_id`. This is a hard fail, **not** a
   last-write-wins tiebreak.
3. The exporter reads the resolved head of a chain. An unresolvable chain yields **no export**,
   never a guess.

Commands and results stay **inside the hashed artifact**, not in DB columns. The DB holds identity
and hashes; the artifact holds the narrative. This keeps the tamper surface small and row size
bounded.

---

## §6 — Export

**Transport: push-per-terminal-run. There is no broker-facing pull surface.** (`t_5247914a` §3b,
which rejected tunnel/always-on reachability as inverting the trust direction and coupling
fail-closed behaviour to laptop uptime.)

- The exporter watches run-completion events and **POSTs one envelope per completed gate-relevant
  run**. The broker never reaches into local SQLite — the operator's laptop is neither reachable nor
  an Actions runner.
- **Idempotency key: `(repository_id, task_id, run_id)`.** The broker's insert-only mirror rejects
  resubmission with a different outcome or SHA for the same key.
- **The monotonic cursor is the exporter's, not the broker's.** The exporter tracks the highest
  exported `task_events.id` **against local Kanban** so it can resume after being offline. This is
  deliberately *not* a `give-me-everything-after-N` API the broker calls; exposing that would
  re-create the pull surface `t_5247914a` §3b rejected.
- **Detection without polling prose:** a `provenance_ready` row in `task_events` (`kind` is already a
  first-class column; `idx_events_run` already exists) emitted by the kernel at terminalization.
- **Exporter credentials must be read-only against the canonical SQLite.** An exporter that can
  rewrite the source of truth is not an exporter.
- If the laptop is off, only *new* runs go unmirrored and the broker fails closed for those specific
  runs — cleanly, rather than flapping.

### §6.1 — Minimal export (exclusions are normative)

Exported per run: `{run_id, task_id, profile, outcome, subject_sha, verified_head_sha,
artifact_relative_path, sha256, completed_at}`.

The export carries `verified_head_sha`, **not** `final_candidate_sha` — the broker derives the
final candidate SHA itself (§4, v1.1.0 §A.4) and fails closed if it disagrees with the exported
`verified_head_sha`.

**Excluded — normative, not advisory:**

| Excluded | Why |
|---|---|
| card bodies, other cards, comment text | `t_5247914a` §3b minimal-field list |
| `summary`, `error`, `result` | may carry paths, tokens, or user content |
| artifact **contents** | only path + digest ever crosses the wire |
| **absolute workspace paths** | `/Users/<name>/…` leaks machine identity and OS username |

The last row is an addition from `research-scout` and is worth stating explicitly because the
earlier field list named `workspace_path` unqualified. **Only repo-relative artifact paths are
exported.** An absolute path is both a privacy leak and useless to a broker that cannot see the
filesystem.

Missing or ambiguous fields **fail closed**: no attestation.

---

## §7 — Seam to the DSSE design (deliberately deferred)

This contract does **not** decide:

- whether the attestation must bind numeric repo ID and PR number;
- the DSSE envelope format or signing primitive.

Those belong to the security trust design. This document names the seam so the two cannot silently
disagree: `repo_numeric_id` is **absent in v1.0.0**, and adding it is a `provenance_version` bump.

**Authoritative DSSE design: `t_8513bc6e`** — *"Security design: authenticate PR #36 reviewer roles
without candidate-controlled trust"*, `status=done`, `assignee=security-reviewer`, created by
`factory-director`. Its first `security-reviewer` comment (`created_at=1787270558`, 12,139 chars) is
the binding recommendation.

> **Correction to an earlier revision of this document.** A prior draft recorded `t_8513bc6e` as
> non-existent. That was **my error, not a dangling reference.** The card is present in
> `~/.hermes/kanban/boards/account-gen/kanban.db` (156 tasks). The CLI board resolver misrouted the
> lookup — the known defect carded as `t_9b4f8ded` (*explicit `--board` ignored when
> `HERMES_KANBAN_DB` is set*). Verified by opening the DB directly, read-only:
> `t_8513bc6e: PRESENT`. Absence of a CLI result is not evidence of absence when the resolver
> itself is under repair.

### §7.1 — Binding constraint from `t_8513bc6e` (changes this contract)

The security design answers **YES** to the question of whether the shared GitHub identity defeats
API-derived role provenance:

> *"the shared `SiWarlock` GitHub identity makes the GitHub review API insufficient by itself. The
> API authenticates one GitHub account, association and review `commit_id`; it cannot distinguish
> `code-reviewer`, `qa-verifier`, and `security-reviewer`."*

Independently verified against the live repository:

```
gh pr view 36 --json reviews   ->  [{"author": "SiWarlock", "state": "COMMENTED"}]
git log --format='%an' -8 origin/main   ->  8x "Cody Clayton"
```

Every factory role acts through **one** GitHub account. Therefore GitHub review-API identity is
**necessary but not sufficient**: it authenticates *repository, PR, and `commit_id`*, and cannot
authenticate *which factory role performed the review*.

**This is precisely why §3.2's kernel-stamped `task_runs.profile` is the trust anchor.** The Kanban
run record is the only place where role is established by the dispatcher rather than asserted by
the actor. The security design states the same conclusion independently:

> *"Kanban run profile/claim records, not artifact fields or comments, establish role."*

Division of authority, now settled:

| Fact | Authenticated by |
|---|---|
| repository identity, PR number, head SHA | GitHub API (numeric IDs) |
| **factory role** (`code_review` / `qa` / `security`) | **kernel-stamped `task_runs.profile`** |
| artifact content | SHA-256 recomputed from git objects by the broker |
| the binding of all of the above | DSSE/in-toto attestation, asymmetric KMS/HSM key |

`repo_numeric_id` remains **absent from `task_runs` in v1.0.0**: the broker resolves it from GitHub
at attestation time (§3.3) and binds it in the attestation. Adding it to the run row would be a
`provenance_version` bump, and `t_8513bc6e` does not require run-level persistence.

---

## §8 — Rejected alternatives

| Alternative | Why rejected |
|---|---|
| **JSON envelope in `task_runs.metadata`** | Re-creates the parsing surface being eliminated, one layer deeper. No CHECK constraints, no FK, no uniqueness. |
| **Fixed `artifact_1..artifact_n` columns** | Cannot express variable artifact counts; forces either truncation or a schema change per new artifact. |
| **A run-level `final_candidate_sha` column** | Only knowable after the run is terminal, so writing it would require the mutation §5 prohibits, and it invites a false equivalence with the head the run actually verified. The run carries `subject_sha` + `verified_head_sha` (§4.1); the broker derives the final candidate SHA. |
| **Storing only `subject_sha` on the run** | Rejected after `t_5247914a` §4/§7. One SHA makes the *"PR head ≠ requested head"* tamper check unevaluable at the run layer. Both **witnessed** SHAs are carried on every run (§4.1). |
| **`board_slug` as a provenance field** | Measured, not assumed: every repository on this host maps to exactly one board (`account-gen` → 1 board, `hermes-agent` → 1 board; no repo appears under two boards). Each board additionally has its own `kanban.db` file, so a board column inside that file is a constant. `repository_id` (immutable numeric) is sufficient identity scoping. **No `board` or `board_slug` column appears in any table in this contract**, and `verify_adr0007_mechanisms.py` asserts its absence (N2). Revisit only if multi-board-per-repo becomes real. |
| **`sha GLOB '[0-9a-f]*'` for hex validation** | Anchors only the first character; `'a'+'Z'*39` passes. Superseded by `NOT GLOB '*[^0-9a-f]*'` (§2.1, §2.2), demonstrated in `verify_adr0007_mechanisms.py`. |
| **`role` as a stored column** | Forces a migration whenever a profile is reclassified, and creates a second place role can disagree with the dispatcher record. Derived at query time from protected-base policy instead (§2.3). |
| **Broker-facing pull API / tunnel** | `t_5247914a` §3a: inverts the trust direction and couples fail-closed behaviour to laptop uptime. Push-per-run with an exporter-side cursor instead (§6). |
| **Commit-count-based role independence** | Commit count is transport, not identity. Three commits by one actor prove nothing about independence. |
| **Branch name as identity anchor** | Worker-influenced and mutable — same class as trusting caller-supplied role JSON. |
| **UPDATE-in-place corrections** | Destroys the audit trail and lets consumed provenance change under an emitted attestation. |

---

## §9 — Migration and compatibility

1. All new columns are **additive and nullable-on-existing-rows**. Existing rows read as
   `provenance_version = NULL` → treated as *legacy, unattestable*.
2. **No backfill.** A backfilled subject SHA would be inferred, not captured — inferred provenance
   is exactly what this contract removes. Legacy runs are excluded from export rather than
   retroactively blessed.
3. `run_artifacts` is a new table; absence is indistinguishable from "no artifacts recorded", so the
   evaluator requires `provenance_version >= 1` before asserting anything about artifacts.
4. Rollback: drop `run_artifacts`, drop the four added columns. No existing column changes type or
   nullability, so rollback cannot corrupt pre-existing rows.

**Blast radius:** schema change to the kernel DB used by every board and every bot. Requires a
migration path and a backup verified by checksum before application. This is a
`factory-director` apply decision, not an implementer's.

---

## §10 — Acceptance tests (executable)

**Numbering is stable and global across v1.0.0 and v1.1.0.** A1–A18 below are owned by this
document. v1.1.0 §C.1 adds A19 and upward; it does **not** renumber anything here.

| # | Test | Expected |
|---|---|---|
| A1 | Worker attempts to write `role` or `profile` on its own run | rejected; kernel value unchanged |
| A2 | Worker supplies `{path, sha256}`; kernel stamps `created_at` | row present, hash matches recomputed digest |
| A3 | Same profile produces `code_review` and `security` terminal runs | evaluator **FAIL** (§3.2) |
| A4 | Three distinct profiles across the three required roles | evaluator **PASS** |
| A5 | `subject_sha` / `verified_head_sha` hostile values: `'a'+'Z'*39`, `'a'*39+'g'`, `'A'*40`, `'aB'*20`, 39 chars, 41 chars, embedded space | CHECK rejects **every** case (§2.1; `GLOB '[0-9a-f]*'` would accept the first two) |
| A5b | `run_artifacts.sha256` hostile values: `'c'*63+'Z'`, `'c'+'Z'*63`, `'C'*64`, 63 chars | CHECK rejects every case (§2.2) |
| A6 | UPDATE any field on a terminal run | rejected — see v1.1.0 §B for the enforcing mechanism (a documented prohibition alone does not reject it) |
| A7 | Two correction rows target the same `corrects_run_id` | broker **fails closed**, no export |
| A8 | `corrects_run_id` points at a non-existent row | broker **fails closed** |
| A9 | Duplicate `(run_id, artifact_path)` | UNIQUE violation |
| A10 | Re-export an already-watermarked run | no-op, no duplicate attestation |
| A11 | Run with `provenance_version = NULL` (legacy) | excluded from export, not an error |
| A12 | Export payload inspected for `summary` / `error` / `body` | absent (§6.1) |
| A13 | Export payload inspected for any absolute path (`/Users/…`) | absent; only repo-relative `artifact_path` (§6.1) |
| A14 | Implementation run: `subject_sha` vs `verified_head_sha` | equal; **not** special-cased, both populated (§4.1) |
| A15 | Review run where branch advanced after review | `verified_head_sha != subject_sha`; broker fails closed on `PR head != expected_head_sha` |
| A16 | Same `(repository_id, task_id, run_id)` resubmitted with a different outcome | mirror **rejects**; first write wins (§6) |
| A17 | Exporter restarted after downtime | resumes from its own cursor; no duplicate envelopes, no gap |
| A18 | Profile absent from the protected-base `profile → role` map | role unresolved → gate **FAIL**, never defaulted (§2.3) |

v1.1.0 §C.1 continues this table at **A19**. A13–A18 above remain in force and are **not**
superseded by it.

---

## §11 — Provenance of this document

Design constraints in §3.2, §3.3 and §4 were supplied by `platform-engineer` (card `t_8c86298c`
consultation) and **corrected two errors** in the author's earlier model: commit-level rather than
run-level independence, and a final-candidate SHA held as a run column. Both are recorded in §8 as
rejected alternatives so the reasoning survives the decision.

Schema facts in §1 were read from the live database, not recalled.

**Security review is a required next gate and has not occurred.** At time of writing,
`security-reviewer` returns HTTP 402 provider billing errors rather than opinions (observed on
`t_bde9863c`, escalated to `factory-director`). This contract must not be treated as
security-approved.
