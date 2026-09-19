# Contract correction: Structured Run Provenance for Hermes Kanban — v1.1.0

- **Version:** 1.1.0 (corrects v1.0.0-draft, attachment id 4,
  sha256 `38d66a972800f501b6a6e776ea3076b0d1d12e51804a64c4171e9426ced741bd`)
- **Status:** Proposed — security review is still a required, un-run gate
- **Author:** `solution-architect`, card `t_8c86298c`, run 8
- **Revision:** review corrections B1–B4 / N1–N2 applied on card
  `t_81217a6a`, against reviewed head `1ba7a03a` of PR #91194
- **Scope:** v1.0.0 remains in force **except** where §A and §B below
  supersede it. §A.4.1 enumerates the superseded v1.0.0 text exactly.

This is a correction, not a replacement. v1.0.0 §0–§3, §5–§6, §9–§11
stand as written; §2.1, §2.2, §4, §6.1, §8 and §10 are amended only where
§A.4.1 lists them.

Two substantive changes were made in the original v1.1.0, each for a
stated cause: one is a binding director ruling that closes a seam v1.0.0
deliberately left open (§A); the other is an enforceability defect found
in the live source after v1.0.0 was frozen (§B).

Four further corrections were then applied after formal code review of
PR #91194 at `1ba7a03a` (findings B1–B4, N1–N2 on card `t_7b608af5`):

| Finding | Resolution |
|---|---|
| B1 — ruff/UTF-8 errors in the verifier | verifier rewritten; clean under the repo ruff config and under `--select E,F,I,W` |
| B2 — `GLOB '[0-9a-f]*'` validated only the first character | §A.5: negated-GLOB CHECKs, with hostile cases executed |
| B3 — `final_candidate_sha` contradiction across §2.1/§4.1/§4 | §A.4: one normative three-SHA model; §A.4.1 enumerates superseded text |
| B4 — §C.1 renumbering collided with v1.0.0 A13–A18 | §C.1: v1.1.0 additions start at A19; v1.0.0 A1–A18 preserved |
| N1 — verifier docstring cited a nonexistent §5.5 | corrected to §A.1 |
| N2 — verifier `board` column contradicted v1.0.0 §8 | §A.6: column removed; absence asserted |

**Contract source for the security design:** attachment id 3,
`security-design-t_8513bc6e-comment-573.md` (12,212 bytes). Verified
this run: byte-identical to live account-gen comment 573 after
whitespace normalization, differing only by a 67-character provenance
header. All "comment 573 §N" references resolve to that attachment, not
to a cross-board CLI lookup.

**Executable evidence:** `verify_adr0007_mechanisms.py` (this directory),
85/85 checks passing. See §C.

---

## §A — Repository binding: seam closed (supersedes v1.0.0 §3.3, §7, §7.1)

v1.0.0 §7 stated that whether the attestation must bind numeric repo ID
and PR number "belongs to the security trust design", left
`repo_numeric_id` absent, and specified that adding it "is a
`provenance_version` bump". **factory-director has now ruled the
requirement binding.** This document is that version bump; v1.0.0's own
mechanism is being followed, not overridden.

The ruling, applied:

- The DSSE attestation MUST bind the immutable numeric GitHub repository
  ID and the PR number, alongside trust-root commit, subject SHA, final
  candidate SHA, policy version, artifact hashes, run/task IDs, and
  authenticated profiles/outcomes.
- These MUST NOT be parsed from comments, summary, or worker metadata.

### §A.1 The nullable-vs-required rule

v1.0.0 §3.3 rejected persisting a PR number because "a PR may not exist
when the run ends; requiring it would force either a nullable field
everyone treats as optional, or a write after terminalization". That
objection (originally platform-engineer's) is correct and is **not**
discarded. It is resolved by making the requirement *conditional on
export finalization* rather than on run terminalization.

Two columns and one flag:

```sql
repo_github_id   INTEGER,  -- immutable numeric GitHub repository id
event_locator    TEXT,     -- 'pr:<number>' for PR gates; NULL otherwise
export_finalized INTEGER NOT NULL DEFAULT 0
```

- **General local Kanban runs** (`export_finalized = 0`):
  `repo_github_id` and `event_locator` MAY be NULL. Most factory work is
  not a PR gate — research spikes, docs cards, scratch analysis — and
  must not be forced to invent repository identity it does not have.
  Such a run is simply never exported. This is the common case.
- **Runs finalized for broker export** (`export_finalized = 1`): ALL of
  `repo_github_id`, `event_locator` (the PR number for PR gates),
  `subject_sha`, `verified_head_sha`, and every artifact `sha256` are
  mandatory and non-NULL. Any missing, malformed, or ambiguous value
  means the record is **not finalized and not exported**. Fail closed,
  never fail open.

Finalization is a **distinct, later step** than terminalization, run by
the operator/exporter once the PR actually exists. This is what
dissolves the original objection: nothing is written to a terminal run,
and no field is nullable-but-secretly-required. A run that terminates
before its PR exists is simply not yet finalized.

Because v1.0.0 §5 makes terminal rows immutable, finalization is
implemented as an **insert** of a finalization record keyed to the run,
not an UPDATE of the run row. A record that never finalizes is inert,
which is the safe default.

### §A.2 Provenance of the two new fields

Neither is worker-writable, and neither is parsed from prose.

- `repo_github_id` — resolved by the kernel from the workspace `origin`
  remote via an authenticated GitHub API lookup, cached per repository.
  If the lookup fails or is ambiguous the field stays NULL and the run
  is not finalizable. It is **never** derived from the remote URL
  string, because a remote URL is renameable and therefore not immutable
  identity; comment 573 requires the immutable numeric id specifically.
  This preserves v1.0.0 §3.3's "branch name is never an authority"
  principle and extends it to remote strings.
- `event_locator` — supplied at finalization. It remains a **locator,
  not authority**, exactly as v1.0.0 §3.3 framed repository identity.
  The broker still independently resolves and validates it against
  GitHub and still requires exact PR-head equality (comment 573 §4.1).

### §A.3 What did NOT change

The seam closed here is repository binding only. v1.0.0's trust semantics
(§3.1, §3.2), role derivation (§2.3), export exclusions (§6.1), and the
append-only correction model (§5) are untouched.

### §A.4 `final_candidate_sha`: the single normative model (resolves B3)

**This subsection is normative and overrides every other statement about
`final_candidate_sha` in either document.** The reviewed revision of v1.0.0
was internally inconsistent: §2.1 listed `final_candidate_sha` as a
`task_runs` column, §4.1 instructed implementers to carry it on every run,
and the closing paragraph of §4 said it is *not* a run field. An
implementer could have built either schema from that text.

**There are three SHAs, at two layers.**

| SHA | Layer | Storage | Written by |
|---|---|---|---|
| `subject_sha` | Kanban run | `task_runs.subject_sha` (40-char lowercase hex) | kernel, at claim |
| `verified_head_sha` | Kanban run | `task_runs.verified_head_sha` (40-char lowercase hex) | kernel, at terminalization |
| final candidate SHA | DSSE attestation | **no Kanban column** | broker, at attestation time |

Normative rules:

1. Kanban stores exactly the two SHAs the kernel can **honestly witness**:
   what the run started from, and what it had verified when it closed.
   Both are populated on **every** run type; for implementation runs they
   are equal, and that equality is not special-cased.
2. **No table in this contract may define a column named
   `final_candidate_sha` or `final_sha`.** The final candidate SHA is
   derived and bound by the broker at attestation time. A Kanban column of
   that name would assert authority the kernel does not have, and — since
   it is knowable only after the run is terminal — writing it would require
   the very post-terminal mutation v1.0.0 §5 forbids.
3. The broker compares its derived final candidate SHA against the exported
   `verified_head_sha` and **fails closed** on disagreement. That
   comparison is the stale-evidence tamper check; it is only possible
   because the run carries a witnessed head at all.
4. Divergence between `subject_sha` and `verified_head_sha` on a review /
   QA / security run is **legitimate** (the branch advanced), and is
   evidence, not a defect.

Acceptance test A25 below guards rule 2 against a future implementer
helpfully adding the column back; the verifier asserts it directly.

#### §A.4.1 Exactly which v1.0.0 text is superseded

Enumerated so nothing is left to inference:

| v1.0.0 location | Superseded text | Replaced by |
|---|---|---|
| §2.1 column table | the `final_candidate_sha` row and the `role` row | `verified_head_sha` row; `role` stays derived per §2.3 |
| §2.1 CHECK block | `CHECK (subject_sha ... GLOB '[0-9a-f]*')` and `CHECK (role IN (...))` | negated-GLOB CHECKs on `subject_sha` and `verified_head_sha` (see §A.5) |
| §2.2 | `CHECK (length(sha256) = 64 AND sha256 GLOB '[0-9a-f]*')` | `... AND sha256 NOT GLOB '*[^0-9a-f]*'` (§A.5) |
| §4 heading + comparison table | two-column "Subject SHA vs Final candidate SHA" table storing `final_candidate_sha` on the run | three-SHA table in §4, matching the table above |
| §4.1 | every occurrence of `final_candidate_sha` as a run field | `verified_head_sha` |
| §4 closing paragraph | *"Why final candidate SHA is not a run field"* stated **after** §2.1/§4.1 said it was one | §4's *"Why the final candidate SHA is not a run column"*, now the only claim in force |
| §6.1 export field list | `final_candidate_sha` in the exported envelope | `verified_head_sha` |
| §8 | row *"Final candidate SHA as broker-derived only"* (which read as rejecting the model now adopted) | two rows: *"A run-level `final_candidate_sha` column"* (rejected) and *"Storing only `subject_sha`"* (rejected) |
| §10 A14, A15 | `final_candidate_sha` in the expectations | `verified_head_sha` |

Everything else in v1.0.0 §4 — the reasoning for capturing the subject at
claim, and for carrying two SHAs rather than one — is **retained
unchanged**. This corrects the naming and the layer, not the analysis.

### §A.5 Hex validation must check every character (resolves B2)

v1.0.0's CHECK constraints used `sha GLOB '[0-9a-f]*'`. In SQLite that
pattern anchors only the **first** character: the trailing `*` then matches
any 39 remaining characters, uppercase and non-hex included. Demonstrated
executably (§C): `'a' + 'Z'*39` and `'a'*39 + 'g'` are both **accepted**.

The normative form negates the complement class instead, so a character
outside `[0-9a-f]` at *any* position aborts the write:

```sql
CHECK (subject_sha IS NULL OR (length(subject_sha) = 40
       AND subject_sha NOT GLOB '*[^0-9a-f]*'))
CHECK (verified_head_sha IS NULL OR (length(verified_head_sha) = 40
       AND verified_head_sha NOT GLOB '*[^0-9a-f]*'))
CHECK (length(sha256) = 64 AND sha256 NOT GLOB '*[^0-9a-f]*')
```

The explicit `length()` term is still required — GLOB alone cannot express
"exactly 40". Hostile inputs are enumerated in v1.0.0 §10 A5/A5b and are
executed by the verifier against both the SQL CHECKs and the Python
`SHA40`/`SHA256` regexes, so the two layers cannot drift apart.

### §A.6 No `board` column anywhere (resolves N2)

The reviewed `verify_adr0007_mechanisms.py` declared
`board TEXT NOT NULL` on `run_provenance` and carried `"board":
"hermes-agent"` in the hashed record body. That **contradicted** v1.0.0 §8,
which rejects board slug as a provenance field.

Resolved in favour of §8: **the column is removed**, from the verifier DDL
and from the record body it digests. Two reasons, both structural rather
than stylistic:

1. Kanban is already sharded per board on disk
   (`~/.hermes/kanban/boards/<board>/kanban.db`), so a `board` column inside
   one of those files stores a value that is constant for the whole file.
   It is a label, not an identifier.
2. §8's measurement stands: every repository on this host maps to exactly
   one board, so board slug adds no identity scoping that
   `repo_github_id` does not already provide — while adding a
   human-renameable string to a record whose purpose is immutable identity.

Consequently the §D open item *"whether board slug is acceptable in the
export payload or must be an opaque id"* is **withdrawn, not deferred**:
no board slug is exported, so security-reviewer has no such question to
answer. Acceptance test A26 and a verifier check assert the absence.

---

## §B — Immutability needs an enforcement mechanism (corrects v1.0.0 §5, A6)

**This is a defect in v1.0.0, found by reading the live source after the
document was frozen.** v1.0.0 §5 asserts "once `status` is terminal and
`ended_at` is set, no field on that row may be UPDATEd", and test A6
expects "UPDATE any field on a terminal run → rejected". v1.0.0 names no
mechanism that would reject it, and in the current kernel nothing does.

Source anchors, inspected at commit `fbb4454ed` in
`/Users/dreddy/.hermes/hermes-agent`:

- `edit_completed_task_result` (`hermes_cli/kanban_db.py:6181-6243`)
  UPDATEs `task_runs.summary` at `:6220-6223` and `task_runs.metadata`
  at `:6224-6228` on a run whose `outcome` is already `'completed'`.
  A6 fails today.
- `_end_run` (`:4350-4374`) UPDATEs the run row at terminalization, and
  the reclaim path in `claim_task` (`:4662-4677`) UPDATEs it again.

So `task_runs` is a legitimately-mutable table with at least three live
UPDATE paths, one of which fires *after* completion. Placing an
immutability guarantee on that table is unenforceable by assertion.

**Correction.** The immutable terminal provenance record MUST live in
its own append-only table protected by SQLite triggers, rather than
relying on a documented prohibition against updating `task_runs`:

```sql
CREATE TRIGGER trg_run_provenance_no_update BEFORE UPDATE ON run_provenance
BEGIN SELECT RAISE(ABORT, 'run_provenance is append-only'); END;
CREATE TRIGGER trg_run_provenance_no_delete BEFORE DELETE ON run_provenance
BEGIN SELECT RAISE(ABORT, 'run_provenance is append-only'); END;
```

Artifact rows carry a `sealed` flag flipped by the kernel at
terminalization, with a `BEFORE UPDATE ... WHEN OLD.sealed = 1` trigger.
Rows stay editable while the run is live (so a worker may re-declare
before terminalizing) and become immutable the moment the run closes.

v1.0.0's `corrects_run_id` append-only correction model (§5) is
**unchanged and still correct**; this only adds the mechanism that makes
"no UPDATE" true rather than merely stated.

**Residual risk, stated plainly:** a local actor with write access to
`kanban.db` can drop a trigger. SQLite has no in-database privilege
separation. Triggers raise the cost of tampering and make casual or
programmatic mutation fail loudly; they are not a defense against root
on the box. Comment 573 already places canonical dispatcher records
inside the trust boundary, so this does not widen it — and the broker's
independent recomputation of every artifact digest from the git object
(573 §4) is what actually catches a doctored record. **This specific
tradeoff needs security-reviewer sign-off** (§D).

### §B.1 Export cursor

A consequence worth stating, because it is easy to get wrong: the export
watermark must be monotonic in **terminalization order**, which
`task_runs.id` is not — runs terminate out of creation order. The
append-only provenance table's own AUTOINCREMENT `seq` provides this
correctly. Verified empirically (§C): runs 99 and 50 terminating in that
order receive ascending `seq` 5 and 6.

This refines v1.0.0 §6's "highest exported `task_events.id`", which is
directionally right but keys off a table that also receives unrelated
events. Per research-scout, the cursor is **local-exporter-only** and is
never exposed as a broker-facing pull surface: canonical Kanban is
local-only and the broker must never reach into it.

---

## §C — Executable verification performed

`verify_adr0007_mechanisms.py` (this directory) exercises these mechanisms
against real SQLite and real git, so the claims above are demonstrated
rather than asserted:

```
python3 verify_adr0007_mechanisms.py     ->  85/85 checks passed
```

The count rose from 36 to 85 in this revision: the B2 correction added
hostile-input cases against both SQL CHECKs and the Python regexes, and
B3/N2 added structural assertions about which columns must and must not
exist.

**The verifier's SQL is a fixture, not the contract.** Its `CREATE TABLE`
statements are illustrative scaffolding chosen to exercise the mechanisms
under test; they carry additional columns that this contract does not
declare and does not require. Only the columns declared normatively — §2.2
(`id`, `run_id`, `artifact_path`, `sha256`, `created_at`) plus `sealed`
(§B) — are part of the contract. An assertion that compared the fixture's
schema to a hardcoded copy of itself would restate the fixture rather than
verify the contract, and no such assertion is made.

Proven, in order of the assertions in this document:

- A row is written for **every** terminal outcome (completed, blocked,
  crashed) — absence of a record can never itself be read as evidence;
  only completed+SHAs+artifacts is `attestable=1`.
- **§A.5 specifically (B2):** the superseded `GLOB '[0-9a-f]*'` pattern is
  shown **accepting** `'a'+'Z'*39` — the defect, demonstrated rather than
  described — while the normative `NOT GLOB '*[^0-9a-f]*'` CHECKs reject
  `'a'+'Z'*39`, `'a'*39+'g'`, `'A'*40`, `'aB'*20`, 39 chars, 41 chars and
  an embedded space, on **both** `subject_sha` and `verified_head_sha`,
  and reject the 64-char equivalents on `run_artifacts.sha256`. Valid
  full-length lowercase hex is still accepted in each case. The Python
  `SHA40`/`SHA256` regexes are run over the same hostile table so the two
  validation layers cannot drift.
- UPDATE and DELETE on the provenance table abort with
  `sqlite3.IntegrityError: run_provenance is append-only`, and the
  record survives the tamper attempt byte-identical (§B).
- Sealed artifact rows reject writes; unsealed rows remain editable.
- Duplicate `run_id` rejected; duplicate `(run_id, path)` rejected.
- `seq` strictly ascending and in a different order from `run_id`,
  proving the cursor tracks terminalization (§B.1).
- Re-export from a watermark yields nothing; export digests unique.
- A `mode=ro` connection raises `attempt to write a readonly database`,
  confirming v1.0.0 §6's read-only exporter requirement is achievable.
- Artifact digest is stable under input ordering and changes when any
  hash changes.
- A real file digest differs from a worker-claimed one — i.e. the kernel
  must compute, never accept, the hash.
- `git rev-parse` yields full 40-hex for HEAD and for tracked blobs; a
  non-git scratch dir yields nothing, so scratch runs are structurally
  non-attestable.
- **§A.1 specifically:** a complete gated run finalizes; a non-gated run
  with NULL repo fields is legal but never finalizes; finalization fails
  closed *naming the offending field* for each of the five mandatory
  fields individually; a remote-string repo id, a malformed event
  locator, an abbreviated SHA, and an unresolved correction chain are
  each refused. Every hostile SHA above is also refused by the
  finalization gate, not only by the SQL layer.
- **§A.4 specifically (B3):** no `final_candidate_sha` / `final_sha`
  column exists, and both `subject_sha` and `verified_head_sha` do.
- **§A.6 specifically (N2):** no `board` / `board_slug` column exists.

This is a mechanism probe against a model of the schema. It is **not** a
substitute for the acceptance tests, which must run against the real
kernel and belong to the implementer.

### §C.1 Additional acceptance tests (extend v1.0.0 §10)

**v1.0.0 A1–A18 all stand unchanged and are NOT renumbered.** A previous
revision of this section reused the numbers A13–A20, colliding with v1.0.0
A13–A18 and silently displacing six still-valid tests (absolute-path
exclusion, implementation-run SHA equality, review-run SHA divergence,
idempotent resubmission, exporter restart, and the profile→role map).
That collision is corrected here: **v1.1.0 additions start at A19.**

Numbering is global across both documents. A new test appends to the end;
no existing number is ever reused for a different test.

| # | Test | Expected |
|---|---|---|
| A19 | Non-gated run with NULL `repo_github_id`/`event_locator` | completes normally, `export_finalized=0`, never exported |
| A20 | Finalization attempted with any one mandatory field missing | refused, fails closed, offending field named |
| A21 | `origin` remote renamed | `repo_github_id` unchanged; remote string never accepted as identity |
| A22 | Finalization performed | original provenance row byte-identical, digest unchanged |
| A23 | Direct UPDATE/DELETE on provenance table | raises `sqlite3.IntegrityError` |
| A24 | `edit_completed_task_result` called on a completed run | `summary`/`metadata` may change; provenance record digest **unchanged** |
| A25 | Schema inspected for `final_candidate_sha` / `final_sha` | absent (guards §A.4 rule 2) |
| A26 | Schema inspected for `board` / `board_slug` | absent (guards v1.0.0 §8; see §A.6) |
| A27 | Runs terminating out of `run_id` order | export `seq` still ascending |

A24 is the regression test for the §B defect specifically: the existing
post-completion mutation path must remain functional for its own purpose
while being provably unable to touch provenance.

Cross-reference for readers of the superseded numbering. **These are not
test definitions** — the left column is a retired v1.1.0-draft label, the
right column is the number now in force (defined in the table above):

| retired draft label | number now in force |
|---|---|
| draft-A13 | A19 |
| draft-A14 | A20 |
| draft-A15 | A21 |
| draft-A16 | A22 |
| draft-A17 (final_sha guard) | A25 |
| draft-A18 (`edit_completed_task_result`) | A24 |
| draft-A19 (UPDATE/DELETE) | A23 |
| draft-A20 (seq ordering) | A27 |

The labels `A13`–`A18` without the `draft-` prefix belong to **v1.0.0 §10**
and always did; that is the collision this correction removes.

A26 is new in this correction (N2). v1.0.0 A5b is likewise new (B2).

---

## §D — Open items owned by others

- **security-reviewer** (required next gate, has NOT occurred): sign-off
  on §B's trigger-based immutability and its stated residual risk;
  sign-off on §A.1's nullable-vs-required split.
- **platform-engineer**: implementation.
- **broker / research-scout**: final candidate SHA derivation stays
  broker-side, out of scope here (§A.4).

Resolved, no longer open: repository id and PR number (§A, director
ruling); DSSE source reference (attachment id 3, verified above); board
slug in the export payload (§A.6 — withdrawn, nothing to sign off);
`final_candidate_sha` storage model (§A.4); hex CHECK strictness (§A.5).

## §E — Limitations of this document

1. **Not security-approved.** v1.0.0 §11's warning stands unchanged.
2. §C is a probe against a schema model, not the live kernel. The
   §C.1/§10 tests must be run against the real kernel before
   implementation is considered verified. The B2 corrections narrow this
   gap for hex validation only: the CHECK constraints are now exercised
   as real SQLite CHECKs, but still on a model table.
3. `repo_github_id` resolution assumes an authenticated GitHub API path
   exists in the kernel for that lookup. I did not verify one exists;
   if it does not, that is implementation work platform-engineer must
   scope, and until then no run is finalizable.
4. The §B residual risk (trigger-droppable by a local root actor) is
   accepted-by-default here and explicitly routed to security-reviewer.
5. No change was made to Hermes source, the live Kanban DB, or any
   profile. This revision touches only the three documents in
   `docs/design/` named in §A.4.1 and this file's own header.
