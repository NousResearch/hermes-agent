# Hermes Skill Contract Schema

**Status**: design only. No live skill files or config are touched by this document.
**Input**: `docs/design/hermes-skill-ops-synthesis.md` (synthesis of SkillOps, arXiv:2605.13716,
and Skill-V, arXiv:2610.11781).
**Schema file**: `docs/design/skill-contract.schema.json` (JSON Schema draft 2020-12).

## Why this shape

The synthesis note's headline finding is that the two source papers disagree on whether a skill
contract is *versioned and falsifiable* at all:

- SkillOps' `(P, O, A, V, F)` is unversioned and graph-embedded — great for cross-skill
  dependency/compatibility checking, but there is no mechanism to ask "is this specific contract
  still correct, and compared to what prior version?"
- Skill-V's `(φ, p, ρ, ν)` is versioned (`ν`) and the outcome judgment `ρ` is an **executable
  rubric** replayed against stored evidence (the Evidence Gate, Eq. 7) — which is exactly the
  "objectively testable" property this task requires.

This schema ports Skill-V's shape as the backbone (`applicability` ≈ `φ`, `outcome_criteria` ≈
`ρ`, `lineage`/`version` ≈ `ν`) and keeps SkillOps' discipline of making every field a *check*,
not prose (`preconditions` is new relative to Skill-V but mirrors SkillOps' `P` and the
"executable, not descriptive" principle from §2). It then adds three things neither paper has,
because the synthesis note explicitly flags them as gaps:

1. **Difficulty-bucketed evidence** (`outcome_criteria.evidence_scope.difficulty_bucket`) — neither
   paper normalizes outcome metrics for task difficulty (synthesis §b).
2. **A minimum-sample guard plus archive-first retirement** (`retirement_state`,
   `evidence_scope.min_sample_per_bucket`) — SkillOps retires via an unverifiable threshold with no
   sample-size floor; Skill-V has no retirement mechanism at all (synthesis §d).
3. **A `reviewed_by` lineage field** acknowledging that Hermes may want a human decision aid rather
   than the fully-automated gates both papers use — neither paper has any human-in-the-loop step
   (synthesis §f). The schema does not *require* a human reviewer (default is automated), but it
   gives the field a place to live so a human override is auditable, not bolted on later.

## Field-by-field: what it means and how it's tested

### `skill_id` + `version`
A skill's identity (`skill_id`) never changes across revisions. `version` splits into
`skill_version` (the procedure/scope) and `rubric_version` (the outcome criteria) because Skill-V's
own worked examples (Table 8) show these incrementing independently — a skill's procedure can stay
at v1 while its rubric gets tightened twice (v1→v2→v3) in response to false negatives, or vice
versa. Keeping them separate means a consumer can ask "did the *behavior* change" and "did the
*bar for success* change" as two different, independently-auditable questions.

**Version bump policy** (SemVer MAJOR.MINOR.PATCH, applied to each of the two version numbers
independently):

| Bump | `skill_version` triggers | `rubric_version` triggers |
|---|---|---|
| MAJOR | Any change to `applicability.protected_constraints`, OR scope narrows/widens in a way that changes which tasks the skill claims to handle (`applicability.scope.task_types` changes) | A `hard_constraints` entry is added, removed, or has its evidence_matcher changed (changes what "pass" can even mean) |
| MINOR | Procedure change that passes `non_regression` (Evidence Gate) against the existing replay buffer — behavior changed but evidence shows it is not worse | `weighted_criteria` or `pass_threshold` change that passes `non_regression` |
| PATCH | Non-functional edit: wording, typo fix in `description`, documentation-only change, no effect on any check | Same, for outcome_criteria descriptions/ids only |

A MAJOR bump is the only kind allowed to *fail* `non_regression` — it is an explicit admission that
old evidence no longer applies (e.g. the skill's scope changed enough that past trajectories are
off-topic for the new version). MAJOR bumps therefore reset the replay buffer rather than requiring
it to pass; MINOR/PATCH bumps must pass the gate against the *existing* buffer unchanged. This
mirrors Skill-V §3.3: "Accepted revisions replace the contract in place... rejected ones leave the
current version untouched," generalized into a three-tier bump instead of Skill-V's implicit
single-tier accept/reject.

### `preconditions`
Each entry names a `check.kind` from a closed enum (`tool_available`, `env_var_set`, `file_exists`,
`credential_present`, `input_schema_match`, `command_exits_zero`, `capability_flag`). These are
chosen because each one reduces to a single boolean subprocess/stat/env-lookup call — an
orchestrator can evaluate the entire precondition list with zero LLM calls before invoking the
skill, same spirit as SkillOps keeping maintenance "nearly zero library-time LLM calls" (synthesis
§f). `hard: false` preconditions are advisory and can be overridden, but the override must be
logged — this is a design choice to avoid preconditions silently becoming decorative.

### `applicability`
`scope.task_types` + `scope.input_schema_ref` are the positive definition of where a skill applies
(Skill-V's `φ`'s applicability half); `scope.excluded_contexts` is a required (possibly empty)
field forcing authors to enumerate *known* false-positive territory rather than leaving it
implicit — a direct response to the fact that both source papers' failure cases in practice came
from skills firing outside their real competence.

`protected_constraints` are the safety invariants a revision may never silently break (ported
verbatim from Skill-V's `φ_i` "protected semantic constraints," App. B.2). They are testable in
the sense that `non_regression` requires *exact preservation* — a textual/structural diff check,
not a judgment call — and any change to this list forces a MAJOR bump.

### `outcome_criteria`
This is the core "objectively testable" requirement. Every criterion (`hard_constraints` and
`weighted_criteria`) carries an `evidence_matcher.kind` from a closed enum of deterministic
check types:

- `exit_code_zero`, `exact_match`, `regex_match`, `file_hash_equals`, `schema_validates` — pure
  mechanical checks, zero ambiguity.
- `golden_trace_replay` — compare a new trajectory's action sequence/outputs against a stored
  golden trace (used when exact output isn't fixed but the *shape* of success is, e.g. "a PR was
  opened with a non-empty diff and a passing CI check").
- `external_command_assert` — run a fixed shell command/test and check its exit code (e.g. `pytest
  tests/test_contract.py::test_pdf_form_filled`).
- `llm_judge_rubric` — the one necessarily-subjective-sounding kind, made objective by **pinning**
  `params.model`, `params.prompt_ref`, and `params.threshold`: the same trajectory fed through the
  same model/prompt/threshold must yield the same verdict, so it is re-runnable and auditable even
  though it is not a pure string comparison. This is only used where no mechanical check is
  possible (e.g. "is this summary accurate"), and the grading function itself is versioned as part
  of the contract (a `prompt_ref` change requires the same bump-and-regression-check discipline as
  any other criterion change).

`pass_threshold` plus the `hard_constraints`/`weighted_criteria` split mirrors Skill-V's
`ŷ_rub(τ) = 1[q(τ) ≥ η ∧ H(τ)=0]` (Eq. 3) directly: hard constraints are the `H(τ)=0` gate,
weighted criteria aggregate into `q(τ)` compared against `η` = `pass_threshold`.

`evidence_scope.difficulty_bucket` + `min_sample_per_bucket` are the Hermes-specific additions
addressing synthesis §b/§d: pass rates, non-regression checks, and retirement decisions are
computed **per bucket**, and a bucket's numbers are not trusted for any maintenance decision until
it has at least `min_sample_per_bucket` trajectories. This directly fills the gap flagged as
"unaddressed" in both source papers.

`non_regression` is the Evidence Gate (Skill-V Eq. 7), generalized to also require minimum sample
coverage before being invoked (synthesis §d, §c cross-reference): a candidate revision is only
checked against buckets that already meet `min_sample_per_bucket`; a revision touching a bucket
with insufficient evidence cannot claim non-regression for that bucket and must either wait or be
scoped as MAJOR (fresh evidence, see bump policy above).

### `lineage`
Ported directly from Skill-V's `ν_i` ("skill version, rubric version, traceability, update reason,
evidence case identifiers," Table 7). `update_reason` and `evidence_case_ids` are mandatory except
for a skill's very first version — this is the audit trail SkillOps is shown to lack entirely
(synthesis §e: "no per-skill version history is tracked"). `reviewed_by` is optional and defaults
to unset/automated, but exists so that if Hermes chooses to put a human in the loop as a decision
aid over the Evidence Gate output (the recommendation in synthesis §f, since neither paper has any
human review step), that decision has a durable place to be recorded rather than living only in a
chat log.

### `retirement_state`
Novel relative to both papers (synthesis §d: SkillOps hard-deletes on an unverifiable threshold
with no sample floor; Skill-V has no retirement mechanism). This schema makes retirement
**archive-first**: `status` moves `active → candidate_for_retirement → archived`, never disappears,
and `min_sample_met` must be `true` before a `candidate_for_retirement` or `archived` status is
valid — directly closing the "no explicit minimum sample size is given anywhere" gap identified in
the synthesis note.

## Worked examples

### Example 1 — `pdf` skill (fill a PDF form)

```json
{
  "skill_id": "pdf",
  "version": { "skill_version": "1.2.0", "rubric_version": "1.1.0" },
  "preconditions": [
    {
      "id": "pdf-cli-present",
      "description": "A PDF manipulation backend (pypdf/pdftk) is importable in the task's Python env.",
      "check": { "kind": "tool_available", "target": "pypdf" },
      "hard": true
    },
    {
      "id": "input-is-pdf",
      "description": "The target file exists and has a .pdf extension or %PDF- magic bytes.",
      "check": { "kind": "file_exists", "target": "${input.file_path}" },
      "hard": true
    }
  ],
  "applicability": {
    "scope": {
      "task_types": ["pdf.fill-form", "pdf.extract-fields"],
      "input_schema_ref": "pdf-fill-request.schema.json",
      "excluded_contexts": ["encrypted-pdf-without-password", "scanned-image-only-pdf-no-ocr"]
    },
    "protected_constraints": [
      "never writes output outside the task workspace",
      "never flattens/destroys original form field names without an explicit flatten=true request"
    ]
  },
  "outcome_criteria": {
    "hard_constraints": [
      {
        "id": "output-pdf-valid",
        "description": "The written output file is a structurally valid PDF.",
        "evidence_matcher": { "kind": "schema_validates", "target": "${output.file_path}", "expected": "pdf-structure-validator-v1" }
      },
      {
        "id": "no-workspace-escape",
        "description": "No file was written outside the declared task workspace.",
        "evidence_matcher": { "kind": "external_command_assert", "target": "scripts/check_workspace_containment.sh", "params": { "timeout_s": 10 } }
      }
    ],
    "weighted_criteria": [
      {
        "id": "fields-filled-correctly",
        "description": "Every field named in the request maps to the requested value in the output PDF.",
        "weight": 0.7,
        "evidence_matcher": { "kind": "exact_match", "target": "${output.extracted_fields}", "expected": "${input.requested_fields}" }
      },
      {
        "id": "unrequested-fields-unchanged",
        "description": "Fields not mentioned in the request keep their original values.",
        "weight": 0.3,
        "evidence_matcher": { "kind": "file_hash_equals", "target": "${output.unrequested_field_values}", "expected": "${input.original_field_values}" }
      }
    ],
    "pass_threshold": 0.9,
    "evidence_scope": { "difficulty_bucket": "standard", "min_sample_per_bucket": 8 },
    "non_regression": { "metrics": ["disc_rate", "false_negative_rate", "balanced_accuracy"], "replay_buffer_size": 32 }
  },
  "lineage": {
    "previous_version": "1.1.0",
    "update_reason": "Tightened fields-filled-correctly to exact_match after a false pass on whitespace-padded values (evidence case ec-pdf-0091).",
    "evidence_case_ids": ["ec-pdf-0091"],
    "reviewed_by": "automated-gate-only"
  },
  "retirement_state": { "status": "active" }
}
```

### Example 2 — `github` skill (open a PR via `gh`)

```json
{
  "skill_id": "github",
  "version": { "skill_version": "2.0.0", "rubric_version": "2.0.0" },
  "preconditions": [
    { "id": "gh-cli-present", "description": "gh CLI is installed and on PATH.", "check": { "kind": "tool_available", "target": "gh" }, "hard": true },
    { "id": "gh-authed", "description": "gh auth status succeeds for the target repo.", "check": { "kind": "command_exits_zero", "target": "gh auth status" }, "hard": true },
    { "id": "repo-write-access", "description": "The authenticated identity has push/PR-create rights on the target repo.", "check": { "kind": "credential_present", "target": "github.repo_write_scope" }, "hard": true }
  ],
  "applicability": {
    "scope": {
      "task_types": ["github.open-pr", "github.push-branch"],
      "excluded_contexts": ["repo-requires-signed-commits-without-gpg-key", "protected-branch-direct-push"]
    },
    "protected_constraints": [
      "never force-pushes to a branch it did not create",
      "never merges a PR itself — opening only, merge requires separate human/reviewer action"
    ]
  },
  "outcome_criteria": {
    "hard_constraints": [
      { "id": "pr-exists", "description": "A PR with a non-empty diff exists at the returned URL.", "evidence_matcher": { "kind": "external_command_assert", "target": "gh pr view ${output.pr_url} --json number", "params": { "timeout_s": 15 } } },
      { "id": "no-merge-performed", "description": "The PR state is OPEN, never MERGED, immediately after the skill runs.", "evidence_matcher": { "kind": "exact_match", "target": "${output.pr_state}", "expected": "OPEN" } }
    ],
    "weighted_criteria": [
      { "id": "ci-triggered", "description": "At least one CI check run was created on the PR head commit.", "weight": 0.4, "evidence_matcher": { "kind": "external_command_assert", "target": "gh pr checks ${output.pr_url}", "params": { "timeout_s": 15 } } },
      { "id": "description-matches-request", "description": "PR title/body reflect the task's requested change summary.", "weight": 0.6, "evidence_matcher": { "kind": "llm_judge_rubric", "target": "${output.pr_body}", "params": { "model": "claude-sonnet-5", "prompt_ref": "rubrics/pr-description-fidelity-v1.md", "threshold": 0.8 } } }
    ],
    "pass_threshold": 0.75,
    "evidence_scope": { "difficulty_bucket": "standard", "min_sample_per_bucket": 10 },
    "non_regression": { "metrics": ["disc_rate", "balanced_accuracy"], "replay_buffer_size": 40 }
  },
  "lineage": {
    "previous_version": "1.4.0",
    "update_reason": "MAJOR bump: added protected_constraints forbidding self-merge after an incident where the skill merged its own PR without review (evidence case ec-gh-0007). Replay buffer reset per MAJOR-bump policy since old evidence predates the constraint.",
    "evidence_case_ids": ["ec-gh-0007"],
    "reviewed_by": "human:release-eng"
  },
  "retirement_state": { "status": "active" }
}
```

### Example 3 — `arxiv` skill (search arXiv by keyword/author/category)

```json
{
  "skill_id": "arxiv",
  "version": { "skill_version": "1.0.1", "rubric_version": "1.0.0" },
  "preconditions": [
    { "id": "network-egress-allowed", "description": "Outbound HTTPS to export.arxiv.org is permitted in this environment.", "check": { "kind": "capability_flag", "target": "network.https_egress" }, "hard": true }
  ],
  "applicability": {
    "scope": {
      "task_types": ["arxiv.search", "arxiv.fetch-by-id"],
      "excluded_contexts": ["non-arxiv-preprint-server", "full-text-pdf-parsing"]
    },
    "protected_constraints": ["never fabricates a paper ID or abstract when the API returns zero results"]
  },
  "outcome_criteria": {
    "hard_constraints": [
      { "id": "results-schema-valid", "description": "Returned result list validates against the arxiv-result.schema.json contract.", "evidence_matcher": { "kind": "schema_validates", "target": "${output.results}", "expected": "arxiv-result.schema.json" } },
      { "id": "no-fabricated-ids-on-empty", "description": "If the live API returned zero hits, output.results is an empty array, never invented entries.", "evidence_matcher": { "kind": "exact_match", "target": "${output.results_when_api_empty}", "expected": "[]" } }
    ],
    "weighted_criteria": [
      { "id": "relevance", "description": "Top-5 returned results are topically relevant to the query.", "weight": 1.0, "evidence_matcher": { "kind": "llm_judge_rubric", "target": "${output.results[:5]}", "params": { "model": "claude-sonnet-5", "prompt_ref": "rubrics/arxiv-relevance-v1.md", "threshold": 0.7 } } }
    ],
    "pass_threshold": 0.7,
    "evidence_scope": { "difficulty_bucket": "standard", "min_sample_per_bucket": 6 },
    "non_regression": { "metrics": ["disc_rate", "false_negative_rate", "balanced_accuracy"], "replay_buffer_size": 20 }
  },
  "lineage": {
    "previous_version": null,
    "update_reason": "Initial contract authored alongside the arxiv skill; no prior evidence exists.",
    "evidence_case_ids": [],
    "reviewed_by": "automated-gate-only"
  },
  "retirement_state": { "status": "active" }
}
```

## Explicit non-goals of this design

Per the synthesis note's open items, this schema deliberately does **not** attempt to solve:
- the exact algorithm for computing `disc_rate`/`false_negative_rate`/`balanced_accuracy` — the
  field names are fixed, the computation is Skill-V's Eq. 6, left to implementation;
- a specific human-review UI/workflow — `lineage.reviewed_by` only reserves the field;
- cross-skill graph relations (SkillOps' `dep`/`comp`/`red`/`alt` edges) — out of scope for a
  per-skill contract; would be a separate "skill graph" schema if Hermes wants composability
  checks later.
