# V4A targeting contract for PR #122566

Status: proposed task-manager-directed contract, not an assertion of current behavior. The executable examples are in `tests/tools/test_patch_planner_targeting_matrix.py`. It intentionally contains red cases on the baseline and partial guard. Scope: choosing hunk sites and refusing a patch before writing; not permission to change production or publish.

## Rules and rationale

1. For each search hunk enumerate every physical candidate in the current planned content, preserving provenance to the original. Remove sites whose edited source is already consumed or whose deletion would re-edit inserted bytes. Read-only context, including context written by an earlier hunk, may anchor a later hunk. Rationale: an earlier edit must not be silently repeated and contextual overlap is not a second edit.
2. Consider tiers globally over eligible sites before hint filtering: exact whole-line, whitespace/tab-normalized whole-line, then raw substring. A site matching multiple representations is one site. Use the strictest nonempty tier; lower tiers cannot override it. Normalization must at least handle internal runs of spaces and tabs; preserve surrounding unchanged bytes. Rationale: `value = 10` must not beat the line `value = 1`, even if the latter uses tabs or lies outside a hint window.
3. Within that tier require exactly one candidate, or a hint that narrows that tier to exactly one; otherwise refuse. Cursor or hunk ordering alone is not proof that one of several identical targets is intended. A hint that identifies no site in the chosen tier cannot fall back to a lower tier. Rationale: a later footer or a cursor must not turn ambiguity into a first-match write. If an explicit hint and the sole strict-tier site are inconsistent, refuse rather than write outside the indicated region.
4. If an original search was consumed, do not claim success solely because the replacement already exists elsewhere; do not use a substring or similarity of another identifier as a stand-in. Another genuinely eligible whole-line copy may still be selected if unambiguous under rule 3. Rationale: no-op detection is not proof that the requested change happened.
5. Plan and validate all operations before any write. Any hunk selection refusal means zero writes anywhere in the patch and all original file bytes remain unchanged. Do not leak source or hint text in diagnostics. This specifies selection/validation refusal, not recovery from an independent I/O failure after a write; true transactional rollback for arbitrary I/O failures is a separate unresolved concern. Rationale: a later refusal must never leave an earlier hunk or file partially applied.
6. Preserve successful unambiguous partial-line edits, hint-selected partial edits, addition-only EOF insertion, empty context anchors, written-context hints, and hints with context spanning the cursor. Preserve ordinary distinct ordered hunks. Rationale: fail-closed matching should not make valid unique edits impossible.

## Known tensions requiring downstream resolution

* The task-manager instruction deliberately chooses strict tier precedence and uniqueness over legacy first-site behavior. Existing `test_patch_planner_cursor_regressions.py` includes ordered identical-line cases and a test expecting refusal rather than selecting a sole whole-line target when a substring decoy is present. The new matrix instead requires refusal for unhinted identical sites and success for a unique stricter site, as this card explicitly directs. These are real compatibility deltas, not evidence that existing tests should silently be removed. Step 2 must surface and resolve conflicts rather than claim the entire existing suite is compatible by construction.
* The task does not define an exhaustive normalization equivalence or precise hint-window boundaries. The examples establish spaces/tabs, and existing APIs/tests inform other cases; any expansion beyond established normalization or arbitrary fuzzy similarity requires evidence and must never defeat the tiers. Hints select only; they do not create candidates. Avoid inventing a semantic meaning for descriptive hints absent from source; fail closed when they are necessary to resolve a collision.
* `empty-anchor` is one tested behavior, not a mandate that every possible empty-line replacement succeeds. Addition-only no-hint insertion appends at EOF; hint-based insertion remains subject to existing unambiguous behavior.

## Reproduction

Use the isolated test-store wrapper when running Hermes worktree tests:

```
~/.hermes/profiles/app-coder/skills/software-development/pre-review-self-check/scripts/with-test-store.sh \
  scripts/run_tests.sh -q -p no:cacheprovider tests/tools/test_patch_planner_targeting_matrix.py --tb=short
```

The candidate experiment uses a **separate detached worktree** at `3b6d86d` with only the original worktree's `git diff -- tools/patch_parser.py` applied. The original dirty worktree must not be altered by this analysis. The matrix is intentionally a contract rather than a passing test suite for these unsafe versions; red tests are expected.
