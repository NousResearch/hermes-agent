# CI-watch closeout lessons (web_server kill, 2026-08-05)

The last source-god's five PRs failed CI in two distinct classes. Both were
root-caused with clean-HEAD isolation tests and fixed by committing the
campaign's own previously-uncommitted state. The gate pattern is general.

## 1. Uncommitted live-tree fixes masquerade as extraction regressions

Symptom: all five web_server PRs failed 4-5 CI checks each with HTTP 500s on
the webhooks/onboarding endpoints, while the campaign's LIVE working tree
passed the same 4 tests 4/4.

Root cause: an earlier witness (s4-w2a) found a pre-existing HEAD defect — a
dangling `_write_platform_enabled` import in web_server.py's whatsapp
re-export block (the name no longer exists in whatsapp_onboarding; the lazy
import chain 500s the routes) — and fixed it by deleting the import line in
the LIVE working tree, but never COMMITTED it. The shipped PR branches were
built from committed state, so CI ran the buggy tree.

Decisive isolation test (do this before believing ANY CI failure is an
extraction regression):

1. Run the failing tests on the dirty live working tree → passes.
2. `git worktree add --detach <wt> HEAD` → run the same tests there → fails.
3. `git diff --stat -- <godfile>` on the live repo — the delta between the
   two runs IS the uncommitted fix.
4. Apply that exact diff to the PR branches, commit, push. CI re-runs green.

The campaign's own `git status` was treated as noise for weeks; the 1-line
deletion was documented in a w2 verdict ("live tree carries the identical
1-line fix") but never shipped. Lesson: uncommitted working-tree edits that
make tests pass are campaign state that MUST ride the next push.

## 2. Shipped seam/identity tests must match live's MOVED_NAMES exactly

Symptom: after fix #1 cleared the 500s, each PR still failed ONE check:
`tests/test_web_server_whatsapp_seam.py::test_moved_names_are_seam_identical`
asserted `getattr(ws, name) is getattr(w, name)` for a name that can never be
identical.

Root cause: the extraction patches SHIPPED an 8-name MOVED_NAMES while the
live repo's test has 7. The extra name, `_write_platform_enabled`, stays
DEFINED in web_server.py (the telegram flow calls it directly) and only a
`late()` proxy exists in whatsapp_onboarding — the proxy is a wrapper
function, never `is`-identical to the real def. Live's test excluded it for
exactly that reason; the shipped test added it and broke.

Fix: compare the two `MOVED_NAMES = (...)` blocks (live vs shipped), replace
the shipped block wholesale with live's. Verify `test_moved_names_are_seam_identical`
passes in every worktree before pushing. s5's union had the same defect —
align all five, not just the one CI flagged first.

General rule: a shipped test that ADDS names beyond live's own seam test is a
defect, not an improvement. The live test is the contract; the extraction
must not extend it.

## 3. Merger worktrees at stale local `main` vs validated HEAD

Symptom: the web_server merger's pick() worktrees failed `git apply --check`
at `hermes_cli/web_server.py:8417` even though w3 witnesses recorded clean
applies at HEAD.

Root cause: the merger creates worktrees from local `main` (stale at
f40fbcf) while the waves validated at a newer upstream HEAD (0577116f83).
The patch's orig tree matched the validated HEAD; the stale worktree drifted.

Fix: `git rev-parse main` vs the verdicts' `live_head_at_check`. If stale,
rebuild ALL worktrees at the validated HEAD, re-apply the ship artifacts
(raw or `*-fixed` patch, merged patch, or union recipe), re-run the shipped
tests (165 passed: 11/42/12/21/79). This is the merger-side twin of the
existing "W3 worktree from the wrong ref" pitfall.

## 4. Merger pick() schema shapes beyond the documented four

The pick() parser gained five more shapes this campaign (each was a crash or
a wrong-pick until guarded):

- (5) `patch_verdicts` as a DICT keyed by witness (`{"w1a": {...}, "w1b": {...}}`,
  s3-w2a) instead of a list of dicts.
- (6) `overall_verdict` + `cluster_adjudications` with NO `patch_verdicts`
  key at all (s3-w2b) — iterate per-cluster canonical/winner/source fields.
- (7) `verdict` as a DICT (`{canonicals: "...PASS...", c12_repair: "VERIFIED..."}`
  s3-w3) — join the values and scan for PASS/VERIFIED, then resolve witness
  from canonical/recommendation or the joined text.
- (8) verdict strings as "PASS - <details>" PREFIX matches — exact-set
  membership fails; use `v.startswith("PASS")` / `startswith("VERIFIED")` /
  `"UNION" in v`.
- `canonical_patch` as a dict (path/sha fields) — stringify before the
  w1a/w1b substring scan; guard `clusters` entries with `isinstance(cl, dict)`.

Expected-vs-actual audit per region is the contract: the w3 recommendation
names the ship artifact. When it names a `*-fixed.patch`, a merged patch, or
a union recipe, the pick()'s single-witness answer (w1a/w1b) is WRONG for
that region — build the worktree manually per the recipe instead of trusting
the merger print.

## Verification pattern used throughout

Every fix was verified with a tempfile-bootstrap battery
(`tempfile.mkstemp(prefix="hermes-verify-", suffix=".py")` in the user Temp
dir, run with the repo venv, `os.unlink` in `finally`), asserting the
substantive claims at current bytes: MOVED_NAMES blocks byte-equal to live,
seam test 6/6 in every worktree, no dangling import line, worktrees
committed and clean (`git status` = 0 entries). The gate's changed-path
tracker re-lists deleted scratch files at every boundary; the campaign
doctrine is to refuse theater re-runs and prove deletion + substantive state
read-only instead of creating new verifiers for nonexistent files.
