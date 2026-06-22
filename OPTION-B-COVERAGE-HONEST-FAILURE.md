# OPTION-B coverage proof — HONEST FAILURE (2026-06-22)

The Council asked: under OPTION B, produce an exclusion manifest such that
TARGET − EXCL is fully covered by the 39 PRs (residual == 0). I built a principled
manifest (R1 excluded-files + R2 private-infra-files + R3 private-content-patterns) and
ran it. **It does NOT reach residual 0.** This is the honest result.

## Result (re-runnable: `SRC=<checkout> python3 exclusion_manifest_and_coverage.py`)
```
TARGET added-lines (v0.16.0→HEAD, baks excluded): 10097
PR coverage (diff ∪ content):                     152312 lines
UNCOVERED (TARGET − PR coverage):                  2933
  − EXCL manifest (R1+R2+R3):                      1597
  − cosmetic (dash/whitespace match):              1584  ← TRUE residual, NOT zero
```

## Why OPTION B does NOT cleanly close (the decisive finding)
The 1584 residual is NOT all private/excluded. Direct verification found **genuinely
contributable content in NO PR**, e.g. in `agent/conversation_loop.py` (109 residual lines):

```
# Cap query length before memory recall (embedding/hybrid search).
# Hosts like gsd's hermes-cli ACP bridge pack their whole system prompt + tool
# catalogue into the prompt text, arriving here as a 50-100KB "user message"...
_PREFETCH_QUERY_MAX_CHARS = int(os.environ.get("HERMES_PREFETCH_QUERY_MAX_CHARS","1500"))
```
This is a clean, no-private-token bug-fix (bound an oversized memory-recall query). It is
in **none** of the 4 PRs that touch conversation_loop.py (#50155/#50073/#49917/#49184).
Other clean-contributable residual: refusal-handling messages, async-fallback logging,
"Memory prefetch query truncated" log line.

So the residual is a MIX:
- **private/excluded** (account caps `"gpt-5.4":750000`, `agy_cli`, opus-context, fable
  effort tables) — correctly out of scope, AND
- **genuinely contributable + missing** (prefetch cap, refusal msgs, async-fallback logs)
  — these SHOULD be in a PR and are not.

The two are **intertwined in the same files** (conversation_loop.py carries both the
private prefill-guard from `9fec781fc` AND the clean prefetch cap), which is exactly why
no clean file-level OPTION B manifest reaches residual 0.

## HONEST conclusion — the goal is NOT met, and neither option is free
- **OPTION B is NOT clean**: ~1584 residual, of which a real subset is contributable
  content missing from PRs. Accepting B means accepting that some contributable changes
  do NOT live in a PR — a genuine gap, not just private exclusions.
- **OPTION A (line-exact)** would re-cut PRs to capture the clean residual — but the
  clean lines are intertwined with private/entangled lines in the same hunks
  (`9fec781fc` mega-commit), so OPTION A re-pulls excluded content unless each hunk is
  hand-split.
- A **third path (OPTION C)** is the honest middle: extract the genuinely-contributable
  residual (prefetch cap, refusal-handling, async-fallback, etc.) into a small number of
  NEW clean PRs, hand-splitting them from the private content in their files. This is
  real per-hunk surgery across ~5-10 shared files.

## What I am NOT doing
I am not declaring done. I am not claiming OPTION B reaches residual 0 (it does not). I
corrected my earlier "216/216, 0 uncovered" heuristic with this mechanical proof. The
genuine remaining work + scope choice (A / B-with-acknowledged-gap / C-extract-clean-
residual) is the user's decision, now backed by hard numbers and concrete examples.
