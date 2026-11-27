# DEGRADED accounting and verdict math

A reviewer is `DEGRADED` when council.py couldn't get a
schema-valid response, OR when the redaction safety net refused
to send, OR when credential pre-flight failed. `DEGRADED` is
**not** "PASS-with-fluff" — it means "this reviewer contributed
nothing." The math below is what stops a single failed reviewer
from quietly degrading a BLOCK to a PASS.

## The two kinds of DEGRADED

- **Hard DEGRADED:** the reviewer's CLI couldn't run, returned
  non-zero with no output, or produced output that failed schema
  validation. The `reason` field explains which.
- **Soft DEGRADED (for verdict math only):** credential
  pre-flight failed, OR the redaction safety net refused to send.
  The reviewer is *available* but not *invokable* in this run.
  Both are counted as DEGRADED for the council-verdict math below
  so that a credential-locked reviewer can't accidentally be
  treated as PASS.

## Verdict computation

```
non_degraded = reviewers minus DEGRADED
degraded     = reviewers minus non_degraded

if any non_degraded.verdict == BLOCK:
    council.verdict = BLOCK           # exit 2
elif non_degraded is empty:
    council.verdict = DEGRADED        # exit 3 if --strict else 0
elif degraded is non-empty:
    council.verdict = PASS_DEGRADED   # exit 3 if --strict else 0
else:
    council.verdict = PASS            # exit 0
```

The `DEGRADED-prevents-clean-PASS rule` (the council-adopted
correction to v1.0) is the third branch: if any requested
reviewer is DEGRADED, the verdict cannot be the unqualified
`PASS` of the v1.0 contract. It becomes `PASS_DEGRADED` (still
exit 0 by default) with `consensus_notes` calling out which
reviewers were degraded. `--strict` upgrades this to exit 3 so
CI can refuse a clean PASS with missing reviewers.

## Exit code precedence

When multiple conditions apply:

1. `REDACTION_FAILED` (exit 4) wins over everything — no review
   happened, exit 4 is the only honest answer.
2. `BLOCK` (exit 2) wins over `DEGRADED` (exit 3). A run that
   has a blocking finding from one reviewer and a DEGRADED on
   another is `BLOCK`, not `DEGRADED`.
3. `DEGRADED` (exit 3, only under `--strict`) wins over `PASS`
   (exit 0). Non-strict mode downgrades both `DEGRADED` and
   `PASS_DEGRADED` to exit 0; strict mode keeps them at 3.

`--accept-degraded` is the explicit override: it downgrades
exit 3 to 0 even under `--strict`. Use it when a human has
reviewed the coverage gap and decided the run is still good
enough. The flag is logged in the output but doesn't appear in
the JSON schema (it's a runtime-only override).

## Risk and confidence aggregation

- `risk_level` = max-of across non-degraded reviewers (low <
  medium < high).
- `confidence` = min-of across non-degraded reviewers (a degraded
  reviewer drags confidence down, even though it doesn't
  contribute findings).

This means a single high-confidence `BLOCK` from one reviewer
produces a high-confidence `BLOCK` overall; a single
low-confidence `PASS` from one reviewer and a high-confidence
`PASS` from another produces a low-confidence `PASS` overall.

## blocking_findings dedup

The union of blocking findings across non-degraded reviewers,
deduped by string equality with order preserved. A finding that
two reviewers independently raised once is one finding in the
union, not two. This is intentional: it makes the BLOCK list
shorter and the action items clearer, at the cost of losing
"two reviewers independently raised this" as a signal. If you
want that signal, read `reviews.<rev>.blocking_findings`
directly.

## The silent-failure failure mode this prevents

Without the DEGRADED-prevents-clean-PASS rule, a single failed
reviewer could quietly degrade a `BLOCK` to a `PASS` with full
high confidence — exactly the silent-failure failure mode this
skill is supposed to prevent. The original v1.0 draft had this
bug; the council review caught it; v1.2 fixes it.

## Verification recipe

```bash
# Force a degraded outcome (no reviewers available)
python council.py --file clean.md --kind plan --title "test" \
  --reviewers nonexistent --no-council --strict
echo "expect exit=3 (DEGRADED, strict)"

# Override the strict refusal
python council.py --file clean.md --kind plan --title "test" \
  --reviewers nonexistent --no-council --strict --accept-degraded
echo "expect exit=0"

# BLOCK wins over DEGRADED: pipe an artifact that
# reviewer=claude will BLOCK (after you've established the
# contract) and reviewers=claude,nonexistent; expect exit=2.
```
