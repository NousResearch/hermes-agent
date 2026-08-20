# False positives in `verify_done.sh`

The auditor's regex is intentionally aggressive. The list below
documents patterns that *do* contain a "claim"-shaped word but should
not trigger the auditor. Each comes with how to suppress.

## 1. Third-party / vendor output

When you quote a vendor's response, their prose might say "done" or
"successfully" without being your claim.

**Examples:**
- "Stripe webhook: Setup complete."
- "GitHub Actions: All checks have completed."

**Fix:** Wrap the quote in a fence without a matching claim-shape
preceding it, or precede the quote with "Vendor says:" / "Per logs:"
to make it clear it's a citation.

## 2. Citations of external docs / READMEs

When pasting documentation excerpts that contain "done" or "fixed".

**Fix:** Use a code fence prefixed with a language hint and a comment
indicating it's a citation.

## 3. Spanish past-tense or subjunctive forms

`quedo` (past), `quedó` (past), `queden` (subjunctive) — these are
sometimes legitimate claims, sometimes just normal text.

**Fix:** Match against your *own* agent voice. If you're quoting a
third person ("el servicio quedó activo"), prefix with the speaker:
"[service log] quedó activo".

## 4. Prose like "I haven't done X yet"

The regex tries to skip these by anchoring on `\b`. But edge cases like
"haven't" slips through.

**Fix:** Reword — "X is not yet done" with no `\b` boundary before
"done" generally trips the match correctly. Not a bug, just a
heuristic.

## 5. Long messages where proof is far from claim

The auditor's window is 2 lines either side. If your draft has:

```
Listo.

[... 20 lines of explanation ...]

$ docker ps
```

...the auditor flags line 1 because the proof (line 21) is beyond the
window.

**Fix:** Either:
- Move the proof to within 2 lines of the claim.
- Or restate the claim right above the proof:

  ```
  Hago `$ docker ps` para confirmar:
  
  $ docker ps
  Container ...
  ```
  
  This way the rephrased claim is followed by proof.

## 6. The auditor's own output

If you run the auditor *within* a draft reply (e.g., embedding its
output as documentation), the auditor sees its own printed `✓` /
`✗` markers and accepts them as proof. That's correct behavior — the
embedded proof counts.

## What's *not* in scope

The auditor does NOT attempt to check semantic truth. It only catches
the *form* of unsourced claims. Verifying that a claimed exit code
matches the actual one is the agent's responsibility.
