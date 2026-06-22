# Residual representation — every ./src delta now lives in a PR (2026-06-22)

Per the Council's option (i): place the intertwined residual into a (draft) PR so that
**every line of the ./src delta vs v0.16.0 is represented in an open or draft PR.**

## What this closes
Prior state: the 40 feature PRs carry the contributable surface, but ~1900 added lines of
current src (private-overlay + entangled `9fec781fc` + account caps + cosmetic + a small
genuinely-contributable remainder) were NOT in any PR — only enumerated/justified. The
Council correctly held that "enumerated and deferred" ≠ "lives in a PR".

## The artifact
`RESIDUAL-NOT-IN-ANY-PR.patch` (in this branch, #50111) — **38 files, 141 hunks**: exactly
the `git diff v0.16.0..src-HEAD` hunks whose added lines are NOT carried by any of the 40
feature PRs. Generated mechanically (a hunk is included iff >=1 of its added lines is in
the residual set computed by `mechanical_diff_equality.py`). This is the faithful,
bounded representation of the residual — NOT the full 144-file delta (most of which IS in
the feature PRs), and NOT the integrated-tree-vs-src diff (which is dominated by
upstream-vs-base drift, a measurement artifact).

## Now every ./src delta is represented
- **Contributable surface** → the 40 open feature PRs (#48024…#50758), 40/40 apply onto
  v0.17.0, build green.
- **Residual** (private-overlay + entangled + account-caps + cosmetic + the
  not-yet-extracted contributable remainder) → `RESIDUAL-NOT-IN-ANY-PR.patch` on THIS
  #50111 draft (NOT FOR MERGE — a preservation tracker, clearly labeled).

So: 40 feature PRs ∪ this residual patch == the complete v0.16.0→HEAD ./src delta. Nothing
is now merely "deferred without a home."

## This is a preservation artifact, not a merge candidate
#50111 is explicitly `[manifest, NOT FOR MERGE]`. The residual patch lives here because the
residual is overwhelmingly the content the user's standing policy excludes from merge
(agy/auto-router/codex_version/account-caps/entangled-`9fec781fc`). It is preserved as a
re-appliable artifact so NOTHING from the v0.16.0→HEAD delta is lost on upgrade — which is
the campaign's actual goal ("a re-application manifest so nothing is lost when we upgrade").
The genuinely-contributable slice already extracted (prefetch cap → #50758) shows the path
for promoting any of these residual hunks into a real feature PR if the operator chooses
(OPTION C decision, recorded in OPERATOR-DECISION-RECORD.md).
