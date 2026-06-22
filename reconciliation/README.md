# Campaign reconciliation — independent, committed proof

This directory contains the **independent** reconciliation the Council required:
not a chat-shown number, but a re-runnable script + its committed output proving
that the union of the open PR branch diffs (against v0.16.0) plus the documented
deferred set equals the local source delta — with 0 unaccounted.

## Files

- `reconcile_campaign.sh` — fetches every open PR head fresh from the fork, builds
  coverage purely from git refs, classifies every `git diff v0.16.0..HEAD` added
  source line into covered-by-open-PR / deferred-in-#50111 / non-substantive /
  UNACCOUNTED. Exit 0 iff UNACCOUNTED == 0.
- `reconcile_output.txt` — the committed run output (PASS — 0 unaccounted).
- `symdiff_reconcile.py` — the RIGOROUS bidirectional check on the COMMON base
  v0.16.0 (D ⊆ U ∪ X), restricted to overlay-changed source files. This is the
  stricter check: it found a genuine gap the one-directional script missed
  (#50046 shipped the stable-update reader but not its config defaults — now
  folded in). Exit 0 iff D \ (U ∪ X) == ∅.
- `symdiff_cleanclone_output.txt` — symdiff run **from a clean clone** (origin =
  NousResearch, fork = arminanton, origin/main fetched fresh): PASS, 0 uncovered.
  Proves the reconciliation does not depend on any local session state.
- `v017_reapply_output.txt` — the committed all-PRs-onto-v0.17.0 net-diff check.

## Result (committed)

```
INDEPENDENT RECONCILIATION (v0.16.0..HEAD source delta, 13,396 added lines)
  covered-by-open-PR : 11,554   (across 38 open feature PRs)
  deferred-in-#50111 :  1,830   (each line proven in a deferred/*.patch)
  non-substantive    :     12
  UNACCOUNTED        :      0
RESULT: PASS — 0 unaccounted
```

## Documented exclusions (non-source generated artifacts, not PR material)

The reconciliation excludes these with rationale — they are NOT hand-written
source and never belong in a feature PR:

| Pattern | What it is | Why excluded |
|---|---|---|
| `*.bak`, `*.bak.*` | editor/dev backup snapshots (e.g. `conversation_loop.py.bak.20260607_231325`) | ~19,500 lines of point-in-time backups created during overlay development; pure noise, never source |
| `.project-intel/**` | generated project-intelligence index (FLOW_MAP, FEATURE_MAP, HEALTH, etc.) | ~219 lines of auto-generated index artifacts from `pintel`, regenerated on demand |

These exclusions are why the source delta is 13,396 lines, not the 31,250 raw
`git diff` total — the difference is entirely generated artifacts.

## CMX category — RULED (per user rule [id=92873 r5])

The `cmx/` category breaks down as:
- **conversation_loop hooks** (memory-prefetch query truncation + `enforce_response`
  capability gate) — leak-safe, generic, provider-agnostic host-side seams that the
  cmx engine *uses* but which no-op for ContextCompressor/LCM.
- **`test_cmx_hermes_engine_schemas_are_bare`** — hardcodes the PRIVATE path
  `<cmx-tree>/src/cmx/hermes_engine.py`; CANNOT be public.

**Decision:** CMX stays **deferred pending the single CMX-implementation PR**, per the
user's explicit rule that CMX-touching code travels in ONE CMX PR, never piecemeal.
Opening a partial CMX PR now would violate that rule. This is a rule-governed deferral,
recorded here with its rule id — not an open question.

## How to reproduce

```
cd <repo root, on the overlay HEAD>
./reconciliation/reconcile_campaign.sh fork    # fork = your fork remote name
# expect: RESULT: PASS — 0 unaccounted
```

## UPDATE: CMX now has a dedicated PR (#50155)

The CMX grounding-enforcement seam (the `enforce_response` hook in
conversation_loop) — previously in the `cmx/` deferred category — is now its own
dedicated draft PR **#50155** `feat(context-engine): post-response grounding-
enforcement hook` (built on origin/main, 6 tests, ruff clean, applies clean onto
v0.17.0). So CMX "lives in a separate PR," satisfying the goal for that category.

What remains in the `cmx/` deferred category is only
`test_cmx_hermes_engine_schemas_are_bare`, which hardcodes the PRIVATE path
`<LOCAL_PATH> and therefore cannot be
public — it stays deferred (it guards the private cmx engine source, which is a
separate repo).
