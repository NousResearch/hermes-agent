# R9.5 FINAL — STATUS: DONE (pending review + push)

## RC1: `1a1bb48f` (tag rc1, unmoved) | UPSTREAM: `dee30d12`
## RC2: `15672997bfbc219a28964fa9a7ff480351af4e18` (parent `dee30d12`)
## METHOD: `cherry-pick --no-commit` in worktree `hermes-r94` (zero conflicts)

## TEST: 219 passed, 0 failed, 0 skipped (py3.14+numpy; log `rc2_suite.log`)
## SECURITY: green inside 219; 0 critical/leakage
## PERFORMANCE: vs RC1 within noise (R9.4 delta stands)
## GOLD: R1–R4 green, labels untouched
## BACKUP: parity green; scratch-only policy
## OFFLINE: stdlib-only + `network_dependencies() == []`
## LLM: 0

## MANIFEST: 116 files, blob-based + blob-verified (`manifest.json`)
## PROVENANCE: `provenance.json` (rc1 unmoved/unmutated)

## PUSH: PENDING (review PASS required first; then commit+rc2 tag only)
## RELEASE: NOT PUBLISHED (separate phase)
