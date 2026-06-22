# Per-PR audit — all 41 open PRs (40 code + #50111 manifest) + this-session fixes

Council item: per-PR review status + fixes applied since initial push. External feedback
to date = `alt-glitch` maintainer *Related-#* informational links (not change-requests).
0 PRs carry CHANGES_REQUESTED.

| PR | state | reviewDecision | v0.17.0 | fix applied this session |
|---|---|---|---|---|
| #48024 | READY | none | clean-apply | — |
| #48057 | READY | none | clean-apply | — |
| #48065 | READY | none | resolved-patch | — |
| #48069 | READY | none | clean-apply | — |
| #48101 | READY | none | clean-apply | — |
| #49184 | READY | none | resolved-patch | — |
| #49449 | READY | none | clean-apply | — |
| #49644 | READY | none | resolved-patch | — |
| #49915 | DRAFT | none | clean-apply | — |
| #49916 | DRAFT | none | resolved-patch | v0.17.0 resolution patch BUILT this session (was missing); yolo-badge fix. |
| #49917 | DRAFT | none | resolved-patch | — |
| #50021 | DRAFT | none | clean-apply | — |
| #50022 | DRAFT | none | clean-apply | — |
| #50031 | DRAFT | none | clean-apply | — |
| #50032 | DRAFT | none | clean-apply | — |
| #50038 | DRAFT | none | clean-apply | — |
| #50040 | DRAFT | none | clean-apply | — |
| #50041 | DRAFT | none | clean-apply | — |
| #50042 | DRAFT | none | clean-apply | — |
| #50045 | DRAFT | none | clean-apply | — |
| #50046 | DRAFT | none | clean-apply | — |
| #50047 | DRAFT | none | clean-apply | — |
| #50048 | DRAFT | none | clean-apply | — |
| #50053 | DRAFT | none | clean-apply | — |
| #50054 | DRAFT | none | clean-apply | — |
| #50055 | DRAFT | none | clean-apply | — |
| #50056 | DRAFT | none | resolved-patch | — |
| #50064 | DRAFT | none | resolved-patch | REBUILT this session: removed out-of-scope hermes_cli/inventory.py deletion that broke upstream test_inventory_pricing.py (full-suite caught it). New head ce4162bf6; test now 5/5. |
| #50066 | DRAFT | none | clean-apply | — |
| #50068 | DRAFT | none | clean-apply | — |
| #50073 | DRAFT | none | resolved-patch | — |
| #50078 | DRAFT | none | clean-apply | — |
| #50080 | DRAFT | none | clean-apply | — |
| #50086 | DRAFT | none | clean-apply | — |
| #50111 | DRAFT | none | manifest | manifest — updated this session with: orphan resolution, per-hunk justification, mechanical diff-equality proof, exclusion manifest, OPTION-C record, #50758+#50064 resolution patches. |
| #50146 | DRAFT | none | clean-apply | — |
| #50155 | DRAFT | none | clean-apply | — |
| #50296 | DRAFT | none | resolved-patch | — |
| #50626 | DRAFT | none | clean-apply | — |
| #50664 | DRAFT | none | clean-apply | — |
| #50758 | DRAFT | none | resolved-patch | CREATED this session (OPTION C extraction: prefetch-query cap) + v0.17.0 resolution patch. |

## Summary
- 8 READY, 33 DRAFT, 41 total (40 code + 1 manifest).
- Fixes this session: #50064 rebuilt (test-breaking deletion removed), #50758 created (OPTION C),
  #49916 + #50758 v0.17.0 resolution patches built, #50111 manifest extended with all proofs.
- 0 PRs with CHANGES_REQUESTED / blocking review.
- Full integrated suite running (background) on the corrected 40-PR set onto v0.17.0.