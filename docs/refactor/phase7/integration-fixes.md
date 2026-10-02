# Preserved adjacent-phase integration fixes

The independent Phase 5, Phase 6 and Phase 7 branches remain independent.
The combined branch is a verification checkout, not a required baseline.

Annotated Git tag `refactor/phase7-integration-fixes` preserves commit
`a75f79c78cc9c368577abe2886c421bb6d10281c` and its combined ancestor history.
The tag keeps that history reachable even if the integration branch or worktree
is later removed. The tag is local; it has not been pushed.

## Ownership and disposition

| Change in the preserved commit | Owner | Disposition |
| --- | --- | --- |
| Remove restored static provider aliases; use canonical normalization in CLI auth | Phase 5 providers | Merge reconciliation. The standalone Phase 5 branch already uses `normalize_provider_identity`; do not import Phase 5 into Phase 6 for this. |
| Import Actual endpoint helpers directly from `providers.route_identity` in agent and CLI consumers | Phase 5 providers | Apply when reconciling provider and authentication extractions. Preserve Phase 6's independent endpoint owner until that integration. |
| Remove duplicate URL helpers from `hermes_cli.route_identity`, retaining application route/context-pin operations | Phase 5 providers / application boundary | Combined ownership repair. Do not delete Phase 6's independently required helpers in isolation. |
| Call `auth.providers.minimax` directly from runtime provider dispatch | Phase 6 auth | Already present in standalone Phase 6 at `2b460a8673c9c579296f0c335e0aeb6c1bd99fdc`; integration repair restores it after the merge. No backport needed. |

Direct source comparison on 2026-10-02 confirmed the standalone Phase 5
normalization call and Phase 6 MiniMax call. No independent phase code was
changed while preserving these fixes.

## Reuse at final integration

Inspect `git show refactor/phase7-integration-fixes` against the actual combined
result. Reapply only unresolved changes; do not blindly cherry-pick the entire
commit onto an independent phase. Preserve the canonical provider and auth
owners without introducing forwarding paths or cross-phase dependencies.

The original combined verification is recorded in `phase7.7.md`,
`phase7.7-verification.json`, `phase7.7-test-results.txt` and the installed-wheel
receipts in this directory. The preserved implementation passed 363 combined
regressions and 108 runtime tests. These are historical results, not a claim
that a future merge has been verified; rerun the relevant suites on that merge.
The documented Windows session-policy baseline defect remains open.
