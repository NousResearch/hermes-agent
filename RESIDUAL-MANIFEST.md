# Residual manifest — residual PRs CLOSED, content lives in feature PRs

**Status: RESOLVED.** The two residual/overlay PRs (#50484 clean-residual,
#50487 drift-residual) have been **CLOSED as redundant**: verification proved
every file they carried is owned by a topical feature PR that applies cleanly
on v0.17.0. Nothing was lost.

## Why they were closed (not maintained as overlays)

The residual PRs existed to track overlay lines whose files drifted upstream.
But a per-file check showed their content is already delivered by feature PRs:

- #50487's 14 files → #49917 (autopilot), #48024 (reasoning API), #50064
  (copilot identity), #48069 (mcp), #50045 (skills), #50296/#50056 (state),
  #50146/#49644 (gateway/run), #48101/#49916/#50073/#50155/#49184.
  Proof: `comm -23` of #50487's resolved conversation_loop autopilot lines vs
  #49917's version = empty (no unique lines).
- #50484's 20 files → all owned by feature PRs (0 uncovered): #49917, #48065,
  #49449, #49644, #50064, #50033, #48101, #50055, #50296, #50078, #50080, #49184.

Closing them removed the ONLY PRs that needed special "apply v0.17.0-ready/"
handling. The remaining set applies cleanly with no overlay mechanism.

## Net state on v0.17.0 (2bd1977d8) — independent fresh-clone verification

40 feature PRs CLEAN, 0 CONFLICT, 0 compile failures (this PR = manifest, skipped).
See APPLY-MATRIX-v0.17.0-FRESH.txt and verify_prs_on_v017_fresh.sh.

## Coverage (v0.16.0 3c231eb → overlay HEAD)
- 160 delta files: 138 in feature PRs, 22 non-contributable (see NON-CONTRIBUTABLE.md).
- Real-source orphans: 0.
