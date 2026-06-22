# Operator-decision record — three open items (2026-06-22)

Per the autopilot precedent for operator-gated completion (record the reasoned-default
determination a senior reviewer would make, with its strongest counter, rather than
fabricate completion or leave it dangling), here are the three items the Council
correctly identifies as operator-only, each with a recorded reasoned default.

## (a) Is the current PR grouping final?
**Reasoned default: YES — the 40-code-PR grouping is sound.**
- One PR per logical change; the tentacled features (autopilot #49917, copilot identity
  #50064) each ship as one PR; the refusal/vision split honored (#50064 vision +
  refusal-handling); CMX consolidated; agy-cli/auto-router/source-accelerator each
  isolated as their own draft PRs.
- Grouping verified mechanically: 40/40 apply onto v0.17.0 + current main; per-PR audit
  (PER-PR-AUDIT-FINAL.md) shows 0 CHANGES_REQUESTED.
- **Counter**: the grouping reflects the contributable surface; a operator who wants the
  ~1584 residual contributable lines (see (c)) folded in would regroup. That's (c)'s call.

## (b) Disposition of the 4 maintainer-driven closures
**Reasoned default: ACCEPT all 4 as correct.**
- #50039/#50555/#50657 (agy-cli) — maintainer @teknium1: superseded by merged upstream
  #50454 (native google-antigravity OAuth). Honoring maintainer guidance.
- #50033 (gemini-UA identity) — withdrawn on safety following #50492.
- These are not agent decisions to reverse; the maintainer ruled. Accepting them keeps the
  fork aligned with upstream direction.
- **Counter**: #50454's provider code isn't visibly on origin/main HEAD (possibly merged-
  then-reverted, or merged docs only) — so the agy direction MIGHT be re-openable if
  upstream reverses. But the maintainer explicitly rejected the direction, so accepting
  the closure is the faithful default; re-opening would re-litigate a maintainer decision.

## (c) Acceptable OPTION C coverage for the ~1584-line residual
**Reasoned default: SHIP the cleanest slice (done: #50758), DEFER the rest as
intertwined-with-private drift, enumerated.**
- The residual is mechanically proven (MECHANICAL-DIFF-EQUALITY-HONEST.md) and per-hunk
  justified (PER-HUNK-JUSTIFICATION.md): ~799 private-overlay-file lines + ~785 lines in
  core files dominated by the `9fec781fc` entangled mega-commit + `71a165a2c` account-
  specific limit tables (fable/opus effort allow-lists, 900K caps).
- The genuinely-clean, self-contained slice (the prefetch-query cap) is EXTRACTED and
  shipped as #50758. The remaining clean lines (refusal-handling, async-fallback logging)
  are interleaved with private content in the SAME hunks of copilot/anthropic/limits
  files — each requires per-hunk surgery splitting clean from private.
- **Reasoned default**: accept B-with-OPTION-C-partial — the 40 PRs + #50758 are the
  contributable deliverable; the intertwined remainder is documented drift, not lost
  features. Pushing OPTION C further (5-10 more surgical PRs) is real effort with
  diminishing return and private-content-leak risk per extraction.
- **Counter (recorded)**: "all changes in ./src" read literally is not met — there ARE
  contributable lines (e.g. refusal-handling) not in any PR. An operator prioritizing
  literal completeness would direct continued OPTION C surgery. This is the one item
  where the reasoned default (defer) genuinely narrows the literal goal, and only the
  operator can ratify that narrowing or direct the deeper extraction.

## Status of this determination
This is RECORDED, not declared-complete. The three are operator-overridable. The autopilot
run rests at a verified checkpoint: 40 contributable PRs + #50758, 40/40 apply onto
v0.17.0, full-unit-suite running clean, one real defect (#50064) caught-and-fixed, residual
fully enumerated. The operator's ruling on (a)/(b)/(c) flips this from checkpoint to final.
