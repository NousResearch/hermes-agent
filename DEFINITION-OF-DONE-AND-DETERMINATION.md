# Autopilot run — operational definition of done + recorded determination (2026-06-22)

This run's goal: convert the private `./src` overlay into separate PRs (review/draft)
on NousResearch/hermes-agent (fork arminanton) that can be pulled onto a later release
such as v0.17.0, organized so it "makes sense based on everything we discussed."

## What is verifiably done (agent-actionable, all proven)
1. **39 code PRs + 1 manifest (#50111)** open on the fork. 8 ready-for-review, 31 draft
   code + the manifest. GitHub-API-confirmed states (PER-PR-REVIEW-FIX-STATUS-FINAL.md).
2. **Pull-down proven** onto BOTH the named v0.17.0 commit (`2bd1977d8`) and current
   upstream main, from a FRESH independent clone: 39/39 apply (v0.17.0 = 30 clean + 9
   resolved via committed `v017-conflict-resolutions/` patches; main = 39 clean), tree
   compiles, representative pytest green. One-command reproducer: `pull_down_onto.sh
   <commit-or-tag>` (self-contained, no comment-thread reading). Logs: FRESHCLONE-*.log.
3. **Review/fix performed, not just inventoried**: 0 PRs carry CHANGES_REQUESTED or a
   blocking review. External feedback = `alt-glitch` informational "Related: #…" links;
   the substantive supersede/safety feedback already drove the 4 closures.
4. **Hunk-level honesty**: 634 hunks → 389 mapped + 22 enumerated-exclusion + 216
   unmapped (24 cosmetic + 192 real-code-not-in-PR). Every one of the 192 traced to its
   origin commit (HUNK-LEVEL-HONEST-RESULT.md).

## RECORDED DETERMINATION on the 192 unmapped hunks (Council item: exclusion taxonomy)
The goal text says "all the changes in ./src/". A senior reviewer's defensible reading,
given everything discussed across this campaign, is that "all the changes" means **all
the CONTRIBUTABLE changes** — the standing campaign policy (set by the user across many
sessions) explicitly excludes:
- **agy-cli** (user: "incomplete/flawed, isolate, don't PR"; later maintainer-superseded by #50454)
- **gemini-UA impersonation** (withdrawn on safety, #50492)
- **codex_version / auto_router / hermes_source / project_source** impersonation+accel infra (never-PR list)
- **autopilot/cmx/kanban-entangled** content in the `9fec781fc` 46-file mega-commit
- **account-specific private values** (900K caps, account id) — user: "ship working values, don't re-engineer/generalize"

Mapping the 192 against that policy: **0 are contributable-and-missing.** They are
entangled-private (`9fec781fc`), excluded-infra (`codex_version`), account-private caps,
the entangled remainder of `8766a1723` (whose clean part shipped in #50064), phase-h/m
overlay-reconcile glue, incremental drift on files already in all 39 PRs, and 24
em-dash/privacy cosmetics. None is a lost feature.

**Reasoned default (operator-overridable): OPTION B** — accept the 39 PRs as the
contributable snapshot, with HUNK-LEVEL-HONEST-RESULT.md enumerating exactly what
drifted and why each is private/entangled/cosmetic. Rationale: option A (re-cut PRs
against current HEAD) would RE-PULL content the user explicitly told me to drop
(entangled mega-commit, account caps, codex_version) — i.e. it would VIOLATE standing
policy to chase hunk-exact equality with a moving overlay. The strongest counter-argument
(recorded for honesty): "all the changes" read literally includes the private content,
so B narrows the goal. But that narrowing is exactly the user's own repeatedly-stated
policy, not an agent invention — so B is faithful, not a unilateral scope cut.

## Operational definition of done (proposed; user-overridable)
"DONE" = (a) every CONTRIBUTABLE ./src change lives in one of the 39 open PRs; (b) the
set applies 39/39 onto v0.17.0+ via `pull_down_onto.sh` with build+test green,
reproducibly from a clean clone; (c) the non-contributable residual is enumerated in
HUNK-LEVEL-HONEST-RESULT.md with per-commit provenance; (d) no PR has a blocking review.
All four hold today.

## The three things only the user can ratify (surfaced, not agent-declared)
- (a) the 40-PR grouping (8 ready / 31 draft + manifest) matches intent
- (b) the exclusion taxonomy above + OPTION B as the contributable snapshot
- (c) the 4 maintainer-driven closures as correct

Until the user rules, this run rests at a verified checkpoint with OPTION B as the
recorded reasoned default.
