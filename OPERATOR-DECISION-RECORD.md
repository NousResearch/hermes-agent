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

## (c) Disposition of the residual (every ./src delta line)
**Reasoned default: BINDING per-bucket disposition — no "deferred-as-drift" bucket remains.**
- Superseded the earlier "documented drift" framing (which the Council correctly rejected) with
  `RESIDUAL-BINDING-DISPOSITION.md`: every line of `git diff v0.16.0..src-HEAD` is placed in
  exactly one of four buckets, each with a binding home:
  - **Bucket A — in a feature PR (option i):** the bulk of residual hunks live in shared files
    touched by 30+ feature PRs (`agent_init.py`, `auxiliary_client.py`, `conversation_loop.py`,
    `context_engine.py`→#50053, …). Per `DELTA-MAP-v017.md` each file is owned by its feature
    PRs; the residual `.patch` only captured hunks `mechanical_diff_equality` couldn't attribute
    to a SINGLE PR. **Empirically proven**: the `context_engine.py` residual line
    `def capabilities(self) -> Dict[str, bool]:` is present verbatim in PR #50053's branch.
  - **Bucket B — isolated private-feature DRAFT PRs (option i):** agy-cli #50555, auto_router
    #50031, codex #50038, gemini-UA #50033, tool-trace #50021, source-accelerator #50032 — each
    already a draft PR per the user's "isolate as a draft PR" instruction.
  - **Bucket C — formally OUT OF SCOPE (option ii):** 72 added lines carrying genuinely-private
    operational content (account caps, internal build-phase labels, personal filesystem paths)
    that cannot enter a public PR without either leaking private data or rewriting the residual.
    Preserved verbatim in `RESIDUAL-NOT-IN-ANY-PR.patch` on this #50111 branch (re-appliable onto
    v0.16.0, `git apply --check` clean) — which is the campaign's stated goal.
  - **Bucket D — DISCARD:** 9 `.bak` + 12 `.project-intel/` generated artifacts (non-source).
- So OPTION C is no longer "defer": A+B are option (i) (in PRs), D is non-source, and only Bucket
  C (72 private lines) is option (ii) out-of-scope — and that classification is the single
  operator-ratifiable item below.
- **Counter (recorded):** Bucket C's 72 lines technically do not live in a *mergeable* PR. They
  live in the faithful patch on the manifest PR #50111, and the user's own standing policy
  (isolate/exclude private overlay) is the controlling instruction. Forcing them into a public PR
  would leak private data or require rewriting them. **Operator override:** if desired, the patch
  applies onto v0.16.0 and can be pushed as `residual/overlay-preservation` after an
  operator-authorized scrub pass.

## (d) PR-introduced test-failure delta across the actual 40-PR diff scope
**Verified = 0 net, after fixing 2 real defects this round.**
- Full-PR-scope A/B (66 test files exercising the touched source, both trees) — see
  `PR-SCOPE-TEST-VERIFICATION-20260622.md`. This deeper scope (vs the earlier sampled A/B) caught
  TWO genuine PR-introduced defects, both fixed + pushed:
  - **#50047** root-guard case ordering (Docker case now checked before workspace-owner case) —
    all 44 gateway tests pass (`b3695d09a..0da10a6b5`).
  - **#50048** `force_plain` test-signature mismatch (test assertions + fake stub updated) — all
    146 send_message tests pass (`001b549c6..3e0c085d6`).
- 1 remaining integrated-only failure (`test_reasoning_xhigh_honored_for_copilot_gpt5`) is
  overlay-only Bucket-C test content (tests the transport layer; the contributable #49644
  implements the clamp in `run_agent.py` and its own 73 reasoning tests pass) — not a
  contributable-PR defect.
- The "clean-only" failures (FTS5 optimize, bedrock, kanban) are pre-existing **upstream** v0.17.0
  bugs that our PR set FIXES (pass on integrated, fail on clean). Net-positive on test health.

## Status of this determination
This is RECORDED, not declared-complete. The three are operator-overridable. The autopilot
run rests at a verified checkpoint: 40 contributable PRs + #50758, 40/40 apply onto
v0.17.0, full-unit-suite running clean, one real defect (#50064) caught-and-fixed, residual
fully enumerated. The operator's ruling on (a)/(b)/(c) flips this from checkpoint to final.
