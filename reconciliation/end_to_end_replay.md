# End-to-end replay — stack all PRs on clean v0.17.0 (Council #1)

Method: from a fresh v0.17.0 (2bd1977d) worktree, apply every feature PR's net diff
(stacked), then verify every overlay-added SOURCE line is present in the replayed tree
or in the deferred set.

## Result
- **38/39 PRs stacked** onto clean v0.17.0 (#50056 skipped on stack because its sibling
  #50073 already carries the shared lines — normal for a stacked replay).
- **10 overlay-added lines not literally present** — and ALL 10 are lines my OWN fixes
  this session DELIBERATELY IMPROVED (the replay tree has the better version):
  - 7 in `test_model_metadata.py`: the `gpt-5.4: 900_000` empirical-override lines I
    reverted in #50078 (public PR keeps 272K; the 900K is deferred/account-specific).
  - 2 in `google_user_agent.py` + 1 in `copilot_auth.py`: bare `read_text()`/`write_text()`
    that I fixed by adding `encoding="utf-8"` (#50033, #50064 — the repo's PLW1514 rule).
- So the replayed tree is a **superset of the overlay's intent**: every overlay change is
  present, plus 3 legitimate improvements (encoding-safe I/O, public-value test assertions)
  that supersede the overlay's original lines. 0 overlay intent is lost.

## Independent spot-check (Council #3 — method NOT the agent's own scripts)
Plain `git apply` (manual, not verify_all_prs_on_v017.sh) of 3 PRs onto a fresh v0.17.0:
- #48024 (5 files, 1200 lines): APPLIED clean, exit 0
- #50146 (2 files, 146 lines): APPLIED clean, exit 0
- #50046 (6 files, 609 lines): APPLIED clean, exit 0

## The 3 explained deleted lines (Council #4 — findable here)
`tests/agent/test_model_metadata.py` — the overlay DELETED the public `gpt-5.4: 272_000`
expectation (changing it to the private `900_000` override). PR #50078 INTENTIONALLY keeps
`272_000` (the 900K override is account-specific, deferred per the copilot-limits category).
So the overlay's deletion-of-272K is correctly NOT in the public PR — the private 900K
change lives in the deferred set (`private-feature-mixed/`). This is documented here and in
`complete_diff_coverage.md` so it is locatable without re-deriving.
