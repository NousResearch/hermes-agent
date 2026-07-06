# HQ Hermes local ahead commit classification — 2026-07-05

## Scope

- Repo: `C:/Users/82109/AppData/Local/hermes/hermes-agent`
- Branch: `main`
- HEAD: `8b97f5e51`
- origin/main: `7fde19afc`
- Basis: `git fetch --prune origin`, then `origin/main...HEAD` and net tree diff.
- This is a **classification / preservation report**, not a merge, reset, rebase, or cherry-pick.

## Verified live state

```text
## main...origin/main [ahead 74, behind 84]
?? reports/hq_hermes_pre_update_audit_20260705_043838.md
?? reports/hq_local_ahead_74_commit_inventory_20260705.json

git rev-list --left-right --count origin/main...HEAD = 84	74
git diff --shortstat origin/main..HEAD = 240 files changed, 11456 insertions(+), 6063 deletions(-)
git cherry -v origin/main HEAD = 61 non-merge local patch commits, all reported as +
```

Interpretation: local `main` is divergent. There are 74 local-only commits by ancestry, 61 non-merge local patch commits, and 13 merge/integration commits. No non-merge local patch commit was reported patch-equivalent to current `origin/main` by `git cherry`.

## Classification counts

- `bugfix`: 33
- `adopt`: 11
- `integrate-better`: 13
- `defer`: 4
- `delete-by-policy`: 13

## High-level recommendation

1. Do **not** fast-forward, reset, or merge directly from this active `main`. Preserve this state first.
2. Rebuild a clean candidate from current `origin/main` and port logical stacks, not the raw 74-commit history.
3. Highest priority to port: clarify approval/choice UX bugfix stack, Desktop/Gateway runtime reliability fixes, Windows restart/ws-loop fixes.
4. Treat merge/update glue commits as `delete-by-policy` for the clean candidate: they are evidence of prior integration attempts, not logical product changes to cherry-pick.
5. Defer broad HQ harness/eval/health-dashboard and optional haptics work unless the immediate goal is HQ AGI/eval infrastructure rather than Hermes update safety.

## Duplicate/restaged subject groups

- 4× `merge: refresh fork main total candidate with latest upstream main`
- 3× `fix(desktop): tolerate delayed clarify responses`
- 2× `feat(clarify): restore native multi-select prompts`
- 2× `fix(clarify): cover task report scope choices`
- 2× `fix(clarify): enforce constrained selection responses`
- 2× `fix(clarify): enforce multi-select response bounds`
- 2× `fix(clarify): harden constrained-choice semantics`
- 2× `fix(clarify): keep multi-select choice ux visible`
- 2× `fix(clarify): reject empty required multi-select replies`
- 2× `fix(clarify): reject mixed constrained responses`
- 2× `fix(desktop): bound live terminal output events`
- 2× `fix(desktop): distinguish lost gateway and bound tool result events`
- 2× `fix(desktop): drop unused clarify import`
- 2× `fix(desktop): explain prompt submit timeouts`
- 2× `fix(desktop): tolerate slow backend readiness during startup`
- 2× `merge: bring fork main candidate onto latest upstream main`
- 2× `style(desktop): satisfy clarify lint spacing`
- 2× `style(desktop): satisfy restaged prompt submit lint`

## Full 74-commit classification table

| # | Commit | Subject | Area | Class | Rationale |
|---:|---|---|---|---|---|
| 1 | `cf5cf315a` | fix(desktop): tolerate delayed clarify responses | desktop | `bugfix` | HQ-required clarify approval/choice UX reliability; preserve, but squash/deduplicate repeated restaged commits into one clean stack; duplicate subject appears multiple times |
| 2 | `1492c1718` | fix(desktop): distinguish lost gateway and bound tool result events | desktop,agent,tests | `bugfix` | runtime/Desktop/Gateway reliability fix; preserve after current-origin conflict review and focused tests |
| 3 | `18e8f4465` | fix(desktop): bound live terminal output events | tests | `bugfix` | runtime/Desktop/Gateway reliability fix; preserve after current-origin conflict review and focused tests |
| 4 | `d7e347c1f` | merge: bring fork main candidate onto latest upstream main | misc | `delete-by-policy` | merge/integration glue; rebuild a clean candidate from current origin/main rather than carrying this merge commit as a logical change |
| 5 | `a73b6e648` | feat(desktop): add file-backed session pins bridge | desktop | `adopt` | matches HQ Desktop/session organization preference; preserve as a user-facing Desktop improvement |
| 6 | `c1710d52e` | fix(gateway): harden Windows restart path | gateway,tests | `bugfix` | runtime/Desktop/Gateway reliability fix; preserve after current-origin conflict review and focused tests |
| 7 | `723cdf2ab` | feat(desktop): sort numbered sessions consistently | desktop | `adopt` | matches HQ Desktop/session organization preference; preserve as a user-facing Desktop improvement |
| 8 | `2bd044507` | fix(desktop): explain prompt submit timeouts | desktop | `bugfix` | runtime/Desktop/Gateway reliability fix; preserve after current-origin conflict review and focused tests |
| 9 | `b7c7d78e7` | feat(clarify): restore native multi-select prompts | desktop,gateway,tools,agent,tests | `bugfix` | HQ-required clarify approval/choice UX reliability; preserve, but squash/deduplicate repeated restaged commits into one clean stack; duplicate subject appears multiple times |
| 10 | `336352acd` | fix(clarify): harden constrained-choice semantics | desktop,gateway,tools,tests | `bugfix` | HQ-required clarify approval/choice UX reliability; preserve, but squash/deduplicate repeated restaged commits into one clean stack; duplicate subject appears multiple times |
| 11 | `33f588e87` | fix(desktop): tolerate slow backend readiness during startup | desktop | `bugfix` | runtime/Desktop/Gateway reliability fix; preserve after current-origin conflict review and focused tests |
| 12 | `3f17156b7` | fix(clarify): keep multi-select choice UX visible | desktop,tools,tests | `bugfix` | HQ-required clarify approval/choice UX reliability; preserve, but squash/deduplicate repeated restaged commits into one clean stack; duplicate subject appears multiple times |
| 13 | `f2f675a46` | fix(clarify): enforce multi-select response bounds | tools,tests | `bugfix` | HQ-required clarify approval/choice UX reliability; preserve, but squash/deduplicate repeated restaged commits into one clean stack; duplicate subject appears multiple times |
| 14 | `cb0410001` | fix(clarify): reject empty required multi-select replies | desktop,tools,tests | `bugfix` | HQ-required clarify approval/choice UX reliability; preserve, but squash/deduplicate repeated restaged commits into one clean stack; duplicate subject appears multiple times |
| 15 | `d5e24331d` | style(desktop): satisfy clarify lint spacing | desktop | `integrate-better` | style/lint-only support commit; do not carry standalone, squash into the related logical change if still needed |
| 16 | `65fda9573` | chore: checkpoint HQ Hermes workspace updates | desktop,gateway,tools,cli/config,tests,skills,scripts | `defer` | very broad checkpoint commit spanning many areas; split into logical changes before adoption |
| 17 | `6a1f376ec` | feat: add safe desktop control routing status surface | tools,cli/config,tests,docs | `integrate-better` | valuable safety/control capability, but touches tool surface and should be reviewed against current upstream architecture before adoption |
| 18 | `2d76f95ea` | feat(computer-use): add safe input proposals | tools,tests | `integrate-better` | valuable safety/control capability, but touches tool surface and should be reviewed against current upstream architecture before adoption |
| 19 | `6e5145ef1` | feat(computer-use): add browser input executor | tools,tests | `integrate-better` | valuable safety/control capability, but touches tool surface and should be reviewed against current upstream architecture before adoption |
| 20 | `9f85bc2b6` | feat(computer-use): add Windows UIA read-only enumeration | tools,tests | `integrate-better` | valuable safety/control capability, but touches tool surface and should be reviewed against current upstream architecture before adoption |
| 21 | `0590ae4fe` | feat(computer-use): add safe local CDP browser backend | tools,tests | `integrate-better` | valuable safety/control capability, but touches tool surface and should be reviewed against current upstream architecture before adoption |
| 22 | `4202eaf95` | chore: normalize fork main total whitespace | tests | `adopt` | small test/docs portability or expectation update; low-risk to keep after adjacent area verification |
| 23 | `efa2f10ae` | feat: add HQ harness evidence dashboard integration | tests,scripts | `defer` | HQ/AGI-evaluation infrastructure is valuable but broad/speculative; preserve report and review separately from Desktop/Gateway update path |
| 24 | `a12326a22` | feat(hq): add eval gate quality harness | gateway,cron,agent,cli/config,tests,reports,scripts | `defer` | HQ/AGI-evaluation infrastructure is valuable but broad/speculative; preserve report and review separately from Desktop/Gateway update path |
| 25 | `2e42f2504` | merge: refresh fork main total candidate with latest upstream main | misc | `delete-by-policy` | merge/integration glue; rebuild a clean candidate from current origin/main rather than carrying this merge commit as a logical change |
| 26 | `7987663d0` | fix: stabilize fork main total verification on Windows | desktop,gateway,cron,tools,cli/config,tests | `integrate-better` | needs manual review against current origin/main before adoption |
| 27 | `9a17d6b24` | merge: refresh fork main total candidate with latest upstream main | misc | `delete-by-policy` | merge/integration glue; rebuild a clean candidate from current origin/main rather than carrying this merge commit as a logical change |
| 28 | `7be1b00e8` | test: keep Windows env for gateway config bridge subprocess | tests | `adopt` | small test/docs portability or expectation update; low-risk to keep after adjacent area verification |
| 29 | `8ebe8c492` | merge: refresh fork main total candidate with latest upstream main | misc | `delete-by-policy` | merge/integration glue; rebuild a clean candidate from current origin/main rather than carrying this merge commit as a logical change |
| 30 | `0e90dea3b` | test: normalize profile isolation path assertions on Windows | tests | `adopt` | small test/docs portability or expectation update; low-risk to keep after adjacent area verification |
| 31 | `46193b8b8` | docs: fix desktop control routing MDX links | docs | `adopt` | small test/docs portability or expectation update; low-risk to keep after adjacent area verification |
| 32 | `74bb8c18f` | merge: refresh fork main total candidate with latest upstream main | misc | `delete-by-policy` | merge/integration glue; rebuild a clean candidate from current origin/main rather than carrying this merge commit as a logical change |
| 33 | `b4d1beb4e` | test: make new runtime guard tests Windows portable | tests | `adopt` | small test/docs portability or expectation update; low-risk to keep after adjacent area verification |
| 34 | `dba26fbac` | fix(desktop): tolerate delayed clarify responses | desktop | `bugfix` | HQ-required clarify approval/choice UX reliability; preserve, but squash/deduplicate repeated restaged commits into one clean stack; duplicate subject appears multiple times |
| 35 | `307924c72` | fix(desktop): distinguish lost gateway and bound tool result events | desktop,agent,tests | `bugfix` | runtime/Desktop/Gateway reliability fix; preserve after current-origin conflict review and focused tests |
| 36 | `6527445b1` | fix(desktop): bound live terminal output events | tests | `bugfix` | runtime/Desktop/Gateway reliability fix; preserve after current-origin conflict review and focused tests |
| 37 | `3051d4f0d` | fix(desktop): explain prompt submit timeouts | desktop | `bugfix` | runtime/Desktop/Gateway reliability fix; preserve after current-origin conflict review and focused tests |
| 38 | `b3123618c` | feat(clarify): restore native multi-select prompts | desktop,gateway,tools,agent,tests | `bugfix` | HQ-required clarify approval/choice UX reliability; preserve, but squash/deduplicate repeated restaged commits into one clean stack; duplicate subject appears multiple times |
| 39 | `13c8445f4` | fix(clarify): harden constrained-choice semantics | desktop,gateway,tools,tests | `bugfix` | HQ-required clarify approval/choice UX reliability; preserve, but squash/deduplicate repeated restaged commits into one clean stack; duplicate subject appears multiple times |
| 40 | `2e7d55643` | fix(desktop): tolerate slow backend readiness during startup | desktop | `bugfix` | runtime/Desktop/Gateway reliability fix; preserve after current-origin conflict review and focused tests |
| 41 | `45e737a75` | fix(clarify): keep multi-select choice UX visible | desktop,tools,tests | `bugfix` | HQ-required clarify approval/choice UX reliability; preserve, but squash/deduplicate repeated restaged commits into one clean stack; duplicate subject appears multiple times |
| 42 | `e78c7f85b` | fix(clarify): enforce multi-select response bounds | tools,tests | `bugfix` | HQ-required clarify approval/choice UX reliability; preserve, but squash/deduplicate repeated restaged commits into one clean stack; duplicate subject appears multiple times |
| 43 | `fa522dae9` | fix(clarify): reject empty required multi-select replies | desktop,tools,tests | `bugfix` | HQ-required clarify approval/choice UX reliability; preserve, but squash/deduplicate repeated restaged commits into one clean stack; duplicate subject appears multiple times |
| 44 | `c11ada21e` | style(desktop): satisfy clarify lint spacing | desktop | `integrate-better` | style/lint-only support commit; do not carry standalone, squash into the related logical change if still needed |
| 45 | `edfa58b50` | fix(desktop): drop unused clarify import | desktop | `integrate-better` | needs manual review against current origin/main before adoption |
| 46 | `d4ba30a27` | fix(clarify): enforce constrained selection responses | desktop,tools,agent,tests | `bugfix` | HQ-required clarify approval/choice UX reliability; preserve, but squash/deduplicate repeated restaged commits into one clean stack; duplicate subject appears multiple times |
| 47 | `35aa9f395` | fix(clarify): reject mixed constrained responses | tools,tests | `bugfix` | HQ-required clarify approval/choice UX reliability; preserve, but squash/deduplicate repeated restaged commits into one clean stack; duplicate subject appears multiple times |
| 48 | `13eaee13c` | style(desktop): satisfy restaged prompt submit lint | desktop | `integrate-better` | style/lint-only support commit; do not carry standalone, squash into the related logical change if still needed |
| 49 | `2aa38fba5` | fix(clarify): cover task report scope choices | tools,tests | `bugfix` | HQ-required clarify approval/choice UX reliability; preserve, but squash/deduplicate repeated restaged commits into one clean stack; duplicate subject appears multiple times |
| 50 | `76f87f8a9` | merge: bring fork main candidate onto latest upstream main | misc | `delete-by-policy` | merge/integration glue; rebuild a clean candidate from current origin/main rather than carrying this merge commit as a logical change |
| 51 | `61505325e` | fix(desktop): tolerate delayed clarify responses | desktop | `bugfix` | HQ-required clarify approval/choice UX reliability; preserve, but squash/deduplicate repeated restaged commits into one clean stack; duplicate subject appears multiple times |
| 52 | `aaf04e0f5` | fix(desktop): drop unused clarify import | desktop | `integrate-better` | needs manual review against current origin/main before adoption |
| 53 | `e8368cfc7` | fix(clarify): enforce constrained selection responses | desktop,tools,agent,tests | `bugfix` | HQ-required clarify approval/choice UX reliability; preserve, but squash/deduplicate repeated restaged commits into one clean stack; duplicate subject appears multiple times |
| 54 | `6f3029374` | fix(clarify): reject mixed constrained responses | tools,tests | `bugfix` | HQ-required clarify approval/choice UX reliability; preserve, but squash/deduplicate repeated restaged commits into one clean stack; duplicate subject appears multiple times |
| 55 | `8ba27ee18` | style(desktop): satisfy restaged prompt submit lint | desktop | `integrate-better` | style/lint-only support commit; do not carry standalone, squash into the related logical change if still needed |
| 56 | `ff795c13b` | fix(clarify): cover task report scope choices | tools,tests | `bugfix` | HQ-required clarify approval/choice UX reliability; preserve, but squash/deduplicate repeated restaged commits into one clean stack; duplicate subject appears multiple times |
| 57 | `9cbdedeff` | fix(clarify): honor configured Desktop clarify timeout | misc | `bugfix` | HQ-required clarify approval/choice UX reliability; preserve, but squash/deduplicate repeated restaged commits into one clean stack |
| 58 | `51150b9c7` | fix(clarify): resolve latest-origin Desktop integration | desktop | `bugfix` | HQ-required clarify approval/choice UX reliability; preserve, but squash/deduplicate repeated restaged commits into one clean stack |
| 59 | `2d43187f7` | style(desktop): sort project store imports | desktop | `integrate-better` | style/lint-only support commit; do not carry standalone, squash into the related logical change if still needed |
| 60 | `4607d1758` | fix(desktop): expire stale clarify requests | desktop | `bugfix` | HQ-required clarify approval/choice UX reliability; preserve, but squash/deduplicate repeated restaged commits into one clean stack |
| 61 | `6b950d625` | merge: bring total candidate onto latest upstream main | misc | `delete-by-policy` | merge/integration glue; rebuild a clean candidate from current origin/main rather than carrying this merge commit as a logical change |
| 62 | `ea9c92334` | test(config): expect platform-native Hermes home | tests | `adopt` | small test/docs portability or expectation update; low-risk to keep after adjacent area verification |
| 63 | `e9db66991` | merge: refresh total candidate onto current upstream main | misc | `delete-by-policy` | merge/integration glue; rebuild a clean candidate from current origin/main rather than carrying this merge commit as a logical change |
| 64 | `533cca9ab` | merge: record active main stack as integrated into total candidate | misc | `delete-by-policy` | merge/integration glue; rebuild a clean candidate from current origin/main rather than carrying this merge commit as a logical change |
| 65 | `52206678e` | merge: refresh total candidate before active apply | misc | `delete-by-policy` | merge/integration glue; rebuild a clean candidate from current origin/main rather than carrying this merge commit as a logical change |
| 66 | `2875dad0c` | fix: harden desktop ws loop stall diagnostics | cli/config,tests | `bugfix` | runtime/Desktop/Gateway reliability fix; preserve after current-origin conflict review and focused tests |
| 67 | `545277df4` | feat: gate desktop haptics debug telemetry | desktop | `defer` | Desktop haptics telemetry is optional UX/debug work; keep preserved but lower priority for update readiness |
| 68 | `10b092d82` | test: sync desktop renderer expectations | desktop | `adopt` | small test/docs portability or expectation update; low-risk to keep after adjacent area verification |
| 69 | `d00f0f4e9` | chore: clean up desktop build warnings | desktop | `adopt` | contained Desktop build/perf cleanup; keep if current Desktop tests/build pass |
| 70 | `8bcbc5179` | perf(desktop): import Tabler icons directly | desktop | `adopt` | contained Desktop build/perf cleanup; keep if current Desktop tests/build pass |
| 71 | `dc065158f` | chore(hq): stage active main update integration | misc | `delete-by-policy` | merge/integration glue; rebuild a clean candidate from current origin/main rather than carrying this merge commit as a logical change |
| 72 | `d6c19f820` | fix(desktop): restore clarify keyboard shortcuts | desktop | `bugfix` | HQ-required clarify approval/choice UX reliability; preserve, but squash/deduplicate repeated restaged commits into one clean stack |
| 73 | `55e9e47eb` | merge: integrate origin/main security drift | misc | `delete-by-policy` | merge/integration glue; rebuild a clean candidate from current origin/main rather than carrying this merge commit as a logical change |
| 74 | `8b97f5e51` | merge: integrate latest origin/main drift | misc | `delete-by-policy` | merge/integration glue; rebuild a clean candidate from current origin/main rather than carrying this merge commit as a logical change |

## Net-diff feature buckets to port/review

### Priority A — preserve as bugfixes

- Clarify native multi-select / constrained-choice / empty-response / stale-request / keyboard-shortcut stack.
- Desktop/Gateway runtime reliability: delayed tool results, lost gateway distinction, live terminal output bounds, slow backend readiness, prompt-submit timeout explanations.
- Windows Gateway restart path and Desktop websocket loop stall diagnostics.

### Priority B — adopt if focused verification passes

- Numbered session sorting and file-backed session pins bridge.
- Small Windows/test/docs portability fixes.
- Desktop build-warning and icon import performance cleanup.

### Priority C — integrate-better / split before porting

- Safe desktop control routing and computer-use proposal/browser/CDP/UIA stack: valuable, but tool-surface and safety-sensitive.
- Broad checkpoint commit `65fda9573`: split into separate Desktop/settings/tool/skill changes before adoption.
- Style/lint support commits: squash into their parent logical fixes, not standalone.

### Priority D — defer

- HQ eval gate / harness / health dashboard / memory-trust research artifacts: preserve for the HQ AGI/eval roadmap, but keep out of the immediate Hermes update candidate unless explicitly approved.
- Desktop haptics debug telemetry: optional UX/debug work; defer unless current Desktop target needs it.

### Drop from clean candidate history

- Raw merge/update glue commits such as `merge: refresh...`, `merge: integrate origin/main...`, and `chore(hq): stage active main update integration`. Reconstruct a clean branch from `origin/main` instead.

## Suggested next read-only commands

```bash
cd /c/Users/82109/AppData/Local/hermes/hermes-agent
# Review only the Priority A stack subjects
git log --oneline --reverse origin/main..HEAD --grep='clarify\|gateway\|desktop ws\|restart\|prompt submit' --regexp-ignore-case
# Compare final tree for clarify-related files
git diff --stat origin/main..HEAD -- tools/clarify_tool.py tools/clarify_gateway.py agent/tool_executor.py apps/desktop/src/components/assistant-ui/clarify-tool.tsx
```

## Notes and caveats

- This report intentionally avoids applying, staging, rebasing, resetting, or cherry-picking any commit.
- It uses subjects/paths/net diff as classification evidence. A final adoption decision still needs current-origin conflict review and focused tests per feature stack.
- The existing untracked pre-update audit report is preserved; this report and the JSON inventory are new untracked report artifacts.
