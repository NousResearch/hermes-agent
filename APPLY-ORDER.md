# Operator replay guide — apply the PR set onto v0.17.0 (or any later release)

This manifest re-applies every (contributable) private-overlay change onto a fresh upstream
release without reconstructing anything by hand. Proven against
**v0.17.0 = `2bd1977d8fad185c9b4be47884f7e87f1add0ce3`** by `verification/stack-apply-v017.sh`
(SET-LEVEL: all PRs applied together on ONE tree) and `verification/verify-patches.sh`
(the 6 forward-port patches).

## Current open set: 39 code PRs (the #50111 manifest itself is NOT a code PR)

**Withdrawn / not in the set** (do NOT apply — maintainer-aligned, see
`MAINTAINER-FEEDBACK-DISPOSITION.md`): #50033 (gemini-cli-UA), #50555 + #50657 (agy-cli) — all
CLOSED for Google-account-ban safety (#50492 removed that provider category upstream); #50039,
#50049, #50484 closed earlier (superseded / duplicate / invalid-overlay-residual).

## TL;DR — the set stacks cleanly, order-independent

`stack-apply-v017.sh` applies all 39 open PRs onto one v0.17.0 tree:
**39/39 CLEAN, 0 conflicts, 130 changed .py compile, 0 PR-introduced test failures**
(the 1 slice failure is pre-existing on bare v0.17.0). You do **not** need a strict dependency chain.

## Deterministic recipe

```bash
V017=2bd1977d8fad185c9b4be47884f7e87f1add0ce3
git checkout -b replay/on-v0.17.0 "$V017"

# 1. The 33 clean PRs — cherry-pick each PR's net diff (git diff origin/main...<head>),
#    or `git apply --3way`. Order-independent (disjoint / non-drifted files):
#    #48024 #48057 #48065 #48069 #48101 #49184 #49449 #49915 #49917 #50021 #50022 #50031 #50032 #50038 #50040 #50041 #50042 #50045 #50046 #50047 #50048 #50053 #50054 #50055 #50066 #50068 #50078 #50080 #50086 #50146 #50155 #50626 #50664
#
# 2. The 6 forward-port PRs — apply the PRE-RESOLVED v0.17.0 patch (NOT the raw PR), each
#    carrying its conflict resolution baked in (verified apply-clean + tests pass):
#    git apply v017-patches/PR-49644-onto-v0.17.0.patch   # /reasoning subcommands (take-theirs superset)
#    git apply v017-patches/PR-49916-onto-v0.17.0.patch   # YOLO badge fix (take-theirs)
#    git apply v017-patches/PR-50056-onto-v0.17.0.patch   # kanban_db imports (combine sqlite3+subprocess)
#    git apply v017-patches/PR-50064-onto-v0.17.0.patch   # copilot identity (drop stale test)
#    git apply v017-patches/PR-50073-onto-v0.17.0.patch   # compression config (combine keys)
#    git apply v017-patches/PR-50296-onto-v0.17.0.patch   # review isolation (take-theirs add)
```

## Verification

- `bash verification/stack-apply-v017.sh`  — set-level: 39/39 stack CLEAN on v0.17.0.
- `bash verification/verify-patches.sh`     — the 6 forward-port patches apply-clean + tests pass (6/6).
- `bash verification/reproduce-coverage.sh` — 165 src-delta = 131 in open PRs + 25 DISCARD + 9 withdrawn + 0 orphans.
