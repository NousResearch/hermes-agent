# Background Memory Curation — Validation Handoff (2026-10-29)

## Status

Focused memory and write-approval tests passed (93 tests), and all 14 repository checks passed after rebasing onto `origin/main`. The PR is open and GitHub reports it `MERGEABLE`; no CI checks have been reported yet. The full suite was not run locally. The active profile config already contains `memory.background_review_allow_deletes: true`, but Hermes v0.22.5 does not read this new setting; it takes effect only after installing a revision that contains this implementation.

## Branches and PRs

| Repository | Branch | HEAD | PR | Mergeable |
|---|---|---|---|---|
| `NousResearch/hermes-agent` | `fix/background-memory-auto-curation` | `e3d8b8056f3d1379abacad28c51b6ffa326a02e7` | #146671 | MERGEABLE |

- PR #146671: https://github.com/NousResearch/hermes-agent/pull/146671
- Merge order: standalone; no dependent PR.

## Validation results

### Hermes Agent — PR #146671

| Gate | Command | Result |
|---|---|---|
| Focused tests | `scripts/run_tests.sh tests/tools/test_memory_tool.py tests/tools/test_write_approval.py -q` | 2 files, 93 tests passed, 0 failed |
| Repository checks | `PATH="$HOME/.local/bin:$PATH" python scripts/check` | 14 checks passed; code health 0 blocking, 0 advisory |
| Baseline code health | Same repository check vs `origin/main` `088f657ecf77` | 0 new blocking or advisory findings |
| Remote PR | `gh pr view 146671 --repo NousResearch/hermes-agent` | OPEN, head `e3d8b8056f3d1379abacad28c51b6ffa326a02e7`, MERGEABLE |
| CI | `gh pr checks 146671 --repo NousResearch/hermes-agent` | No checks reported at measurement time |
| Remote branch | `git ls-remote joe-fork refs/heads/fix/background-memory-auto-curation` | Head matches PR SHA |
| Working tree | `git status --short` | Clean after handoff commit |

## Baseline comparison

`origin/main` at `088f657ecf77` was the base used for the rebase and the code-health baseline. The focused test files passed on the rebased branch. The full suite was not run, so no full-suite failure comparison is claimed. GitHub had not reported CI checks at the last measurement.

## Reproduce

```bash
cd /Users/m1n1/.hermes/worktrees/background-memory-auto-curation
scripts/run_tests.sh tests/tools/test_memory_tool.py tests/tools/test_write_approval.py -q
PATH="$HOME/.local/bin:$PATH" python scripts/check
gh pr view 146671 --repo NousResearch/hermes-agent
gh pr checks 146671 --repo NousResearch/hermes-agent
```

## Remote state — last measured (2026-10-29)

The PR is OPEN and MERGEABLE. Its fork branch points to `e3d8b8056f3d1379abacad28c51b6ffa326a02e7`; no CI checks had been reported. Re-check the PR and checks before merging:

```bash
gh pr view 146671 --repo NousResearch/hermes-agent --json state,mergeable,headRefOid,url
gh pr checks 146671 --repo NousResearch/hermes-agent
```

## What this does not validate

- The full Hermes Agent test suite was not run locally.
- No CI checks were reported when measured; CI may start or update later.
- The opt-in does not restore entries already removed or rewritten. Backups are required for recovery.
- The already-set config key is ignored by Hermes v0.22.5 until the implementation is installed.

## Eyeball checklist

- `tools/memory_tool.py`: verify only unattended background review is gated by the new opt-in.
- `hermes_cli/config_defaults.py` and `cli-config.yaml.example`: verify the default is false.
- `tests/tools/test_memory_tool.py` and `tests/tools/test_write_approval.py`: verify default staging, opted-in edits, atomic batch behavior, and existing approval semantics.
- `website/docs/user-guide/features/memory.md`: verify scope and recovery instructions.
- No secrets are part of the diff; only code, tests, config documentation, and this handoff are included.

## Reversion

See PR #146671’s **Reversion criteria and rollback** section. To stop future destructive background edits, set `memory.background_review_allow_deletes: false` (or remove the key) in the active profile config. Reverting the implementation is `git revert e3d8b8056f` on the branch. Restoring prior memory requires a backup of the active profile’s `$HERMES_HOME/memories/MEMORY.md` or `USER.md`.
