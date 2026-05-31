---
name: recover-fork-sync
description: >-
  Recover pebble-tech/hermes-agent when the daily upstream sync workflow fails.
  Use when a sync-failure GitHub issue is opened, a Cursor automation webhook
  fires with event sync-failure, or the user asks to fix fork sync / rebase
  FEATURE_BRANCHES / ops-overlay. Read FORK.md first. Self-contained in this
  repo — do not depend on nudge-hermes-agents.
---

# Recover fork upstream sync

Fix **source branches** so the scheduled `sync-upstream.yml` workflow can rebuild
integration `main`. Never patch `main` directly — sync force-rebuilds it from
upstream + cherry-picks.

## Read first

1. [FORK.md](../../../FORK.md) — branch model and recovery overview
2. [FORK_SYNC_AUTOMATION.md](../../../FORK_SYNC_AUTOMATION.md) — webhook + secrets setup
3. Payload / issue body — machine-readable `sync-failure` block (below)

## Remotes

| Remote | Repo | Use |
|--------|------|-----|
| `origin` | `pebble-tech/hermes-agent` | Push fixes here |
| `upstream` | `NousResearch/hermes-agent` | Fetch only; rebase onto `upstream/main` |

After clone: `git fetch origin upstream --prune`

**Local `main` must track `origin/main`**, not `upstream/main`:

```bash
git branch -u origin/main main
```

## Inputs (webhook or issue)

Parse the `sync-failure` YAML block or webhook JSON:

| Field | Meaning |
|-------|---------|
| `upstream_sha` | NousResearch `main` at failure time |
| `failed_step` | `rebase-ops-overlay`, `rebase-feature-branch`, `cherry-pick-main`, `focused-tests`, `plugin-smoke-test`, `push` |
| `failed_branch` | Source branch to fix (`ops-overlay` or a `FEATURE_BRANCHES` entry) |
| `conflict_files` | Paths from rebase/cherry-pick |
| `workflow_run_url` | Actions log |
| `issue_number` | Open `sync-failure` issue (if any) |

If `failed_branch` is missing, read the workflow log URL for the failing step.

## Decision tree

### A. Upstream PR closed / change absorbed upstream

Symptoms: rebase conflict on a `FEATURE_BRANCHES` entry; upstream already has
equivalent code; NousResearch PR closed without merge or merged elsewhere.

1. Confirm on `upstream/main` (grep / log / compare).
2. **PR or push to `ops-overlay` only:**
   - Remove branch from `FEATURE_BRANCHES` in `.github/workflows/sync-upstream.yml`
   - Update `FORK.md` upstream PR table
   - Short commit message: `ops(fork): drop #NNNN <branch> from sync`
3. Delete dead branch from origin: `git push origin --delete <branch>`
4. Do **not** rebase the obsolete feature branch.

Example: #17165 WhatsApp text batch superseded by upstream `b0ce47daa` → drop
`feature/whatsapp-inbound-text-batch` from `FEATURE_BRANCHES`.

### B. Rebase conflict — patch still needed

1. `git checkout -B <failed_branch> origin/<failed_branch>`
2. `git rebase upstream/main` — resolve conflicts
3. Run focused tests (see below)
4. `git push --force-with-lease origin <failed_branch>`

### C. Cherry-pick conflict rebuilding `main`

Two feature branches touch the same lines. Fix the **later** branch in
`FEATURE_BRANCHES` order, or reorder/drop one via ops-overlay if redundant.

### D. Test or plugin smoke failure

Fix on the branch that introduced the regression (usually the feature branch
whose commits fail tests after rebase). Run the same focused tests locally.

## Focused tests (match CI)

```bash
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[all,dev]"
sudo apt-get update && sudo apt-get install -y ripgrep   # Linux CI only
python -m pytest \
  tests/gateway/test_pre_gateway_dispatch.py \
  tests/gateway/test_session.py \
  tests/hermes_cli/test_plugins.py \
  -q --tb=short
```

## After fix — trigger sync

Prefer opening a **PR to the source branch** (`ops-overlay` or feature/*) for
review. After merge (or direct push if automation policy allows):

```bash
gh workflow run sync-upstream.yml --repo pebble-tech/hermes-agent --ref main
gh run list --repo pebble-tech/hermes-agent --workflow sync-upstream.yml --limit 1
# wait for success
```

Verify:

```bash
git fetch origin upstream
git rev-list --count origin/main..upstream/main   # should be 0 after successful sync
```

## Closeout

1. Confirm sync workflow **success**
2. Comment on the `sync-failure` issue with what changed
3. Close the issue

## Hard rules

- Do **not** commit recovery fixes to integration `main`
- Do **not** leave `main` tracking `upstream/main`
- Do **not** add obsolete branches back to `FEATURE_BRANCHES`
- Minimize scope — fork ops only, no unrelated Hermes core changes
