# Dev PR Automation Reconciler

Hermes can run PR automation without a public GitHub webhook by polling GitHub
with `gh` from the stable gateway host.

## Command

```bash
HERMES_HOME="$HOME/.hermes/profiles/dev" \
scripts/run_dev_pr_automation_reconciler.py \
  --db-path "$HOME/.hermes/profiles/dev/state.db" \
  --repos Felippen/Oryn,Felippen/hermes-agent,Felippen/hermes-ops
```

The command performs one bounded pass:

- list open PRs in the managed repos
- refresh Hermes PR state and merge readiness
- request Copilot review once per PR head when missing
- delegate fixes only when `hermes:auto-fix` is present
- merge only when `hermes:auto-merge` is present and all merge gates pass
- release only for merged `Felippen/Oryn` PRs with `hermes:auto-release`

## Cadence

Run from launchd or cron every 2-5 minutes. The script uses a lock file
(`/tmp/hermes_dev_pr_automation_reconciler.lock` by default), so overlapping
runs exit without doing work.

## Safety

No labels means observe-only. The reconciler records state and audit rows, but
does not mutate PR branches.

Auto-merge remains blocked unless both are set:

```bash
HERMES_DEV_MERGE_EXECUTOR_ENABLED=true
HERMES_DEV_BRANCH_PROTECTION_CONFIRMED=true
```

Keep those disabled until GitHub branch protection or an equivalent independent
CI backstop is confirmed.
