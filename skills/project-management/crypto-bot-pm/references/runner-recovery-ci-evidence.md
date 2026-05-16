# Runner recovery and CI evidence notes

Use this reference for gated local Gitea runner recovery when S006 has an open PR but CI evidence is pending.

## Conservatively consume runner-recovery approval

1. Run the control-plane preflight quartet first.
2. Run `tools/crypto_bot_gitea_runner_recovery.py --inspect --format json` before any execution.
3. If inspect reports `PASS`, the runner container is up, registration succeeded, and token/instance empty loops are absent, do not run `--execute` or recreate the container. The approval has been satisfied by the already-compliant state.
4. Continue only with read-only CI/PR evidence probes unless a new exact approval expands scope.

## Read-only evidence to collect

- Runner container name/image/network/labels from the helper and `docker ps`.
- `workflow_dispatch_invoked`, `pr_mutation_invoked`, and `merge_invoked` flags from the helper report.
- PR/CI audit JSON path and `ci.statuses_count`.
- Direct read-only Gitea combined status for the S006 head SHA when useful.

## CI pending despite healthy runner

If runner recovery is healthy but CI remains pending, check workflow `runs-on` labels against the runner labels. A healthy runner with labels like `linux,crypto-bot-python-313,ubuntu-latest` is label-compatible with existing `ubuntu-latest` jobs; if jobs still wait, investigate scheduler/run state read-only before mutating workflow, PR, checks, or merge state.

Do not solve label mismatch by editing workflow files, dispatching workflows, mutating PR/check/status state, or merging unless a separate exact approval authorizes that action.