# Resume Autoresearch Run — {{research_id}}

Resume one previously paused run. Do not create or modify cron jobs, and do not ask the user questions. The final response is the only delivery payload.

## Run

- Goal: `{{goal}}`
- Domain: `{{domain}}`
- Research ID: `{{research_id}}`
- Run directory: `{{run_dir}}`
- Workspace: `{{run_dir}}/workspace`
- Scripts directory: `{{scripts_dir}}`

## Procedure

1. Read and reconcile persisted state:

```bash
python "{{scripts_dir}}/state.py" status "{{run_dir}}"
python "{{scripts_dir}}/state.py" read-checkpoint "{{run_dir}}"
python "{{scripts_dir}}/state.py" check-limits "{{run_dir}}"
python "{{scripts_dir}}/plan.py" read "{{run_dir}}"
python "{{scripts_dir}}/plan.py" summary "{{run_dir}}"
python "{{scripts_dir}}/evaluate.py" read-results "{{run_dir}}"
python "{{scripts_dir}}/evaluate.py" stats "{{run_dir}}"
```

2. Inspect the git workspace:

```bash
python "{{scripts_dir}}/workspace.py" current-branch "{{run_dir}}/workspace"
python "{{scripts_dir}}/workspace.py" log "{{run_dir}}/workspace" --oneline
```

Resume only from a consistent state. If an interrupted experiment branch is active, compare it with persisted plan and checkpoint data, evaluate it, then explicitly merge or revert before selecting another experiment.

3. Stop and synthesize immediately if limits are already exceeded.
4. Reset control only after state reconciliation:

```bash
python "{{scripts_dir}}/state.py" control "{{run_dir}}" --action none
python "{{scripts_dir}}/state.py" update-status "{{run_dir}}" executing
python "{{scripts_dir}}/registry.py" update "{{research_id}}" --phase executing
```

5. Continue at `checkpoint.next`, skipping experiments already marked merged, reverted, or failed.
6. Follow the same control, limit, branch, evaluation, logging, and checkpoint invariants as a fresh run.
7. Pause after three consecutive execution failures. Generate the final report when the plan finishes, the user stops the run, or a limit is reached.

The active session must have replaced the persisted main cron job ID with this resume job's ID before this job starts. Never treat usage reporting as an execution limit. Never claim resume succeeded unless state, registry, and workspace checks all pass.
