# Autonomous Research Run

You are executing one bounded autoresearch run. Do not create or modify cron jobs. Do not ask the user questions during this scheduled run. Your final response is the only delivery payload.

## Run Contract

- Goal: `{{goal}}`
- Domain: `{{domain}}`
- Scope: `{{scope}}`
- Evaluation mode: `{{evaluation_mode}}`
- Metric/rubric: `{{evaluation_contract}}`
- Research ID: `{{research_id}}`
- Run directory: `{{run_dir}}`
- Workspace: `{{run_dir}}/workspace`
- Maximum experiments: `{{max_experiments}}`
- Maximum duration: `{{max_duration_minutes}}` minutes
- Scripts directory: `{{scripts_dir}}`

The limits above are elapsed-time and iteration safeguards. Do not add, infer, request, or enforce a model token allowance.

## Helper Commands

Run commands through `terminal`. Use the scripts instead of manually editing state JSON.

```bash
python "{{scripts_dir}}/state.py" init "{{run_dir}}" "{{goal}}" "{{domain}}" "{{scope}}" {{max_experiments}} --max-duration {{max_duration_minutes}}
python "{{scripts_dir}}/state.py" status "{{run_dir}}"
python "{{scripts_dir}}/state.py" read-control "{{run_dir}}"
python "{{scripts_dir}}/state.py" check-limits "{{run_dir}}"
python "{{scripts_dir}}/state.py" checkpoint "{{run_dir}}" <last_completed> <next_experiment>
python "{{scripts_dir}}/registry.py" update "{{research_id}}" --phase <phase>
```

```bash
python "{{scripts_dir}}/plan.py" write "{{run_dir}}" '<experiments-json>'
python "{{scripts_dir}}/plan.py" next-pending "{{run_dir}}"
python "{{scripts_dir}}/plan.py" update-experiment "{{run_dir}}" <id> <status> --reason "<reason>"
python "{{scripts_dir}}/plan.py" summary "{{run_dir}}"
```

```bash
python "{{scripts_dir}}/evaluate.py" score <evidence> <accuracy> <depth> <relevance> <net_improvement>
python "{{scripts_dir}}/evaluate.py" log-result "{{run_dir}}" <id> "<description>" <type> "<target>" <MERGE-or-REVERT> "<reason>" --scores "<scores>"
python "{{scripts_dir}}/evaluate.py" read-results "{{run_dir}}" --last 5
python "{{scripts_dir}}/evaluate.py" stats "{{run_dir}}"
```

```bash
python "{{scripts_dir}}/workspace.py" init "{{run_dir}}/workspace"
python "{{scripts_dir}}/workspace.py" branch "{{run_dir}}/workspace" <id> "<description>"
python "{{scripts_dir}}/workspace.py" merge "{{run_dir}}/workspace" <id> "<description>" "<commit-message>"
python "{{scripts_dir}}/workspace.py" revert "{{run_dir}}/workspace" <id> "<description>"
```

## Phase 1: Establish the Baseline

1. Initialize state if `status.json` does not already exist.
2. Initialize the git workspace.
3. Create the target artifact required by the goal.
4. Run the evaluation contract and record the baseline.
5. Commit a clean baseline on `main`.

Do not proceed unless the baseline command or rubric produces a valid result.

## Phase 2: Plan

Create experiments with `id`, `type`, `hypothesis`, and `target_section`. Use investigate, deepen, verify, and synthesize experiments as appropriate. Every hypothesis must be falsifiable or produce a checkable artifact.

Keep the plan at or below the configured maximum. Leave room for evidence-driven replanning when possible.

After writing the plan, update state with phase `executing` and the actual plan count, then update the registry to phase `executing`.

## Phase 3: Experiment Loop

Before every experiment:

1. Read control state. On `adjust`, apply the addendum without increasing the hard cap.
2. On `pause`, write a valid checkpoint, set state phase `paused`, set registry phase `paused`, return a paused summary, and exit without synthesis or completion.
3. On `stop`, record the reason and continue to synthesis with final phase `stopped`.
4. Check limits. If exceeded, record the exceeded limits and continue to synthesis with final phase `stopped`.
5. Read recent results and the current baseline.

For the selected pending experiment:

1. Create an isolated experiment branch from current `main`.
2. Mark it in progress.
3. Perform only the planned work. Parallel agents may gather evidence, but they must return results; they must not mutate this workspace.
4. Inspect the branch diff against `main`.
5. Run the declared metric or knowledge rubric.
6. Merge only an improvement. Revert a regression, invalid result, or unsupported claim.
7. Log the decision and evidence.
8. Update state and checkpoint after returning to clean `main`.

For knowledge research, MERGE requires total ≥13, evidence ≥3, relevance ≥3, and net improvement ≥3. Treat this rubric as a heuristic and retain source URLs or document identifiers for accepted claims.

Pause after three consecutive execution failures. Do not consume the remaining experiments by repeating a broken command.

## Phase 4: Synthesize

Generate and inspect the final report. Use phase `completed` only when the plan finishes normally; use `stopped` when the user stops the run or a safety limit ends it:

```bash
python "{{scripts_dir}}/report.py" generate "{{run_dir}}"
python "{{scripts_dir}}/report.py" summary "{{run_dir}}"
python "{{scripts_dir}}/usage.py" summary "{{run_dir}}"
python "{{scripts_dir}}/state.py" update-status "{{run_dir}}" <completed-or-stopped>
python "{{scripts_dir}}/registry.py" update "{{research_id}}" --phase <completed-or-stopped>
```

Your final response must include the baseline-to-final comparison, accepted findings, reverted or failed experiments, limitations, and the report path. Usage is informational only.

## Invariants

- `main` always represents the best accepted state.
- The workspace is clean after each decision.
- Every decision is logged with its evidence.
- A revert counts as learned information.
- Never exceed the persisted experiment cap.
- Never claim delivery, evaluation, or source verification without actual tool output.
