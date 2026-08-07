---
name: autoresearch
description: "Use when running bounded autonomous research experiments."
version: 1.1.0
author: Tugrul Guner (@tugrulguner), Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [research, autonomous, experiments, git, cron]
    category: research
    related_skills: [arxiv, ml-paper-writing]
---

# Autoresearch Skill

Run a bounded branch → experiment → evaluate → merge/revert loop in a persistent background job. This skill orchestrates existing Hermes tools and does not add a model tool, promise ground-truth evaluation for subjective research, or expose model token limits.

## When to Use

Use this skill when the user wants:

- iterative ML, code, or prompt optimization against a measurable metric;
- long-running knowledge research where each addition must pass a rubric;
- recurring competitive or technical research that preserves only improvements;
- a resumable background run with pause, stop, and status controls.

Do not use it for a quick factual lookup, a task that needs continuous user steering, or subjective work without an agreed evaluation rule. Default to normal in-chat research unless the user explicitly asks for autonomous, background, iterative, or autoresearch behavior.

## Prerequisites

- Git and Python 3.11 or newer must be available to `terminal`.
- Background execution requires a persistent Hermes scheduler. Ephemeral processes that exit immediately cannot run scheduled work.
- The launch session must have access to `cronjob`, `terminal`, `read_file`, `write_file`, `patch`, and the research tools needed by the goal.
- Choose a deterministic metric for ML/code work. For knowledge work, state the rubric and evidence gates before launch.

No external Python package is required. The helper scripts use the standard library.

## How to Run

1. Infer the goal, domain, scope, evaluation mode, and sensible bounds from the request. Ask at most one question only when the evaluation target is genuinely ambiguous.
2. Load `templates/cron_prompt.md` with `skill_view` and capture the absolute `skill_dir` from the tool result.
3. Generate a research ID:

```bash
python "<skill_dir>/scripts/state.py" gen-id "<domain>"
```

4. Set the run directory to `${HERMES_HOME:-$HOME/.hermes}/autoresearch/<research_id>` and initialize it:

```bash
python "<skill_dir>/scripts/state.py" init \
  "<run_dir>" "<goal>" "<domain>" "<scope>" <max_experiments> \
  --max-duration <minutes>
```

5. Fill the cron template using literal values. Create a one-shot job with `cronjob(action="create", schedule="1m", repeat=1, skills=["autoresearch"], workdir=<run_dir>/workspace, prompt=<filled prompt>)`.
6. Optionally create a finite watchdog job from `templates/watchdog_prompt.md`. Use `schedule="every 15m"`, the same workspace, and a repeat count covering the configured duration plus one final check.
7. Register the run and returned job IDs with `scripts/registry.py`. Completion means the registry, run directory, and job identifiers agree.

Use the default delivery target so results return to the originating conversation. On a local-only CLI session, scheduled output is stored rather than pushed to a live channel; do not promise a notification unless the user selected a gateway-connected destination.

## Quick Reference

| Action | Command |
|---|---|
| Generate ID | `python scripts/state.py gen-id <domain>` |
| Initialize | `python scripts/state.py init <run_dir> <goal> <domain> <scope> <count> --max-duration <minutes>` |
| Read status | `python scripts/state.py status <run_dir>` |
| Check limits | `python scripts/state.py check-limits <run_dir>` |
| Pause | `python scripts/state.py control <run_dir> --action pause` |
| Request resume | `python scripts/state.py control <run_dir> --action resume` (then the active session creates and registers a replacement one-shot job) |
| Stop | `python scripts/state.py control <run_dir> --action stop` |
| Read checkpoint | `python scripts/state.py read-checkpoint <run_dir>` |
| Plan summary | `python scripts/plan.py summary <run_dir>` |
| Result statistics | `python scripts/evaluate.py stats <run_dir>` |
| Generate report | `python scripts/report.py generate <run_dir>` |
| Usage report | `python scripts/usage.py summary <run_dir>` |

All relative script paths in this table are relative to the loaded skill directory.

## Procedure

### 1. Define the evaluation contract

For ML or code optimization, record:

- the command that produces the metric;
- whether higher or lower is better;
- the baseline value;
- an invalid-run condition such as timeout, crash, or missing output.

For knowledge research, score evidence, accuracy, depth, relevance, and net improvement from 1–5. Merge only when total ≥13, evidence ≥3, relevance ≥3, and net improvement ≥3. Scores are an explicit heuristic, not ground truth; every accepted claim must still retain source evidence.

Completion criterion: another agent can repeat the evaluation without guessing the target or acceptance direction.

### 2. Initialize durable state

Use `state.py` rather than hand-writing JSON. It atomically creates configuration, status, control, plan, result, and workspace state under the run directory.

The run has two non-model bounds:

- `max_experiments`: a hard iteration cap, including replanned experiments;
- `max_duration_minutes`: a cooperative wall-clock stop checked before and after each experiment.

The duration check is not an operating-system kill timer. Individual tool calls and the scheduled agent remain subject to Hermes's own time and iteration safeguards. Hermes scheduler and provider safeguards remain authoritative. Do not add per-task model token controls or convert informational usage reporting into enforcement.

Completion criterion: `state.py status` and `state.py check-limits` both return valid JSON.

### 3. Initialize the workspace and baseline

Use `workspace.py init`, then create the target artifact and commit the baseline on `main`. The baseline must be evaluable before the first experiment.

Knowledge runs normally use `research.md`; code/ML runs use the files named by the goal. Do not place secrets, datasets with restricted licenses, virtual environments, or generated model weights in the workspace.

Completion criterion: `git status --short` is clean and the baseline metric or rubric score is recorded.

### 4. Build the experiment plan

Use `plan.py write` with experiments containing an ID, type, hypothesis, and target. Mix:

- `investigate` for initial evidence;
- `deepen` for weak sections or hypotheses;
- `verify` for independent corroboration;
- `synthesize` for comparisons and conclusions.

Plan fewer experiments than the hard cap so evidence-driven replanning has room. Replanning may replace weak pending work but must never raise the configured cap.

Completion criterion: every experiment has a falsifiable hypothesis or checkable deliverable.

### 5. Run one isolated experiment at a time

Before every experiment:

1. Read `control.json` with `state.py read-control`.
2. Run `state.py check-limits` and stop when `exceeded` is true.
3. Read recent results and the current baseline.
4. Create the experiment branch with `workspace.py branch`.
5. Perform only the planned change.
6. Evaluate the real diff and metric/rubric.
7. Merge improvements with `workspace.py merge`; discard regressions with `workspace.py revert`.
8. Log the decision, update status, and write a checkpoint.

A revert is a successful experimental result. Three consecutive execution failures pause the run for diagnosis rather than consuming the remaining experiment count blindly.

Completion criterion: the process is back on `main`, the workspace is clean, and the decision exists in `results.log`.

### 6. Resume and control safely

- `pause`: checkpoint and exit without marking the run complete.
- `resume`: the active session verifies phase `paused`, writes control action `resume`, fills `templates/resume_prompt.md`, creates a new one-shot job with the same `skills`, `workdir`, and delivery target, and updates the registry's `cron_job_id` before the job is due. The resumed job reconciles state, clears control to `none`, and moves both state and registry to `executing`. Never ask a cron-run session to schedule another cron job.
- `stop`: synthesize accepted work, report the stop reason, and mark the run stopped.
- `adjust`: treat the addendum as guidance, but reject any change that would exceed the original hard cap.

Completion criterion: control state, checkpoint state, and registry state describe the same run phase.

### 7. Synthesize and verify

Generate the report from accepted results and inspect the final workspace directly. Include:

- the final metric or evidence-backed findings;
- baseline-to-final comparison;
- merged, reverted, failed, and unattempted experiments;
- limitations and unresolved questions;
- informational usage when available.

Return the report in the scheduled job’s final response. Never claim cron delivery occurred unless the job output or destination verifies it.

Completion criterion: report claims trace to accepted experiments and the final branch contains no reverted changes.

## Pitfalls

1. **Treating self-evaluation as truth.** Preserve citations and use hard evidence gates; label rubric scores as heuristic.
2. **Letting cron jobs recursively schedule work.** The active session creates jobs. Scheduled runs execute the supplied task only.
3. **Using an unbounded loop.** Experiment count and duration must both be positive before launch.
4. **Merging before evaluation.** Evaluate the branch diff against the current `main`, then merge or discard.
5. **Counting longer text as better research.** Net improvement requires new verified information, correction, or synthesis.
6. **Running parallel writes against one workspace.** Parallelize evidence gathering with `delegate_task`, but serialize branch mutation and merge decisions.
7. **Losing state in a subagent sandbox.** Durable runs use scheduler sessions and host-visible run directories; delegation is only for bounded subtasks whose returned result is consumed immediately.
8. **Promising notifications from local CLI cron.** Local-only sessions save output but do not push it to a messaging channel.

## Verification

- [ ] Goal, scope, evaluation direction, and baseline are explicit.
- [ ] Experiment and duration bounds are positive and persisted.
- [ ] No user-facing model token limit is configured or enforced.
- [ ] Control is checked before every experiment.
- [ ] Each experiment starts from current `main`.
- [ ] Every merge/revert decision is logged with evidence.
- [ ] Failed experiments cannot silently modify the baseline.
- [ ] Checkpoint and registry data support a real resume.
- [ ] Final report matches git history and accepted results.
- [ ] Delivery behavior is verified for the actual session type.
