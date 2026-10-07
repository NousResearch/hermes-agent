---
name: kingmaker
description: Use when picking or ranking a cloud LLM by cost per task.
version: 0.1.0
author: Sean Alt (Salt-555), Hermes Agent
license: MIT
platforms: [linux, macos]
metadata:
  hermes:
    tags: [llm, model-selection, benchmarking, cost-control, cron]
    category: research
---

# Kingmaker Skill

Keeps the main model on the highest-scoring Hermes Index model whose measured
average $/task is strictly under a per-task budget the user chooses. Setup
installs a no_agent cron job; a tick then costs nothing.

## First-time setup

One run: show the live board, ask, install, record, create the job.

1. **Sample the live board.** `terminal(command="python3
   ${HERMES_SKILL_DIR}/references/kingmaker-job.py --sample", timeout=120)`
   prints every ranked model with its Hermes Index score and measured `$/task`,
   then budget spans (the price ranges over which one model stays best). Quote
   nothing from memory: names and prices churn.
   *Done when:* the live table and budget spans are in front of the user.
2. **Ask the budget**, with the spans visible: "What do you want to spend per
   task?" Accept a dollar figure or "no limit". If the answer excludes every
   usable model, say so and let them adjust.
   *Done when:* the number selects at least one non-provisional model with an
   exact provider id, and the user has confirmed it.
3. **Ask apply vs notify.** `apply` switches the main model when the best
   in-budget option changes; `notify` reports the change and never writes config.
   *Done when:* the user has chosen one.
4. **Ask the cadence.** Presets: daily `0 7 * * *`, twice daily `0 7,19 * * *`,
   weekly `0 7 * * 1`, monthly `0 7 1 * *`; a custom expression is fine.
   *Done when:* the user has confirmed a cron expression.
5. **Install the script.** Copy `references/kingmaker-job.py` to
   `$HERMES_HOME/scripts/kingmaker-job.py` and `chmod +x`. It installs as
   shipped; the answers are a file, not constants to edit.
   *Done when:* the file exists there, is executable, and runs with no output.
6. **Record the answers.** Run `--setup --cap <n|none> --mode <apply|notify>
   --cadence "<cron>" --deliver <route>` (see Commands); it writes
   `answers.json` beside `SKILL.md`. Re-running merges and never loses the job id.
   *Done when:* the command prints the answers it wrote.
7. **Create the job** from `references/cron-job.md` with `cronjob_manage`
   (`action: create`): `no_agent`, that script, the chosen schedule, the chosen
   delivery route (`origin`, `local`, or `platform:chat_id`).
   *Done when:* the call returns a job id and reports the job as scheduled.
8. **Record the job id** with `--setup --cron-job <job_id>`, so a later
   reconfigure updates that job instead of creating a second one.
   *Done when:* `--setup` shows that `cron_job_id`.
9. **Verify the tick.** Fire the job once with `cronjob_manage` (`action: run`).
   *Done when:* the run reports ok with empty output: a silent tick, since the
   chosen model either already matches the configured one or was applied.

## Where the answers live

`answers.json` beside `SKILL.md` is this install's only storage; the tick reads
it. The only Hermes config this skill ever writes is the main model, via
`hermes config set model.default`. Derived state (`latest.json`, `apply_state.json`,
`apply_receipt.json`, `ledger.jsonl`) lives in `$HERMES_HOME/data/kingmaker/`.

## Commands

```
python3 $HERMES_HOME/scripts/kingmaker-job.py          the bare run cron executes
python3 $HERMES_HOME/scripts/kingmaker-job.py --sample live table of models, scores, $/task
python3 $HERMES_HOME/scripts/kingmaker-job.py --setup --cap 0.30 --mode apply \
        --cadence "0 7 * * *" --deliver origin         record the answers
python3 $HERMES_HOME/scripts/kingmaker-job.py --setup  print the answers
python3 $HERMES_HOME/scripts/kingmaker-job.py --setup --cron-job <id>   record the job id
```

One-off overrides on any run: `--cap 0.25`, `--cap none`, `--apply-main`, `--notify`.

## Pitfalls

- Map a leaderboard name to a provider id on family AND version evidence, never
  string similarity; an unmapped-but-stronger in-budget row blocks the switch and
  is surfaced, never silently skipped.
- Provisional rows marked with `*` never qualify.
- The budget is measured `$/task`, strictly under, not a token price; the pick
  maximizes index score inside the cap.
- Decided outcomes (page changed, no eligible model, refused smoke or apply,
  concurrent manual change) print a notice and exit 0, so the job delivers a
  message rather than reporting a broken job. Only exit 1 is a failure.
- A run that exits 2 means the leaderboard page shape changed: re-derive the
  row-shape constants (`ROW_SHAPES`) in the script rather than loosening the
  sanity gate.
