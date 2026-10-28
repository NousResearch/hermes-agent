---
name: kingmaker
description: Use when picking or ranking a cloud LLM by cost per task.
version: 0.2.0
author: Sean Alt (Salt-555), Hermes Agent
license: MIT
platforms: [linux, macos]
metadata:
  hermes:
    tags: [llm, model-selection, benchmarking, cost-control, cron]
    category: research
---

# Kingmaker

Keeps your main model on the best-value model in the Hermes Index: the smartest
you can get without passing a per-task budget you set, re-checked on a schedule.
A tick runs a script only, so checking costs nothing.

The tuning knob is one number: what a point of intelligence is worth to you in
dollars. A cheaper model wins whenever its dollars-per-point is under that number.

## When to Use

- Setting up, once, a job that keeps the main model on the best cost/quality
  point of the Hermes Index; it maintains itself afterwards.
- Reconciling it: re-ask the questions below and re-record the answers.
- Don't use for: picking a model for one task, or anything that needs our own
  benchmarks. Selection is ex-ante by design: leaderboard numbers only.

## Setup

Walk the user through these in plain language, with the board in front of both of
you. Show it first (the loader fills in the path):

    terminal(command="python3 ${HERMES_SKILL_DIR}/references/kingmaker-job.py --sample", timeout=120)

It prints the ranked board, the Pareto frontier with each model's `$/point`, and
today's pick. Never quote names or prices from memory; they churn.

Ask, and say what each answer does for them:

1. **Budget.** "What's the most you'll spend per task? I never go above it, so a
   pricey model can't run up your bill." Dollars, or "no limit."
   *Done when:* the cap admits at least one non-provisional model with an exact
   provider id, and the user confirms the figure.
2. **The point price.** "How much is one point of intelligence worth to you, in
   dollars? This is the whole tuning knob. Set it low and I favor cheap models;
   set it high and I favor smart ones; say 'none' and I always take the smartest
   I can afford. It's what lets me catch a model that's nearly as smart for half
   the price." Suggest a starting number from the board's `$/point` column, show
   what it would pick and what that saves and gives up, and let them tune it.
   *Done when:* the user gives a $/point figure or "none".
3. **Act or report.** "When a better deal appears, should I switch your model
   myself, or just tell you?" `apply` switches; `notify` never writes config.
   *Done when:* the user picks one.
4. **How often.** "How often should I check? Daily, twice daily, weekly, monthly,
   or a custom schedule." Presets: `0 7 * * *`, `0 7,19 * * *`, `0 7 * * 1`,
   `0 7 1 * *`.
   *Done when:* the user confirms a cron expression.
5. **Where notices go.** "Where should updates land: this chat, saved locally, or
   a specific channel?"
   *Done when:* a delivery route is chosen (`origin`, `local`, or `platform:chat_id`).

Then install and arm it:

6. Copy `references/kingmaker-job.py` to `$HERMES_HOME/scripts/kingmaker-job.py`;
   `chmod +x`.
   *Done when:* the file exists, is executable, and runs with no output.
7. Record the answers: `--setup --cap <n|none> --usd-per-point <n|none>
   --mode <apply|notify> --cadence "<cron>" --deliver <route>`. Writes
   `answers.json` beside `SKILL.md`; re-running merges and keeps the job id.
   *Done when:* the command prints the answers it wrote.
8. Create the job from `references/cron-job.md` with `cronjob_manage`
   (`action: create`: `no_agent`, this script, the schedule and delivery above).
   *Done when:* the call returns a job id and reports it scheduled.
9. Record the id: `--setup --cron-job <job_id>`, so a reconfigure updates that
   job rather than creating a second one.
   *Done when:* `--setup` shows that `cron_job_id`.
10. Fire once with `cronjob_manage` (`action: run`).
    *Done when:* the run reports ok with empty output: the pick already matches the
    configured model or was applied.

## How it picks

Utility is `index - ($/task)/(point price)`; the pick is the highest-utility model
that is under the budget cap, non-provisional, and mapped to a real provider id
(unmapped names and provisional `*` rows never qualify). The cap is a hard ceiling
on measured `$/task`; the point price is the only preference, and `none` makes a
point free so the rule degrades to plain max index. At any price the pick lands on
the board's Pareto frontier. Math and traps: `references/selection-policy.md`.

## Quick Reference

    python3 $HERMES_HOME/scripts/kingmaker-job.py          the scheduled run
    ... --sample                                           board, frontier, pick
    ... --setup --cap 2 --usd-per-point 0.02 --mode apply \
        --cadence "0 7 * * *" --deliver origin             record answers
    ... --setup                                            print answers
    ... --setup --cron-job <id>                            record the job id

One-off overrides: `--cap <n|none>`, `--usd-per-point <n|none>`, `--apply-main`,
`--notify`.

## Storage

`answers.json` beside `SKILL.md` holds `cap_per_task_usd`, `usd_per_point`
(`null` = no trade), `mode`, `cadence`, `deliver`, and `cron_job_id`. The only
Hermes setting written is the main model, via `hermes config set model.default`.
Run state lives in `$HERMES_HOME/data/kingmaker/`.

## Pitfalls

- Map a leaderboard name to a provider id on family AND version evidence, never
  string similarity; a stronger in-budget row we can't map blocks the switch and
  is surfaced, never silently skipped. Provisional rows never qualify.
- The index is a frozen point estimate (no CI; identical across daily snapshots),
  so a capability tolerance is a user preference, not a measured sigma. Choose the
  point price with the board in view; two neighbours can straddle it.
- Decided outcomes (page changed, no eligible model, refused smoke or apply,
  concurrent manual change) print a notice and exit 0, so the job delivers a
  message rather than looking broken. Only exit 1 is a real failure.
- Exit 2 means the leaderboard page shape changed: re-derive `ROW_SHAPES` in the
  script rather than loosening the sanity gate.

## Verification

`--setup` prints the recorded answers, and `cronjob_manage` (`action: list`) shows
it scheduled. Fire `cronjob_manage` (`action: run`): ok with empty output is a
quiet tick, meaning the pick already matches the configured model or was applied.
