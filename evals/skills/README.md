# Skill write-side evaluation (Arm A)

This directory contains the no-model, no-API-spend first arm proposed in
[issue #96704](https://github.com/NousResearch/hermes-agent/issues/96704).
It measures whether the self-improvement loop is accumulating useful skills or
dead weight from an existing Hermes profile.

## Run

```console
python evals/skills/runner.py --home ~/.hermes --window-days 30 \
  --as-of 2026-09-25T00:00:00+00:00 \
  --output evals/skills/results/arm-a.json
```

`--as-of` must be a valid ISO date/time; invalid values fail before a report is
written. Without `--home`, an installed Hermes runtime resolves the active profile.

The runner reads only:

- `~/.hermes/skills/.usage.json` for provenance and use counters;
- `~/.hermes/skills/**/SKILL.md` for the agent-created skill set and declared
  category;
- `~/.hermes/skills/.curator_ledger.jsonl` for creation events and their
  `evidence.session_id` values.

It does not import the agent loop, call a model, mutate telemetry, or index its
own output. This keeps the measurement independent of the system it measures.

## Metrics

- **Creation rate per session**: create ledger events divided by distinct
  session IDs in the inclusive report window `[as_of - window_days, as_of]`.
  Missing or invalid event timestamps are excluded with a warning. The rate is
  `null` when the ledger is unavailable or any counted creation entry lacks a
  session ID; a fabricated denominator is worse than a missing
  number.
- **Trigger precision**: the fraction of agent-created skills old enough to
  have completed the requested window that have a positive use count and a latest
  recorded use in that report window, including both boundaries.
  The current usage sidecar stores only `last_used_at`, not a use-event history,
  so the runner reports a hit only when the latest observed use is inside the
  window. The report includes this limitation as a warning.
- **Duplicate-class rate**: the fraction of agent-created skills in a class
  containing at least two skills. The class is the authored `category`, or the
  first on-disk category directory. This is a deterministic structural proxy,
  not an LLM claim that two skills are semantically identical.

Every skill row is included in the JSON report so a maintainer can audit a
summary back to the source files. The report is intentionally data-only: no
archive/delete/promotion action is taken from a score.

Run the evaluator coverage with `scripts/run_tests.sh tests/evals/test_skill_write_metrics.py`.
