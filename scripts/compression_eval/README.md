# External compression evaluation

Supply a reviewed JSON battery definition independently of the generated report.
It contains the complete ordered list of expected probe names, for example
`["accuracy", "continuity"]`. Keep this definition under version control.

```sh
python scripts/compression_eval/run_context_compression_eval.py \
  --hermes-root /path/to/clean/hermes \
  --harness /path/to/evaluator \
  --battery-definition /path/to/reviewed-battery.json \
  --output /path/outside/source-and-harness/report.json \
  -- python evaluate.py
```

The harness must include `evaluator_digest` and `battery_digest` from the runner's
`HERMES_EVALUATOR_DIGEST` and `HERMES_BATTERY_DIGEST` environment variables in its
report. Probe names and scores must match the independent definition. The evaluator
digest binds harness files, the command, and interpreter; generated `results/`, Git
metadata, virtual environments, and Python caches are excluded. Source cleanliness
and evaluator content are checked again after execution before publication.
