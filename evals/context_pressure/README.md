# Context-pressure evaluation

This is a small, deterministic harness for testing whether a Hermes runtime or
provider configuration handles long context better. It is deliberately
independent of the experimental ARC reasoning backend: each arm is just a
user-supplied Hermes command, so the same fixture can compare normal Hermes,
an experimental branch, a model/provider setting, or a future runtime mode.

## What it measures

`runner.py` creates a fresh workspace and `HERMES_HOME` for every cell, runs a
bounded subprocess, and validates the resulting artifact. Each result records
the exit status, timeout classification, wall time, validator checks, stdout /
stderr tails, and the JSON emitted by Hermes `-z --usage-file`, including token,
cache, API-call, reasoning, and cost fields when the provider reports them.

The distributed-evidence task requires all 50 generated fragments to be
represented by an inspection ledger containing exact SHA-256 hashes and
statuses. It also checks the early, middle, and late anchors, the combined
synthesis, and explicit rejection of every known unverified distractor. A
model's claim that it inspected every file is not accepted as evidence.

## Running it

Use a configured Hermes environment and keep credentials in the environment or
in the normal local secrets store; the runner does not copy `.env` files into
the isolated homes.

```bash
python evals/context_pressure/runner.py \
  --model <model> --provider <provider> \
  --repetitions 3 --timeout 900 \
  --out /tmp/hermes-context-pressure
```

To compare arbitrary commands, repeat `--arm`. The placeholders are
`{python}`, `{prompt}`, `{workspace}`, `{hermes_home}`, `{usage_file}`,
`{model}`, `{provider}`, and `{model_flags}`. The default arm is equivalent to:

```text
{python} -m hermes_cli.main -z {prompt} --yolo --usage-file {usage_file} {model_flags}
```

For example, two configurations can be compared with the same generated task:

```bash
python evals/context_pressure/runner.py \
  --arm legacy='{python} -m hermes_cli.main -z {prompt} --yolo --usage-file {usage_file}' \
  --arm candidate='{python} -m hermes_cli.main -z {prompt} --yolo --usage-file {usage_file}' \
  --repetitions 3
```

The command template is intentionally explicit: branch- or mode-specific
options belong in the command supplied by the experiment, not in this fixture.
Use `--timeout` as a hard per-cell ceiling; partial usage JSON and failure
diagnostics are retained when a run times out. Results are written to `/tmp`
by default and should not be committed as benchmark dumps.

## Interpreting results

The harness is designed to produce negative results. Lower uncached input is
not the same as lower total cost, lower latency, or higher validated success.
Compare validated artifacts first, then calls, cache behaviour, tokens, cost,
and tail behaviour across repeated paired runs. It does not assert anything
about ARC-AGI-3, GPT-6 Astra, `previous_response_id`, provider-side compaction,
or any other provider-native feature; those claims require a direct interface
that exposes and records the relevant mechanism.
