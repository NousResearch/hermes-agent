# Explanation Policy Eval Harness

Measures whether choosing *how* to explain something changes what a reader
can actually do with it — comprehension, transfer, and confidence
calibration — against a fixed-Markdown control.

This is the policy/evaluation layer discussed in #93382, cut down to the part
that needs no response envelope. Everything renders as ordinary Markdown, so
the harness has no dependency on #7191, #61095, or #74334 and adds no core
surface: no model tool, no CLI command, no config key, no system-prompt
change.

## What it does

1. Builds `Signals` for a task — intent, structure, declared prior knowledge,
   risk. These are the bounded, observable inputs from #93382; nothing is
   inferred into a durable profile, and nothing persists between runs.
2. Applies each policy in the matrix to those signals to pick a **modality**
   (`policies.py`). `fixed_markdown` is the control arm: always one concise
   answer, no selection.
3. Asks the model to explain the concept under that modality's render
   instruction. Markdown out, always.
4. Hands the explanation — and nothing else — to a fresh reader model, which
   answers the comprehension items, the transfer item, and states a
   confidence for each.
5. Judges those answers against gold in a separate call that sees gold. The
   reader never does.
6. Emits a scorecard: comprehension, transfer, calibration error, explanation
   length, and wall time per policy.

## Usage

By default every arm — explainer, reader and judge — goes through your
configured auxiliary route, the way the other `evals/` harnesses call
`call_llm`. Pass `--model` to force one; `EVAL_MODEL` in `policies.py`
(`anthropic/claude-opus-5`) is the **reference** model that published
scorecards should use, not a pin the runner enforces.

Either way the model is held constant across policies within a run, which is
what makes a scorecard difference attributable to the policy — and whatever
answered is recorded per row, so a scorecard always names its own model.

```bash
# from repo root, venv active, provider configured
python evals/explanation_policy/runner.py \
    --task value_vs_reference \
    --policies fixed_markdown,smallest_useful,prediction_first \
    --intent learn --structure comparison \
    --repeats 3 \
    --out evals/explanation_policy/results/run1
python evals/explanation_policy/report.py evals/explanation_policy/results/run1
```

Before spending a real run, smoke the whole chain against your provider for a
few hundred tokens:

```bash
python evals/explanation_policy/runner.py \
    --task synthetic --policies fixed_markdown --repeats 1 \
    --out evals/explanation_policy/results/smoke
python evals/explanation_policy/report.py evals/explanation_policy/results/smoke
```

The synthetic task's gold is mechanically checkable ("ALPHA is red"), so a
correct pipeline scores 100% on it. Anything less means the plumbing is
broken, not the policy — and you find that out in one cheap call instead of
twenty-seven expensive ones.

`--knowledge practitioner` exercises the expertise-reversal guard: a reader
who declares prior knowledge gets a retrieval check instead of a worked
example. Each run costs three LLM calls per policy per repeat, so start with
`--repeats 1` while you are changing the matrix.

`--timeout` defaults to 180s. These task names are in nobody's `auxiliary.*`
config, so without it the 30s aux default applies and a slow route makes a
long modality look like a provider failure.

`--repeats` averages within a run. A real comparison needs enough repeats to
separate a policy effect from model variance; three is a smoke number, not an
evidentiary one.

## Scope — what this is and is not

**The reader is a proxy, not a human subject.** This harness measures what
survives an explanation. That is a prerequisite for the human study #93382
describes, not a substitute for it: a win here does not license a claim about
human learning, and the issue's own caveat about dialogue benchmarks applies
to this harness too.

**It is measurement, not an extension point.** There is no hook, callback, or
interface here for anything else to depend on — it is a runner you invoke by
hand, in `evals/`, like `compaction` and `readtool`. Nothing ships to users.

**The stated consumer** is the Flow State vertical described in #93382's
thread; the policy exists to be measured before any of it is built, not
after.

## Promotion gate

Per #93382's own acceptance criteria, a policy earns its way out of `evals/`
only by improving at least one declared primary outcome with no material
regression in the others. Concretely, for anything in this matrix:

- transfer or comprehension up against `fixed_markdown`, and
- calibration error not worse, and
- explanation length and latency cost reported, not hidden.

A policy that wins on comprehension while inflating length 3x and worsening
calibration has not won.

Individual runs under `results/` are gitignored. Publishing a finding means
committing a dated `results/SCORECARD-<date>.md`, the way `evals/compaction`
does — and a null or negative result is as publishable as a positive one.

## Files

| file | what it holds |
| --- | --- |
| `policies.py` | intent/structure/modality enums, the policy matrix, render instructions |
| `fixtures.py` | the concept-comparison task, its gold discriminators and transfer item; `synthetic_task()` for smoke tests |
| `runner.py` | explain → read → judge pipeline, writes `runs.json` + `scorecard.json` |
| `report.py` | scorecard table, deltas against the control, `scorecard.md` |
