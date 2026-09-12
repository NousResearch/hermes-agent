# Explanation policy — first run (2026-08-25): null result, ceiling effect

One concept-comparison task (`value_vs_reference`), three policies, `n=1` each,
reader and judge on the same model as the explainer.

**Model:** `Qwen3.8-27B` (whatever the configured auxiliary route served — not
`EVAL_MODEL`; see caveats). **Signals:** `intent=learn`, `structure=comparison`,
`knowledge=unknown`.

## Results

| policy | modality | comprehension | transfer | calib err | chars | sec |
|---|---|---|---|---|---|---|
| fixed_markdown | concise | 100% | 100% | 0.007 | 530 | 183.9 |
| smallest_useful | comparison_table | 100% | 100% | 0.000 | 770 | 199.4 |
| prediction_first | prediction_first | 100% | 100% | 0.007 | 670 | 95.6 |

Against the `fixed_markdown` control:

- `smallest_useful`: transfer **+0pp**, comprehension **+0pp**, **1.45x** length
- `prediction_first`: transfer **+0pp**, comprehension **+0pp**, **1.26x** length

## Reading

**The instrument has no resolution on this task.** Every arm saturates. The
adaptive policies cost 26–45% more output and buy nothing measurable, which
under the promotion gate in the README is a clear "not promoted" — a policy that
inflates length without moving a primary outcome has not won.

This is a ceiling effect, not a lenient judge. The reader's answers were
substantive and independently correct in all three arms; the transfer item was
answered with the right mechanism (the identifier may no longer resolve after
six months, so the replay cannot reproduce what the user saw), not by restating
the explanation. The judge scored 3/3 items in each arm on answers that deserved
it.

The modality renderers do work and do differ: `concise` produced plain prose,
`comparison_table` produced an actual Markdown table on discriminating axes, and
`prediction_first` produced a prediction question, a `---` separator, and
corrective feedback aimed at the likely wrong prediction. The policies are
behaving as specified. The *task* is the problem.

## What this says about the evaluation contract

A four-discriminator concept comparison is too easy for a proxy reader. If the
control already conveys everything needed for both comprehension and transfer,
no modality can beat it, and the comparison is uninformative regardless of how
many repeats you run.

For #93382's evaluation contract this is a real constraint, and it lands before
any transport work: task families have to be hard enough that the concise
control genuinely loses information, or the whole comparison measures nothing.
Concretely, candidates for the next task generation are more discriminators than
a short answer can carry, transfer items requiring composition of two
discriminators rather than one, and distractor concepts that a surface reading
would conflate.

## Caveats — do not over-read this

- **`n=1` per policy.** This is a smoke number. It is enough to establish that
  all three arms sit at the ceiling, not enough to size any effect.
- **The reader is a proxy, not a human subject.** A ceiling for a model reader
  does not imply a ceiling for a person; the failure mode may be specific to a
  reader that already knows the material from pretraining. This is the sharpest
  limitation of the harness and it is why the README refuses to call this a
  result about human learning.
- **Not `EVAL_MODEL`.** The published reference model is
  `anthropic/claude-opus-5`; this run served `Qwen3.8-27B`, because that is what
  the configured auxiliary route returned. Numbers from different models are not
  comparable, which is why the model is recorded on every scorecard row.
- **Latency here is route noise**, not a property of the policies. The same
  route timed out repeatedly at 180s during earlier attempts.

## Reproduce

```bash
python evals/explanation_policy/runner.py --repeats 1 --timeout 900 \
    --out evals/explanation_policy/results/run1
python evals/explanation_policy/report.py evals/explanation_policy/results/run1
```
