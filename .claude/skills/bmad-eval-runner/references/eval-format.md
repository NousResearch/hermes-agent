# Eval format and the four modes

A case is the unit of evaluation. Every case is `input + rubric + optional state_prefix`. The same case shape feeds all four modes; what changes is which invocations the runner sets up and how the result is judged.

## The case

```json
{
  "id": "create-1",
  "input": "I want a brief for InsuLens, a claims-triage tool for mid-market insurers. Notes are in evals/insulens/files/memo.md",
  "rubric": [
    "brief.md exists and its word count is between 250 and 1500",
    "brief.md names InsuLens and the mid-market insurer segment",
    "brief.md incorporates at least two specific points from memo.md without inventing claims absent from it"
  ],
  "state_prefix": null,
  "files": ["evals/insulens/files/memo.md"]
}
```

Field semantics:

- `id`: stable identifier; used as the case's folder name in the run.
- `input`: the realistic, messy user request. Use real file paths, company names, typos, and casual speech, because a polished input tests a situation the skill rarely meets. The runner sends this verbatim to the invocation, after prepending any `state_prefix`.
- `rubric`: a list of named expectations, each gradeable to `{text, passed, evidence}` by the grader. The strong-versus-weak taxonomy below decides whether each one is worth keeping.
- `state_prefix`: optional bracketed prime that places the skill mid-workflow (see below). Null or absent means the skill starts cold.
- `files`: optional fixture paths staged into the case's clean working directory before the run. A bare filename lands at the workspace root; a nested path keeps its directory structure, so the input can reference it verbatim. Sources resolve against `--project-root`, then the cases file's directory, then as absolute paths.

For trigger cases the shape is lighter: a `query` and a `should_trigger` boolean, because there is no artifact to grade, only whether the skill fired. Those cases are covered in `platform-adapter.md` and `description-optimization.md`.

## state_prefix: turn simulation in one shot

Most multi-turn skills can be evaluated single-shot if the case is designed right. The `state_prefix` is the trick that makes mid-workflow points reachable without a multi-turn simulator. It is a bracketed prime prepended to the input that tells the skill where in its own flow this turn lands and what the user already said:

```
[the skill has already worked through discovery; on turn 4 the user was asked about stakeholders and responded:] User said: "just me and a PM"
```

The runner prepends the `state_prefix` to `input` and sends the combined text as a single message. One input then exercises any mid-workflow moment: a clarifying turn, a correction, a resume after an interruption. This replaces the deferred multi-turn simulator for everything except cases where the conversation arc itself is the deliverable.

Subjective skills (coaching, brainstorming, design facilitation) skip the rubric and rely on human judgment. The `state_prefix` still earns its place there, because it lets a human see the exact mid-run moment they want to judge.

## Strong versus weak expectations

The grader's job is easier and the result is more honest when an expectation is discriminating, meaning a wrong output cannot pass it. A weak expectation is worse than no expectation, because a green check on it reads as proof when it is noise. The grader flags weak expectations when it sees them; write them out of the rubric before they ship.

Weak patterns to avoid:

- Filename-only checks. "brief.md exists" passes for an empty file. Pair existence with a content check.
- Wholly subjective phrasing. "the brief is high quality" cannot be graded. State the property concretely.
- Tautologies. Anything that follows automatically from the prompt being understood proves nothing.

Strong patterns for artifact correctness:

- Specific facts that must appear, such as "incorporates at least two findings from section X."
- Structural claims a wrong output would fail, such as "word count between 250 and 1500."
- Negative assertions, such as "does not introduce content from unrelated sections."
- Frontmatter checks, such as "frontmatter contains title, status, created (ISO 8601), updated."
- Bounded output blocks, such as "the final message contains a JSON object with intent='create'."

Strong patterns for process discipline:

- Side-artifact existence paired with content, such as ".memlog.md captures the pricing decision with its rejected alternative and rationale."
- Transcript tool-call patterns, such as "the transcript contains a call invoking bmad-editorial-review-prose."
- Phase ordering, such as "the polish call occurs after the brief Write and before the final JSON block."
- Read-only enforcement, such as "the input brief.md is byte-identical to the fixture and no Write or Edit targeted it."
- Bidirectional fidelity, such as "every decision in the memlog is reflected in the brief, and no claim in the brief is absent from the input or the memlog."

Most process-discipline checks are deterministic reads of the transcript and filesystem, so the grader confirms them by quoting evidence rather than judging.

## The four modes in detail

### Baseline: skill versus bare model

Run the case input twice in parallel in the same turn, once wrapped by the skill and once against the bare model with nothing around it. The bare-model run is the long-term floor. The skill earns its existence only by producing something the bare model cannot, so when the skill stops beating the bare model the right call is retirement, not another patch. Use baseline when the user asks whether the skill is worth keeping, or as the release check.

### Variant: full versus stripped smallest-version

Run the full skill against a stripped smallest-version of the same skill (passed as `--variant-path`), or against a snapshot of the prior version for an edit, on the same input. This is the two-version comparison made runnable, and it settles the leanness scanner's defend-against-absence findings. If the two outputs tie on the dimension the section was supposed to protect, the section is decoration and gets cut. If the small version is materially and durably worse, the section earned its keep. Variant is how a suspected piece of ceremony gets a real verdict instead of an argument.

### Quality: output versus rubric

Grade a single config's output against the named rubric with the read-only grader in `references/grader.md`. The grader gives no partial credit, puts the burden of proof on a passing grade, and flags any non-discriminating assertion. Use quality when a rubric exists and the user wants to know whether the output meets it, independent of any comparison.

### Trigger and description

Generate near-miss should-trigger and should-not-trigger queries that share keywords, split them, measure real firing through the adapter, and improve the description across bounded rounds with the held-out scores blinded from the improver. The full loop, including the split ratio, the round bound, and feeding prior failed attempts back, is in `references/description-optimization.md`. Trigger detection itself is "did the skill load," abstracted per runtime in `references/platform-adapter.md`.

## Getting a skill to behave non-interactively

Single-shot modes need the skill to produce its deliverable without stopping to ask. Most multi-turn skills expose a headless flag or keyword that suppresses clarifying questions and ends with a structured status block. Trigger it from the input: the literal `Run headless.` at the start, a skill-specific keyword from the skill's own headless section, or enough context that no clarification is genuinely needed. The `state_prefix` also helps here, because a turn that already supplies the answer the skill would ask for keeps the run moving. If a skill has no headless path and the input cannot satisfy its questions, either add a headless mode to the skill or accept that this case needs a human in the loop.
