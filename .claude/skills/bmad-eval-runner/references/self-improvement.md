# Self-improvement: the bounded auto-iterate loop

This is the loop that scans a skill, evaluates it, proposes a fix, applies it, and re-evaluates, repeating until the skill passes or a round bound is hit. It turns a single scan-and-fix pass into a closed loop that keeps going until the evidence says stop. It is the most autonomous mode the runner offers, so it carries the most guardrails: it is opt-in, calibrated to what is at stake, fully logged, and bounded.

The benchmark is a guardrail, never the judge. The human stays the judge. A green run means the change cleared the bar the loop was given, not that the change is correct, and the loop's job is to do the mechanical iteration a human would otherwise do by hand and then hand back a fix plus the evidence for it.

## When to run it, and how hard

The loop is opt-in. It never starts on its own, because applying changes to a skill in a loop is a stronger action than reporting findings, and the user decides when that is warranted.

Calibrate the aggressiveness to the stakes. A throwaway skill the user is still shaping can take a longer loop and a looser bar, because a wrong iteration costs little and is easy to throw away. A skill that other skills already depend on, or one that is shipped and in use, takes a short loop, a strict pass bar, and a close human read of every applied change, because a regression there propagates. Agree the round bound and the pass condition with the user before the first round, and write both into the memlog so the run is auditable against the terms it was given.

## The loop

Each round runs four beats:

1. Scan. Run the builder's scanners against the skill (the five lenses in `bmad-workflow-builder`: architecture, determinism, customization, enhancement, leanness), and collect the findings. On rounds after the first, scan again rather than trusting the prior scan, because the last fix may have moved something.

2. Eval. Run the modes that apply to this skill: quality against its rubric where one exists, variant to settle a leanness defend-against-absence finding, baseline to confirm the skill still beats the bare model. The scan says what looks wrong; the eval says whether it measurably is. A finding the eval cannot confirm is a candidate to note for a human, not to auto-fix.

3. Propose a fix. From the confirmed findings, propose one concrete change. Address the cause the finding names rather than the single case that exposed it (see generalizing, below). Keep the change small enough that the next eval can attribute the delta to it; a round that rewrites five things at once cannot tell you which one moved the score.

4. Apply and re-eval. Apply the proposed change, then re-run the eval from beat 2 and compare. A round that improves the score and breaks nothing else is kept; a round that regresses any mode is reverted before the next round, because an applied change that made things worse is not a base to build on.

Stop when the pass condition is met or the round bound is reached, whichever comes first. The bound is a hard stop: hitting it without passing ends the loop and reports the best state reached, it does not earn extra rounds.

## The full trail goes in memlog

Every round writes to the run's memlog through `scripts/memlog.py`, so the whole reasoning chain is on disk and nothing the loop decided is hidden in a model's head. Per round, log:

- a `decision` entry naming the fix proposed and the finding it answers,
- an `event` entry recording the re-eval delta (which modes ran, the before-and-after score, what regressed if anything),
- a `note` entry when a round is reverted, with why.

At the end, log a `direction` entry summarizing the final state, whether the pass condition was met, and what a human should still review. Because the trail is append-only and typed, a reviewer reads the run back in order and sees what was tried, what each attempt did to the numbers, and why the loop stopped where it did.

## Generalize to intent, do not overfit to the case

The failure that ends most auto-iterate loops is fixing the example instead of the cause. A case fails because the skill mishandled a class of input; patching the skill to special-case that one input passes the case and leaves the class broken, and often the patch is a hardcoded branch that makes the skill worse. Read each finding as a representative of an intent category and fix the category. A case where the skill invented a fact absent from the source is not "handle this memo," it is "the skill does not ground its output in the provided source," and the fix belongs at that level.

When a proposed fix reaches for ALL-CAPS ALWAYS or NEVER or a stack of MUSTs, treat that as a yellow flag, the same way the leanness scanner does. Shouting at the model is usually a sign the fix is patching a symptom; a sharper outcome statement or a small worked example generalizes where a louder rule does not. Prefer the version that explains the reasoning over the version that issues the command.

## Why each guard is here

| Guard | What it prevents |
|---|---|
| opt-in | a loop applying changes the user never authorized |
| stakes calibration | the same aggressiveness on a throwaway and a depended-on skill |
| eval confirms the scan | auto-fixing a finding the evidence does not support |
| one change per round | a round whose delta cannot be attributed to a specific fix |
| revert on regression | building the next round on a change that made things worse |
| round bound | a loop that runs away instead of handing back to a human |
| full memlog trail | reasoning that lives only in the model and cannot be audited |
| benchmark as guardrail, human as judge | treating a green run as proof the change is correct |
| generalize to intent | a hardcoded patch that passes the case and leaves the class broken |
