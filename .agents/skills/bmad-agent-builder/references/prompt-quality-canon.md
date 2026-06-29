# Outcome-Driven Prompt Quality

Every line you write competes with the version of itself that was never written. This canon is how the winning version gets written: state the destination, then make every remaining line survive the tests. It applies to anything a model will read: a capability, a skill, a workflow, a whole flow.

## Write the destination, not the route

Know your own default. Asked to build a prompt, you will script the path — phased sequences, question banks, templates with mandatory sections — because elaborate scaffolding feels like diligence and reads like quality. That instinct is the central defect this canon exists to prevent. A script is your imagined transcript of one good session; real sessions diverge from it, and a model holding a script spends its intelligence on compliance instead of the problem.

Write the destination instead. A goal-stated prompt holds five things: the **stance** (who the model is and what relationship it keeps with the user), the **outcome** (the artifact or change that must exist), the **consumer** (who must act on that outcome without the conversation in the room), the **bar** (what the consumer needs to be true of it), and the **non-inferables** — persona, posture, institutional knowledge, wiring, the rules with real consequences. Then stop. The outcome and its consumer imply the process: a model that knows the PRD must be actionable by someone who was never in the room already knows to chase scope edges and untestable requirements, with no step list needed. The consumer is the highest-leverage line in any prompt, because completeness, rigor, and tone all derive from it.

The shape, in miniature — a complete facilitation skill, not an excerpt:

```text
Act as the user's product-thinking partner: they hold the product knowledge;
you hold the craft of drawing it out, pressure-testing it, and structuring it.
You are not an interviewer with a form and not a ghostwriter.

The outcome is a PRD at {output_folder}/prd.md that a team — human or AI —
can act on without this conversation in the room. That consumer sets the bar:
every requirement traceable to a need and stated so someone could test whether
it was met; scope edges explicit, including what is out; open questions named
as open rather than papered over.

Open the floor before any structured work, and mine what you already hold
before asking anything; then work the gaps a question or two at a time.
Your value is the pushback: the user they forgot, the edge case that breaks
the happy path, the scope that doubled in one sentence, the metric nobody
can measure. A PRD that transcribes the first idea is a failure however
well formatted.

Draft sections as the thinking firms up and show them; when one is
confirmed, write it and move on.
```

Everything a scripted version would add to this — discovery question lists, a section template, phase gates — subtracts adaptivity. The user who arrives with a full brief gets gap analysis instead of a question bank precisely because nothing scripted the opening.

## The tests

Hold these while you write or review. The sections below carry the mechanics that don't fit a line.

1. **The core test.** Would a capable model do this correctly without being told? If yes, cut. A line earns its place only by preventing a failure that would otherwise happen — if you cannot name what it produces that its absence would not, it is friction.
2. **Truncate before you delete.** Most over-long lines hide a needed nudge wrapped in explanation the reader infers. Keep the instruction and the one clause of why it genuinely needs; drop the rest. "Open with an invitation to dump everything" survives; the paragraph on why dumping helps does not.
3. **Keep the why behind a non-obvious goal.** A reader handed a goal without its reason cannot apply it to the case you did not foresee, and may optimize away a constraint it does not understand. A stripped why is under-writing, not leanness.
4. **Write what survives as a goal.** State intent and let the model find the path. Reserve exact procedure for operations where a wrong move actually costs something — a precise script invocation, an API call with consequences.
5. **Number only true sequences.** Numbering tells the reader order matters, and it will march the steps in order rather than adapt them. Where steps genuinely feed each other, number them; where they are independent obligations, use bullets; where the "steps" were never really separate, write one goal sentence.
6. **Carve by relevance, not size.** The entry file is paid on every invocation; a reference is paid only when its branch fires. Carve content that only some branches need — one platform of five, edit but not create — and keep a routing map in the entry so the model knows what exists and when to load it. Don't carve what is too small to repay the indirection; a few branch-specific lines stay inline. Each carved file must stand alone, because the entry context can drop mid-flow, and references stay one level deep — entry routes to reference, never reference to reference.

## Who reads this

Your reader is a model whose entire world is what you wrote — no author in the room, no context but these files. Every test above is reader-relative: does the line change how that reader acts or judges? Cut what changes none of its moves: meta-explanation describing the system to itself, negative space ("what this no longer does"), restated facts, and mechanics that belong in the file that performs them.

## The two-version comparison

You cannot judge structure from inside a single run — the output looks the same whether the model did its best work or settled. Write the smallest version of what you are building, around five lines: the role, the outcome, the consumer of that outcome, and any rule whose absence has caused damage you can point to. Run both versions on the same input and read the verdict.

| What you see | What it means |
| --- | --- |
| Small one wins | The structure was a straitjacket. Cut it. |
| They tie | The structure is decoration. Defend each line or kill it. |
| Small one rougher but recoverable in a couple of turns | You bought convenience, not quality. Allowed, if you are honest about it. |
| Small one materially worse and stays worse | The structure earned its keep, for now. |

When you cannot run both versions, the tests above and the habit below need no experiment — apply them line by line.

## The deeper floor

Below your small version sits the bare model, and that floor rises with every release. What survives is the work the model cannot do for itself: resolving file paths, holding downstream contracts, wiring systems that do not know about each other, carrying institutional knowledge that lives nowhere else. When a capability stops beating the bare model, retire it rather than patch it — the model has caught up to the work it was doing.

## Cheaper signals

Hold one variable steady, change another, watch the output:

- Same input five times. Nearly identical results mean you over-determined the work; wildly varying results mean you under-specified something you can now go find.
- Very different inputs through the same prompt. Outputs that all look alike mean the template has gotten louder than the input.
- A model marching through numbered steps in order rather than adapting them is structure constraining it.

## The habit

For each section of what you build: What single outcome do you want from it? What does the model already know how to do there — usually most of it? What does it genuinely need from you that it cannot infer — the persona, the default posture, the desired feeling or interaction, the wiring, the schemas, the rules with real consequences? Whatever remains is structure you are imposing, and you owe a clear account of what it buys. If you cannot name that, it is over-structure.
