# Context-Refinement Exploration Checklist

Use this to pick the **2–4 highest-leverage questions** for Phase 2. A question
earns a slot only if its answer would change a milestone, an owner, a gate, or a
success criterion. Otherwise pick a default and record it as an assumption.

Batch the chosen questions into a single `AskUserQuestion` call. Phrase each with
concrete, selectable options plus room for the user's own answer.

## The seven dimensions

1. **Definition of done** — What does "finished" look like, observably? What
   artifact must exist, and who has to be able to use it?
2. **Audience / consumer** — Who is this for (internal, customers, a specific
   platform)? Drives voice, format, and review gates.
3. **Scope boundaries** — What is explicitly in vs. out? What is the smallest
   version that still counts as success (MVP vs. full)?
4. **Constraints** — Deadline, budget, brand voice/pillars, tech stack,
   compliance, things to avoid.
5. **Existing assets** — What already exists to build on (drafts, data, code,
   research)? Prevents redundant subtasks.
6. **Approval & risk** — What is outward-facing and therefore needs a human gate
   (publishing, sending, deploying, spending)? What is the cost of getting it
   wrong?
7. **Autonomy level** — How hands-free? Full auto with one final approval, or
   check-in at each milestone gate?

## Defaults when not asking

- **Done** → a complete, reviewer-approved artifact in its natural home.
- **Audience** → infer from the artifact; default to internal/safe voice.
- **Scope** → smallest coherent version that satisfies the stated outcome (MVP),
  with a noted path to expand.
- **Approval** → require HUMAN APPROVAL before any publish/send/deploy.
- **Autonomy** → full auto up to the first outward-facing gate.

## "Hands free / don't ask me" mode

Ask at most **one** blocking question, and only when proceeding on a default
would risk producing the wrong outcome entirely. Otherwise go straight to stated
assumptions and emit the `/goal`.
