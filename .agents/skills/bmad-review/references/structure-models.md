# Structure Models

Reference shapes for the structure pass. Pick the one matching the document's purpose and evaluate the document against its rules; a document that fits none cleanly is judged against the closest model, with the mismatch itself noted as a finding when the shape fights the purpose.

## Tutorial/Guide (Linear)

**Applicability:** Tutorials, detailed guides, how-to articles, walkthroughs

- Prerequisites: setup/context MUST precede action
- Sequence: steps follow strict chronological or logical dependency order
- Goal-oriented: clear "Definition of Done" at the end

## Reference/Database

**Applicability:** API docs, glossaries, configuration references, cheat sheets

- Random access: no narrative flow required; the reader jumps to a specific item
- MECE: topics are Mutually Exclusive and Collectively Exhaustive
- Consistent schema: every item follows an identical structure (e.g., Signature → Params → Returns)

## Explanation (Conceptual)

**Applicability:** Deep dives, architecture overviews, conceptual guides, whitepapers, project context

- Abstract to concrete: Definition → Context → Implementation/Example
- Scaffolding: complex ideas built on established foundations

## Prompt/Task Definition (Functional)

**Applicability:** BMad skills and workflows, prompts, system instructions, agent definitions

- Meta-first: inputs, usage constraints, and context defined before instructions
- Separation of concerns: instructions (logic) separate from data (content)
- Explicit flow: execution order is stated, never implied

## Strategic/Context (Pyramid)

**Applicability:** PRDs, research reports, proposals, decision records

- Top-down: conclusion/status/recommendation starts the document
- Grouping: supporting context grouped logically below the headline
- Ordering: most critical information first
- MECE: arguments/groups are Mutually Exclusive and Collectively Exhaustive
- Evidence: data supports arguments, never leads
