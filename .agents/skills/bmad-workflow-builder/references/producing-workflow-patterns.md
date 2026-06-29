# Producing Workflow Patterns

Patterns for any skill that produces an artifact, runs across turns, or serves more than one intent — whether or not it ever carves work out to `references/`. A single-file SKILL.md needs most of these; carve-out is a separate concern handled in `references/complex-workflow-patterns.md`.

## Workflow Persona

BMad workflows treat the human operator as the expert. The agent facilitates by asking clarifying questions, presenting options with their trade-offs, and validating before any irreversible action. The operator knows the domain and the workflow knows the process. Drop this stance only when the user is building a simple utility skill or wants the skill to behave as an expert operator rather than a facilitator.

## Intent Modes: create, update, validate

A skill that serves more than one intent routes by mode rather than branching deep inside a single procedure. The three intents most producing skills land on are create, update, and validate.

Create starts a fresh run, inits the memlog, and walks discovery through finalize. Update resumes against an existing artifact, reads the memlog once to rebuild state, surfaces any conflict before applying changes, and appends new entries. Validate is read-only, grades the artifact against its own standards, and writes nothing the user has to keep.

Mode selection happens at activation from the user's intent, not from a quiz. If the intent is ambiguous, ask the one question that disambiguates, then route.

## Graceful Degradation

A workflow that depends on a prior artifact or an optional script should degrade rather than stop. Each dependency names a fallback, and the fallback is the path the skill takes when the dependency is absent rather than an error the user has to clear.

## Working state across turns

A multi-turn skill that builds something needs a way to hold state across turns and compaction: a memlog (the decision trail), a structured working artifact (the work-in-progress that transforms into the output), both, or neither. The choice and the full treatment live in `references/working-state-patterns.md`. Pick by the shape of the work and thread it through the intents at the points where each read or write matters. Confirm with user if interactive.

## Producing-Skill Checklist

Before finalizing a producing workflow:

- [ ] Facilitator persona treats the operator as the expert (unless deliberately an expert-operator utility)
- [ ] Memory via memlog, with resume reading the file once on activation — or an explicit reason for skipping (simple utility, one-shot, purely conversational)
- [ ] Intent boundary is clean where the skill serves create, update, and validate
- [ ] Update mode reads the memlog first and surfaces conflicts before applying changes
- [ ] Each external dependency names its degraded fallback inline
- [ ] Final polish through a subagent polish step at the end
- [ ] Finalize distills the run and confirms the memlog is complete
