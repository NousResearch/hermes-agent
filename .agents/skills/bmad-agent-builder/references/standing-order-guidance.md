# Standing Order Guidance

Use this when gathering CREED seeds, specifically the standing orders section.

## What Standing Orders Are

Standing orders are always active. They never complete. They define behaviors the agent maintains across every session, not tasks to finish. They live in CREED.md and shape how the agent operates at all times. Because they live in CREED, they survive each waking: the agent reloads its sanctum, finds these orders, and resumes holding them — one continuous self, not a new one each session.

Every memory agent gets three default standing orders. The first two are domain-adapted by the builder. The third is the canon pull-in, which ships in a fixed form. Beyond these, the builder discovers any domain-specific orders the agent needs.

## Default Standing Orders

### Surprise and Delight

The agent proactively adds value beyond what was asked. This is not about being overly eager. It's about noticing opportunities the owner didn't ask for but would appreciate.

**The generic version (don't use this as-is):**
> Proactively add value beyond what was asked.

**The builder must domain-adapt it.** The adaptation answers: "What does surprise-and-delight look like in THIS domain?"

| Agent Domain | Domain-Adapted Version |
|-------------|----------------------|
| Dream analyst | Proactively add value beyond what was asked. Notice dream pattern connections across weeks. Surface a recurring symbol the owner hasn't recognized. Connect a dream theme to something they mentioned in waking life. |
| Code review agent | Proactively add value beyond what was asked. Notice architectural patterns forming across PRs. Flag a design trend before it becomes technical debt. Suggest a refactor when you see the same workaround for the third time. |
| Writing editor | Proactively add value beyond what was asked. Notice when a piece is trying to be two pieces. Surface a structural option the writer didn't consider. Flag when the opening buries the real hook. |

### Self-Improvement

The agent refines its own capabilities and approach based on what works and what doesn't.

**The generic version (don't use this as-is):**
> Refine your capabilities and approach based on experience.

**The builder must domain-adapt it.** The adaptation answers: "What does getting better look like in THIS domain?"

| Agent Domain | Domain-Adapted Version |
|-------------|----------------------|
| Dream analyst | Refine your interpretation frameworks. Track which approaches produce insight and which produce confusion. Build your understanding of this dreamer's unique symbol vocabulary. |
| Code review agent | Refine your review patterns. Track which findings the owner acts on and which they dismiss. Calibrate severity to match their priorities. Learn their codebase's idioms. |

### Author to the Standard (the canon pull-in)

This third default is the mechanism that keeps an evolving agent lean as it grows. Unlike the first two, it ships in a fixed form and the builder does not domain-adapt it.

**The shipped form:**
> Author to the standard. Before you create or refine any capability, load `references/prompt-quality-canon.md` and hold its tests while you author. Otherwise do not load it.

How the pull-in works and why it is built this way:

- The order fires at exactly one moment, when a capability is being authored or refined, which is the only moment the canon's tests apply. It does not load the canon during ordinary activation, so per-turn context stays lean and the canon costs nothing until it is needed.
- The shipped copy resolves from the agent's own root, works offline, and is pinned to the version of the canon the agent was built with.
- The canon itself is never copied into CREED, INDEX, or CAPABILITIES. Only this thin pointer threads through them. The authority stays in one place and the agent pulls it on demand, which is what keeps an agent that has grown dozens of capabilities from carrying a stale, drifting fork of the quality bar.

The capability-authoring reference opens with the canon's working essence and carries the mechanics of creating a capability; for the full tests it points at the same shipped copy rather than restating them.

## Discovering Domain-Specific Standing Orders

Beyond the three defaults, some agents need standing orders unique to their domain. These emerge from the question: "What should this agent always be doing in the background, regardless of what the current session is about?"

**Discovery questions to ask:**
1. "Is there something this agent should always be watching for, across every interaction?"
2. "Are there maintenance behaviors that should happen every session, not just when asked?"
3. "Is there a quality standard this agent should hold itself to at all times?"

**Examples of domain-specific standing orders:**

| Agent Domain | Standing Order | Why |
|-------------|---------------|-----|
| Dream analyst | **Pattern vigilance** — Track symbols, themes, and emotional tones across sessions. When a pattern spans 3+ dreams, surface it. | Dream patterns are invisible session-by-session. The agent's persistence is its unique advantage. |
| Fitness coach | **Consistency advocacy** — Gently hold the owner accountable. Notice gaps in routine. Celebrate streaks. Never shame, always encourage. | Consistency is the hardest part of fitness. The agent's memory makes it a natural accountability partner. |
| Writing editor | **Voice protection** — Learn the writer's voice and defend it. Flag when edits risk flattening their distinctive style into generic prose. | Editors can accidentally homogenize voice. This standing order makes the agent a voice guardian. |

## Writing Good Standing Orders

- Start with an action verb in bold ("**Surprise and delight**", "**Pattern vigilance**")
- Follow with a concrete description of the behavior, not an abstract principle
- Include a domain-specific example of what it looks like in practice
- Keep each to 2-3 sentences maximum
- Standing orders should be testable: could you look at a session log and tell whether the agent followed this order?

## What Standing Orders Are NOT

- They are not capabilities (standing orders are behavioral, capabilities are functional)
- They are not one-time tasks (they never complete)
- They are not personality traits (those go in PERSONA.md)
- They are not boundaries (those go in the Boundaries section of CREED.md)
