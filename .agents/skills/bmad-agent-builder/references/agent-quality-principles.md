# Agent Quality Principles

The build-plus-scan bar for agents. Loaded at build time so the author works to the standard from the start, and at analysis time so every lens verifies against the same standard.

The universal core lives in the canon, not here. For writing the destination, the tests, the two-version comparison, the deeper floor, the cheaper signals, and the habit, load `references/prompt-quality-canon.md` (shipped copy, resolves from the agent-builder root). Everything below is what agents add on top of that core, because an agent is not a workflow and a few things change.

## Persona is the deliverable

The leanness bar from the canon applies to every internal capability prompt an agent carries. It does not apply to the persona, and this carve-out is load-bearing.

Persona voice, communication-style examples, domain framing, design rationale, and theory-of-mind are investment, not waste. They are the context that lets the agent make judgment calls when a situation does not match any capability prompt, and they are what makes the agent feel like a specific character rather than a generic assistant answering in the house style. A leanness pass never recommends flattening an agent's voice, never trims a communication-style example down to a rule, and never strips the warmth or the framing that gives the persona its shape. The pruning test cuts a capability prompt line when a capable model would produce the same outcome without it. The same test does not cut persona, because the outcome of persona is the character itself, and a flatter version is a different and worse outcome.

So the distinction the canon draws between structure that boxes the model in and intent that frees it cuts differently for persona. The capability prompt says what success looks like and lets the model find the path. The persona is the path the model takes through every capability, and it is the one part of an agent you write out in full.

## The three archetypes

Agents sit on a gradient surfaced as feature decisions, not a menu of separate architectures. Type emerges during discovery and branches only at emit time. `references/agent-type-guidance.md` is the authority on the gradient and the routing questions; the rules below are the quality bar each archetype is held to.

Stateless ships everything in one SKILL.md: overview, mission, identity, communication style, principles, conventions, on-activation, and the capabilities routing table. The whole identity is present at activation, so the leanness bar applies to the capability prompts while the persona content earns its place by the carve-out above.

Memory ships a lean bootloader SKILL.md carrying the identity seed, the Three Laws, the Sacred Truth, Stay in Character, the Persistent Memory directive, the mission, and the four-step activation routing. Everything else lives in the sanctum. The bar here is that communication style, detailed principles, and capability menus must not leak into the SKILL.md, because that content belongs in the sanctum and a bootloader that carries it is a pruning failure. There is no separate session-close section: session close folds into the Persistent Memory directive (capture as you go plus a consolidating pass at close), and the detailed memory guidance loads on the first memory-touch.

Autonomous is the memory agent plus PULSE.md for default wake behavior, named task routing, frequency, and quiet hours, and it gains the Pulse Mode (`--pulse`) activation path. The bar adds that PULSE owns autonomous behavior and nothing PULSE-shaped belongs anywhere else.

## The bootloader is lean by design, not under-built

A memory or autonomous bootloader SKILL.md is supposed to be small, around four hundred tokens as a guardrail rather than a gate. A leanness lens that flags a thin bootloader as missing content has it backwards. The bootloader carries only the DNA needed to find the sanctum and become the agent again; its thinness is the design working, not a gap. Judge a bootloader by whether sanctum-bound content leaked into it, not by its weight.

## The sanctum dimensions

The sanctum is the built agent's runtime memory, the place it reloads on every waking to become itself again, living at `{project-root}/_bmad/memory/{skillName}/`. This is a different thing from the builder's process log, the memlog, which is the builder's own trace written to `.memlog.md` beside the agent's SKILL.md while authoring. The two never blur. When this file or any file you write says memory of the sanctum, it means the agent's runtime memory and never the builder's log.

The sanctum is held to these dimensions:

- All six standard templates exist: INDEX, PERSONA, CREED, BOND, MEMORY, CAPABILITIES. PERSONA, CREED, and BOND carry meaningful seeds rather than empty placeholders, and MEMORY starts empty because it fills at runtime.
- First Breath carries the universal calibration and configuration mechanics plus domain-specific territory beyond the universal set, and the birthday ceremony is present.
- CREED carries its standing orders domain-adapted with concrete examples, including the canon pull-in standing order so an evolving agent authors new capabilities to the current standard.
- wake.py exists and loads the whole sanctum in one pass on every activation, and init-sanctum.py exists with First Breath owning the scaffolding step that runs it. Both match the skill name, and init-sanctum.py's template list matches the templates actually shipped in assets.
- After init runs, the sanctum is self-contained: the agent depends on the skill bundle only for First Breath and init, never for normal operation.

## Internal capability versus a reference to an installed skill

An agent either references an installed skill or carries an internal capability, and both meet the same bar. The capability prompt describes what success looks like; the persona informs how. Choose between the two forms with these criteria, applied identically at build time and at evolve time:

- Reference an installed skill when a skill already covers the capability. Suggest the reference, and always ask before installing anything.
- Author an internal capability only when the capability is genuinely novel, or when it is tightly coupled to the persona such that a generic skill would lose the agent's voice or context.
- When external skills are in play, suggest `bmad-module-builder` to bundle them so the agent ships with its dependencies.

Every internal capability is held to the canon, the same outcome-driven, leanness, and progressive-disclosure standard a standalone skill meets. An internal capability is not a place where the bar relaxes; it is a skill that happens to live inside an agent, and the only thing that changes is that the persona supplies the how.

## customize.toml is the sole config mechanism

Every agent emits a customize.toml. It carries an always-present `[agent]` metadata block (code, name, title, icon, description, agent_type) because that is the install-time roster contract the installer reads, even for an agent that declines the override surface. The override half (activation_steps_prepend, activation_steps_append, persistent_facts) is opt-in, defaults NO for memory and autonomous because the sanctum is their customization surface, is offered for stateless, and defaults NO in headless.

customize.toml is the only build-time configuration surface an agent has. There is no other mechanism, and these are forbidden:

- No installer question that configures the agent.
- No module.yaml authoring by the agent-builder.
- No separate config.yaml authoring as a build-time surface.
- No settings or toggle concept baked into the built agent.
- No identity, communication style, or principles in the customize surface, because that content belongs in PERSONA, CREED, and BOND.

First Breath config and init-sanctum.py are a separate concern and are not build-time configuration. They initialize the agent's runtime sanctum the first time it wakes, which is runtime state, not the build surface. Any customize.toml field that duplicates a sanctum concept is abuse, and First Breath must never be folded into customize.toml.
