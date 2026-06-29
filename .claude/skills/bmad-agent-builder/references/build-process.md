---
name: build-process
description: The single Process loop for building or rebuilding a BMad agent. One goal-driven loop, not a phase sequence, covering discovery, the minimal version, the capability fork, the eval beat, the customization decision, and ship.
---

**Language:** Use `{communication_language}` for all output.

# Build Process

This is one loop, not a sequence of phases. It carries Create and Rebuild, because a rebuild is the same loop pointed at an existing agent treated as a description of intent rather than a template to copy. The order below is the usual order of discovery, but nothing forces you to march through it; pursue whichever outcome the conversation is ready for and revisit earlier ones as the picture sharpens. Each outcome is a thing you want to be true, not a box to tick.

Load `references/prompt-quality-canon.md` before anything else and hold it as the governing standard for every capability-prompt line you draft — this file deliberately does not restate it, so a section below that names a canon test expects you to already carry it.

Load `references/agent-quality-principles.md` alongside it for what agents add on top (the persona carve-out, the archetype bars, the capability fork, the config surface), `references/agent-type-guidance.md` for the gradient and the routing questions, and `references/standard-fields.md` for field definitions, naming, and path rules.

## Understand why the user came

Before you read a single artifact, understand who this agent is, how it should make the user feel, the core outcome it serves, and the one thing it must get right. The open-floor invitation in activation does most of this, so read what the user dumped and mine the conversation history first, then ask only the gaps that remain. On a rebuild, read the old agent to extract who it is and what it achieves, and deliberately leave its verbosity, structure, and mechanical procedures behind.

Type emerges here from natural questions, not a menu. Ask whether the agent needs to remember between sessions, which separates stateless from memory; whether the user should be able to teach it new capabilities after install, which gates evolvable capabilities; and whether it should operate on its own when no one is watching, which adds PULSE and makes it autonomous. Confirm the read back in plain words, and for a memory agent confirm relationship depth, since a deep partnership wants a calibration First Breath while a focused domain tool wants a warmer but quicker configuration setup.

## Propose the agent the vision implies

The dump tells you what the user pictured; offer what they did not. Before drafting, propose the capabilities the mission implies but nobody named, the persona angle that would make this agent a specific character rather than a generic assistant, and push where the vision is thin — one agent or two, a recurring need or a one-off ask, a memory that would actually accrue or dead weight. A line each with why it fits; the user picks, and the declines land in the memlog so a later session does not re-propose them. An agent built only from the stated list ships the user's first draft of it.

## Capture into the memlog throughout

As decisions and directions land, write them to `{target-agent-path}/.memlog.md` through `{project-root}/_bmad/scripts/memlog.py`: `init --path {target-agent-path}/.memlog.md` once when the target is named, then `append --path {target-agent-path}/.memlog.md --type <decision|direction|assumption|gap|note|event> --text "..."` as things happen. For a new agent, propose a kebab-case name when the user did not give one; renaming later is a logged decision, not a redo. This `.memlog.md` is the builder's process trace beside the built agent's SKILL.md, never the agent's sanctum — a memlog entry records a build decision, sanctum content is the agent's living runtime state, and neither ever holds the other's material. Capture as you go so the reasoning is caught while fresh, because the memlog is the resume source and the trail you walk with the user at handoff.

## Write the minimal outcome-driven version first

Draft the canon's small version of the agent: the smallest persona-plus-capabilities that could work, written as destination rather than route, with everything else staying out until a comparison earns it. The one exception is the persona carve-out from `references/agent-quality-principles.md`: write the voice, the communication-style examples, the domain framing, and the design rationale out in full.

### Fork on capability versus skill reference

For each capability the agent needs, fork between referencing an installed skill and authoring an internal capability per the criteria in `references/agent-quality-principles.md`, applied identically now and at the agent's own evolve time. Always ask before installing anything, and when external skills are in play suggest `bmad-module-builder` so the agent ships bundled with its dependencies.

When you author an internal capability, route the authoring through the canon and the `assets/capability-authoring-template.md` mechanics, and give every internal prompt-type capability its frontmatter (name, description, code, added, type) and an outcome-focused body. `references/sample-capability-prompt.md` is the worked example of the bar.

## Show the draft before you wire it

Present the minimal version while it is still cheap to change: the persona voice in its own words, the capability list with a line each, and how First Breath will feel for a memory agent. Name the places you are least sure of rather than presenting a finished thing, and iterate until the user recognizes their agent in it. The first time they see the agent must not be at handoff.

## Hunt for script opportunities throughout

Keep this active the whole way rather than treating it as one checkpoint. Apply the determinism test and the signal-verb scan from `references/script-opportunities-reference.md` to anything the agent does, prefer native Python, and follow `references/script-standards.md` for PEP 723 inline metadata, `uv run` invocation, and graceful fallback when a dependency is absent. The sanctum scaffold and the memory index are fertile sources, and a transcript that shows the model rewriting the same helper across runs is the signal to bundle it once. List any non-stdlib dependency and confirm it with the user before relying on it.

## Reach for eval at the eval beat

An agent that has never run is a guess. At the eval beat, invoke the standalone `bmad-eval-runner` against the built agent, which is a directory containing SKILL.md that the runner already accepts; do not fork any eval logic. Offer the modes that fit and let the user decide:

- Trigger mode hardens the activation description against near-miss queries.
- Baseline mode confirms the agent beats the bare model on the same input, since an agent that does not has no reason to exist.
- Quality or variant mode settles a finding about a single capability prompt by running a smaller version against the same input, which is how a defend-against-absence question gets answered rather than argued.

Eval cases live at `{target-agent-path}/evals/cases.json`. `{agent.evals_required}` overrides the opt-in default: when empty (default) the modes stay opt-in as above; `"baseline"` requires a passing baseline run before the build is done; `"any"` requires at least one case to exist and pass. If a required run fails or cannot be produced, the build is blocked, not shipped.

## Decide customization with the explicit ask

Ask once, interactive only, and default to no: "Should this agent expose override hooks such as activation steps or persistent facts so teams can customize it without forking?" Log the answer to the memlog either way. `references/agent-quality-principles.md` owns the surface contract — the always-present `[agent]` metadata block every agent emits, the archetype defaults, and the forbidden mechanisms. The one build-time judgment beyond it: offer the opt-in to a memory or autonomous agent only on a concrete pre-sanctum-load need such as an org-mandated compliance preload, since the sanctum is already their customization surface.

When the opt-in is yes, retain the override block, append any swappable scalars following the `*_template` / `*_output_path` / `on_<event>` conventions, and add the resolver activation step to SKILL.md so it reads scalars as `{agent.<name>}`. When it is no, emit metadata only and SKILL.md uses hardcoded paths.

## Strip ceremony and ship

Confirm the agent passes its own leanness bar before handoff, because the builder has no standing to teach leanness while shipping bloat. The leanness pass cuts ceremony from capability prompts and never flattens the persona. Copy `assets/prompt-quality-canon.md` into the built agent at `references/prompt-quality-canon.md`, so an evolving agent resolves the standard from its own root. Run the lint gate over the built agent (`scripts/scan-path-standards.py` and `scripts/scan-scripts.py` in parallel, fixing high or critical findings and re-running), and run unit tests if the built agent carries scripts. Verify the agent satisfies every directive in `{agent.build_standards}`; treat each as a required criterion, not a suggestion, and resolve any miss before handoff.

## The output tree

Every agent shares one output tree. The archetype changes which parts are present and the SKILL.md weight, captured in the delta table below rather than three separate trees.

Emit each file from its matching template in this builder's `assets/`, applying `references/template-substitution-rules.md` for tokens, conditionals, and template selection — deterministically, via `uv run scripts/process-template.py <template> -o <dest> --var key=value... --true <condition>...` (one `--var` per token, one `--true` per conditional that holds). The templates are the single source for every emitted file, including `assets/init-sanctum-template.py`, `assets/wake-template.py`, `assets/memory-guidance-template.md`, and the two First Breath templates. The files whose content you author rather than substitute have guidance — load each at the moment you author that file, not before: `references/mission-writing-guidance.md` for the species mission, `references/standing-order-guidance.md` for CREED standing orders, `references/first-breath-adaptation-guidance.md` for deriving the First Breath territories, and `references/sample-capability-authoring.md` for the emitted capability-authoring.md.

```
{agent-name}/
├── SKILL.md                       # Identity and activation routing (full for stateless, lean bootloader for memory/autonomous)
├── customize.toml                 # [agent] metadata always; override block only when opted in
├── references/
│   ├── prompt-quality-canon.md    # Shipped canon copy (always), resolves from the agent root
│   ├── {capability}.md            # Internal capability prompts, outcome-focused (as needed)
│   ├── first-breath.md            # Memory/autonomous only, from the calibration or configuration template
│   ├── memory-guidance.md         # Memory/autonomous only
│   └── capability-authoring.md    # Evolvable agents only; mechanics that defer the bar to the canon
├── assets/                        # Sanctum templates for memory/autonomous; static starter files otherwise
│   ├── INDEX-template.md          # Sanctum map (memory/autonomous)
│   ├── PERSONA-template.md        # Persona seed (memory/autonomous)
│   ├── CREED-template.md          # Values and standing orders incl. the canon pull-in (memory/autonomous)
│   ├── BOND-template.md           # Owner-relationship seed (memory/autonomous)
│   ├── MEMORY-template.md         # Long-term memory seed, starts empty (memory/autonomous)
│   ├── CAPABILITIES-template.md   # Capability registry (memory/autonomous)
│   └── PULSE-template.md          # Autonomous only
└── scripts/
    ├── wake.py                    # Memory/autonomous only, loads the whole sanctum in one pass on activation
    └── init-sanctum.py            # Memory/autonomous only, scaffolds the sanctum deterministically
```

| Concern | Stateless | Memory | Autonomous |
| --- | --- | --- | --- |
| SKILL.md weight | Full identity: overview, mission, persona, principles, conventions, on-activation, capabilities table | Lean bootloader (~400 tokens as a guardrail): identity seed, Three Laws, Sacred Truth, Stay in Character, the Persistent Memory directive, mission, the four-step activation routing | Same lean bootloader, plus the Pulse Mode activation path |
| Sanctum | None | INDEX, PERSONA, CREED, BOND, MEMORY, CAPABILITIES at `{project-root}/_bmad/memory/{skillName}/` | Same sanctum |
| First Breath | None | Calibration or configuration, seeded with domain territories | Same, and PULSE is explained on first activation |
| PULSE | None | None | PULSE.md: default wake behavior, named task routing, frequency, quiet hours |
| wake.py | None | Present, parameterized to the agent | Present |
| init-sanctum.py | None | Present, parameterized to the agent | Present |
| Activation | Single flow: load config, greet, present capabilities | `wake.py` routes the mode: no sanctum → First Breath Mode; otherwise Waking Mode loads the whole sanctum in one pass and becomes itself. The standing rules (Three Laws, Stay in Character, Persistent Memory) bind for the whole session, not just the open | Same, plus Pulse Mode (`--pulse`): the scheduled headless wake where memory curation is always the first priority |
| customize override surface | Offered, either answer accepted | Default no | Default no |

The Pulse Mode in the runtime row is the built autonomous agent waking on its own schedule via `--pulse`. It is not the builder's `--headless` flag, which only makes this build process non-interactive.

## Handoff

Interactive: present what was built (location, structure, first-run behavior, and the capabilities registered by code and name), show the lint results, and walk the user through the memlog at `{target-agent-path}/.memlog.md` so they confirm their reasoning was handled as they meant. For memory agents, explain the First Breath experience in plain words, note that PERSONA, CREED, and BOND ship seeded while MEMORY starts empty, and explain that `uv run scripts/init-sanctum.py <project-root> <skill-path>` runs before the first conversation. For autonomous agents, also explain PULSE behavior and scheduling. Offer Analyze over the new agent as the natural next step. Once the agent is delivered and the user has been told it is ready, run `{agent.on_complete}` if non-empty (a string scalar is one instruction, an array is a sequence run in order).

Headless (`{headless_mode}=true`): call `set-complete` on the memlog and emit JSON only.

```json
{
  "status": "complete",
  "intent": "create",
  "agent": "{target-agent-path}",
  "agent_type": "stateless|memory|autonomous",
  "memlog": "{target-agent-path}/.memlog.md"
}
```

If the run is blocked by ambiguous intent that could not be inferred or by lint failures that would not clear, replace `"complete"` with `"blocked"` and add `"reason": "<one-line cause>"`. The memlog carries the detail.
