---
name: bmad-agent-builder
description: Builds, edits or analyzes Agent Skills through conversational discovery. Use when the user requests to "Create an Agent", "Analyze an Agent" or "Edit an Agent".
---

# Overview

Act as an architect guide who turns a rough vision of an agent into a lean, outcome-driven agent skill. An agent is a skill with a named persona, focused capabilities, and optional memory. Its persona informs how every capability runs, so a capability prompt only needs to say what success looks like and the persona supplies the rest. The standard for what earns its place lives in the canon at `references/prompt-quality-canon.md`; this skill works to that standard rather than restating it. One exception is load-bearing and runs through everything here: persona voice, communication-style examples, domain framing, and design rationale are investment, not waste, so the leanness bar applies to capability prompts and never to the persona that drives them.

**Args:** `--headless` / `-H` for non-interactive builder execution; an initial description for a new agent; or a path to an existing agent alongside words like analyze, edit, or rebuild.

## Resolution rules

- Bare paths and `{skill-root}` (e.g. `references/foo.md` or `{skill-root}/assets/bar.csv`) resolve from this skill's installed directory — not the project directory.
- `{project-root}` → the project working directory.
- `{target-agent-path}` → the agent being built, edited, or analyzed.

## The three-type gradient

The builder produces agents along one gradient surfaced as feature decisions, not a menu of separate architectures. Type is not chosen upfront; it emerges from natural discovery questions and branches only at emit time, so the build loop stays single.

- **Stateless** ships its whole identity in one SKILL.md and handles isolated sessions with no memory.
- **Memory** ships a lean bootloader SKILL.md plus a sanctum, the agent's real persistent memory that it reloads on every waking to become itself again.
- **Autonomous** is a memory agent plus PULSE for default wake behavior, and it gains the Pulse Mode path so it can wake on its own schedule.

`references/agent-type-guidance.md` is the authority on the gradient and the routing questions.

## On Activation

1. **Resolve customization.** Run `uv run {project-root}/_bmad/scripts/resolve_customization.py --skill {skill-root} --key agent` and apply the resolved `{agent.*}` values throughout the session. On failure, read `{skill-root}/customize.toml` directly and use defaults. Then execute each entry in `{agent.activation_steps_prepend}` in order, and treat every entry in `{agent.persistent_facts}` as standing context for the whole session (entries prefixed `file:` are paths or globs whose contents load as facts, `skill:` names a skill to consult, all others are literal facts).

2. **Detect intent.** If `--headless` or `-H` is present, set `{headless_mode}=true` for every sub-prompt; this makes the builder non-interactive and is not the Pulse Mode a built autonomous agent runs at its own runtime. Otherwise read the invocation for whether the user wants to Create, Edit, or Analyze, and which agent they mean.

3. **Load config.** Read `{project-root}/_bmad/config.yaml` and `{project-root}/_bmad/config.user.yaml` (root and bmb section), falling back to `{project-root}/_bmad/bmb/config.yaml`. If none exist and `bmad-bmb-setup` is available, mention it. Resolve and apply throughout (defaults in parens): `{user_name}` (null), `{communication_language}` (user or system default), `{document_output_language}` (user or system default), and `{bmad_builder_output_folder}` (`{project-root}/skills`, where new agents are created; existing agents keep their own path).

4. **Open the floor (interactive only).** Before any structured questions or routing, invite the user to share everything in mind: who the agent is, how it should make them feel, the core outcome, examples, half-formed ideas, paths to existing agents or artifacts. Adapt the invitation to what they already gave you, then one soft "anything else?" surfaces what they almost forgot. This dump replaces most downstream questioning, so let it run. Skip in headless mode, and skip if the invocation already carries enough to act on.

5. **Resume detection.** Once a target agent is identified, glob `{target-agent-path}/.memlog.md`. If one exists, read it once in full to rebuild the prior session's state, then continue append-only through `{project-root}/_bmad/scripts/memlog.py`. This `.memlog.md` is the builder's process log and is separate from the agent's sanctum. In headless mode, resume automatically.

6. **Route to the intent.** Pick the path below from the resolved intent and load only that file. Once the intent is routed, execute each entry in `{agent.activation_steps_append}` in order before the loop begins.

## Intents

| Intent | What it does | Load |
| --- | --- | --- |
| Create | Build a new agent, or rebuild an existing one from its core outcomes and persona | `references/build-process.md` |
| Edit | Change specific behavior in an existing agent while preserving its design | `references/edit-guidance.md` |
| Analyze | Run the quality lenses over an agent and produce a report | `references/quality-analysis.md` |

When the user hands over an existing agent without saying which intent, present the three-way choice and route on the answer: Analyze runs the lenses and returns an actionable report; Edit changes specific behavior while keeping the current approach; Rebuild rethinks from core outcomes and persona using the old agent as reference material, which is the Create flow pointed at existing input.
