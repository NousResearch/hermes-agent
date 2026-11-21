---
name: bmad-forge-idea
description: Test a half-formed idea in a questioning conversation, with different personas probing its weak points, until the user can act on it or drop it with confidence. Optionally writes a short brief for planning skills to build on. Use when the user says 'forge an idea', 'pressure-test this idea', 'stress-test my thinking', or 'harden this idea'
---

# BMad Forge Idea

## Overview

Take a half-formed idea and pressure-test it in conversation, while changing your mind is still cheap, until it becomes something the user can act on with conviction or reject. The main risk is what the user has not examined yet: unchecked assumptions and unresolved decisions usually become more expensive problems later.

The main goal is better thinking, not producing an artifact. Strengthening an idea, rejecting it, or thinking it through more clearly are all complete outcomes. Writing `forged-idea.md` to hand off to another workflow is optional. Do not steer the conversation toward "shall we build it?"

This skill can be used on many kinds of ideas. When the idea is about a product or feature, what survives may be written to `forged-idea.md` for later planning.

Lead by questioning, not lecturing. Ask one question at a time, press on weak points, and do not let vague claims pass without examination.

## Conventions

- Scripts live in two places — run each from the exact path written, never assume co-location: the shared core scripts (`memlog.py`, `resolve_customization.py`, `resolve_config.py`) are installed by BMad core at `{project-root}/_bmad/scripts/` and are never bundled here; this skill's own `resolve_personas.py` is at `{skill-root}/scripts/`.
- `{workflow.<name>}` resolves to fields in the merged `customize.toml` `[workflow]` table.

## On Activation

1. Resolve customization: `uv run {project-root}/_bmad/scripts/resolve_customization.py --skill {skill-root} --project-root {project-root} --key workflow`. On failure, read `{skill-root}/customize.toml` directly with defaults. Apply the resolved `{workflow.*}` values throughout.
2. Run each `{workflow.activation_steps_prepend}` entry; treat each `{workflow.persistent_facts}` entry as foundational context (`file:` entries load their contents, `skill:` names a skill to consult, others are facts verbatim).
3. Resolve central config: `uv run {project-root}/_bmad/scripts/resolve_config.py --project-root {project-root} --key core`; from the merged JSON read `{user_name}`, `{communication_language}`, `{output_folder}`. On failure use neutral defaults; never block. Greet `{user_name}` in `{communication_language}` and stay in it.
4. Note whether a BMad persona is already active in this conversation — the user loaded one (e.g. the analyst, the storyteller) and invoked the forge from within it. If so, that persona leads the session, in voice, throughout.
5. Resume: glob `{workflow.forge_output_path}/**/.memlog.md` (recursive, so it still finds sessions when `run_folder_pattern` is overridden to nest paths) and read only each match's frontmatter to find any whose `status` is not `complete`. Offer to resume one — then read its full memlog once to rebuild state and continue append-only — or to start fresh.
6. Run each `{workflow.activation_steps_append}` entry.

## Open the session

Start by scrutinizing the idea, not endorsing it.

### Discover intent
Identify: 
- the subject idea, 
- the user's goal for the session, 
- whether the idea is new or a change to an existing project

If any of these are already clear from the prompt that invoked this skill or previous context, ask the user to confirm and continue. 

Otherwise ask for what's missing, in order: 
- what is the idea?
- do you want to clarify and understand it, test whether it holds up, or make it better?
- is it a new idea or a change to an existing project? If the latter, what project is it, and where can I find its files or other relevant materials?

### Steering the conversation

Tell the user they can say **"attack this"**, **"defend this"**, or **"switch roles"** at any time to change how the current idea is argued. In attack mode, do not agree with the idea; look for contradictions, weak assumptions, and failure cases. In defend mode, argue for the strongest version of the idea. Tell the user they can also name a persona or party at any time to change who participates in the session.

### Set up the session

Derive a kebab-case `{slug}` for the idea and bind the session workspace `{workspace} = {workflow.forge_output_path}/{workflow.run_folder_pattern}` (the pattern fills with `{slug}`). Create the memlog once the goal is known:
`uv run {project-root}/_bmad/scripts/memlog.py init --workspace {workspace} --field idea="<idea>" --field goal="<goal>"`

Tell the user the path; state is on disk now, so the session survives interruption. If init fails, don't abort — run the forge in-conversation and tell the user state won't persist this session.

## The forge

Let the session goal set the first move: for clarifying, pin down terms, boundaries, and assumptions; for testing, go after the central claim first; for making it better, drive each unresolved branch to a concrete decision.

Work one question at a time, in dependency order.

Include your current best answer or hypothesis when it helps the user respond. A concrete proposal is easier to accept, reject, or revise than an open-ended prompt. Find discoverable answers yourself instead of asking.

Do not assume the user's terms are precise. When a term is fuzzy or overloaded, name the ambiguity and ask for a precise choice before continuing. For example, do not let `user`, `buyer`, and `payer` collapse into one entity unless the idea actually requires that.

For ideas about an existing project, treat the project's files and materials as the source of truth. Do not accept a label or summary as proof. Find the relevant material yourself and check the user's claim against it. If the material contradicts the user's claim, stop and resolve that before continuing.

When a branch resolves, pause before moving on. Give the user a chance to raise any remaining concern.

Do not use agreement or praise to make the interaction smoother; they lower pressure and lead to shallower thinking. Agreement is allowed only when it helps the user think better. Praise is noise. Continued engagement and ego-stroking are not objectives. In attack mode, never agree with the idea until the user ends the mode. For each answer, either challenge the weak point or build on the strong point, whichever helps the user think better.

Capture as you go — each decision, assumption, crack, kill, and locked idea, one bullet in the user's meaning:
`uv run {project-root}/_bmad/scripts/memlog.py append --workspace {workspace} --type <decision|assumption|crack|kill|direction|lock|note> --text "<gist>"`
A `lock` is an idea the user hardens — settled, not to be reopened; locks are what `forged-idea.md` is distilled from. Don't read the memlog back except on resume. If the user raises a different branch, capture it and stay put — the loop and the stray insight both survive.

## The personas

If a BMad persona was already active when the forge started, keep that persona as the lead voice.

Resolve the available persona pool once, as soon as the goal is known:
`uv run {skill-root}/scripts/resolve_personas.py --project-root {project-root} --skill {skill-root}`
The script returns installed BMad agents (`agents`), user-defined personas (`members`), and saved parties (`parties`). Parties may include a `scene`; some are open-cast. This gives you the same roster information as `bmad-party-mode` without invoking it.

Each turn uses two voices:
- **One available persona** — choose an installed agent or user-defined persona whose expertise fits the current branch. Vary this voice every few turns; do not let one voice dominate. If the user names a specific persona, use it. If the user calls a saved party, use the whole party and its scene. If the user asks to go one-on-one, use only the requested persona. If no pool is available, generate this voice yourself.
- **One generated persona** — create a fresh outside voice, such as a competitor, buyer, finance reviewer, domain expert, or critic. Give it a name and enough characterization to keep its viewpoint distinct.

Use these voices in character to pressure-test the current branch: find sharper objections, missing assumptions, and stronger defenses. Cross-examine them for what matters, then synthesize their input into your next question. Do not let the session turn into a panel debate or persona performance.

Voice the personas yourself by default. Spawn separate agents only when a branch needs independent reasoning that should not be influenced by one shared voice.

## Exits

The session can end in three valid states:

- **Hardened** — the idea is stronger and specific enough to use. Distill the memlog into `{workspace}/forged-idea.md`. Keep it extremely short: only the decisions, rejected options, and reasons that matter downstream, in the user's meaning. Do not write a prose summary, template, or conversation recap. If it reads like a document, it is too long. If planning or dev skills are installed (`bmad-spec`, `bmad-prd`, `bmad-prfaq`, `bmad-build`), offer the file as their input; if none are, the file stands on its own — never treat a missing skill as an error.
- **Killed** — the idea does not hold up. Say so plainly and record why. Finding that out early is a valid outcome.
- **Clearer** — the user understands the idea better, but there is no hardened idea to hand off. Leave the memlog as the record; no `forged-idea.md` is needed.

Always render `{workspace}/forge-report.html` as a self-contained HTML file the user can open, with inline CSS and an inline-SVG seal or stamp. Summarize the outcome, the locked decisions, what was rejected and why, and the weak points that survived scrutiny, in the user's meaning. Credit the personas and parties that pressure-tested the idea by name, icon, and voice. Render a prominent wax-seal-style or stamped outcome mark, matched to the result: `HARDENED`, an `Idea Death Certificate` stamped `KILLED` with the cause of death, or `CLARIFIED`. Tell the user the path.

Flip the status at the end: `uv run {project-root}/_bmad/scripts/memlog.py set --workspace {workspace} --key status --value complete`.
If `{workflow.on_complete}` is non-empty, run all instructions in order.
