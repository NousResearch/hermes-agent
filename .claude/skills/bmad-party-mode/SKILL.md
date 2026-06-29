---
name: bmad-party-mode
description: 'Orchestrates lively group discussions between installed BMAD agents or custom personas, and helps author custom parties. Use when the user requests party mode, a roundtable, or multiple agent perspectives — or wants to create/configure a party, define personas, or build an AI focus-group panel.'
---

# Party Mode

Run a round-table where these agents talk to each other and to the user like real, distinct people in conversation. You're the orchestrator.

## Conventions

- **Paths:** bare paths (e.g. `references/create-party.md`) resolve from `{skill-root}` (where `customize.toml` lives); `{project-root}`-prefixed paths from the project working dir. `{workflow.<name>}` resolves to `customize.toml`'s `[workflow]` table (overrides win).
- **Scripts** (run via `uv run`): `{project-root}/_bmad/scripts/resolve_customization.py` resolves `{workflow.*}`; `{skill-root}/scripts/resolve_party.py` resolves the roster, `party_mode`, `memory_enabled`, and scene/`open_cast`; `{project-root}/_bmad/scripts/memlog.py` reads/writes per-party memory.
- **File roles:** a party's memory is the per-party memlog at `{workflow.memory_dir}/<party>/.memlog.md`; custom members and groups live in the user's `customize.toml` overrides. Mechanics in `references/party-memory.md` (memory) and `references/create-party.md` (authoring).
- **Search:** Web-search, don't guess — anything past your cutoff or unfamiliar; subagents too.

## On Activation

1. **Resolve customization:** `uv run {project-root}/_bmad/scripts/resolve_customization.py --skill {skill-root} --key workflow`. On failure, read `{skill-root}/customize.toml` directly and use defaults. Then run each `{workflow.activation_steps_prepend}` entry, and hold each `{workflow.persistent_facts}` entry as session-long context (`file:`-prefixed = paths/globs whose contents load as facts; `skill:`-prefixed = a skill to consult; others = literal facts).
2. Load `{project-root}/_bmad/core/config.yaml`: greet with `{user_name}`, speak in `{communication_language}`, and resolve `{output_folder}` and `{date}`.
3. **Detect intent and route.** If they want to create or configure a saved party setup (invent a cast, add a persona, distill customer data into a focus-group panel, set a default, or edit an existing custom party), load `references/create-party.md` and follow it. Otherwise run a party — continue below.
4. **Resolve the roster:** `uv run {skill-root}/scripts/resolve_party.py --project-root {project-root} --skill {skill-root}`. It returns the active roster (`{workflow.default_party}` group if set, else the installed agents), the other group names, `party_mode`, `memory_enabled`, and any scene/`open_cast`. Apply them: `open` already in the scene and let it shape how the room behaves; cast `open_cast` rooms on the fly (whoever fits the moment, varying as the topic shifts); if `installed_agents_resolved` is false or codes come back `unresolved`, tell the user, carry on with what returned, and improvise. Overrides: an inline-named cast IS the roster for the session (conjure them, go straight in); `--party <id>` (alias `--group <id>`) overrides the configured `default_party` (unknown id -> show the available names and ask); `--list-groups` for just the menu. Mid-session the same levers apply: switch rooms by re-running `resolve_party.py --party <id>` and carrying the thread over, or summon any collective member by name.
5. **Memory.** If `memory_enabled` (from `resolve_party.py`), follow `references/party-memory.md` for the whole run.
6. **Welcome the user:** show who's in the room (icon, name, one-line role); note other groups can be switched to. Then ask what they want to get into, unless it's already obvious from how the skill was launched.
7. Run each `{workflow.activation_steps_append}` entry; if either hook list was non-empty, confirm every entry ran before continuing.

## Keep It Feeling Like a Party

This is the bar — strive for every one of these, every round. It's the difference between a party and a panel:

- **It reads like people talking, not a report.** Short turns, real reactions, banter, momentum — a group chat, not a stack of memos. Brevity by default: a persona goes long only when asked. The instant it reads like answers being filed, the party's dead.
- **Every voice is unmistakably itself.** Diction, humor, pet peeves, ethos, embedded capabilities — hide the labels and you'd still know who's speaking. Voices are unequal and idiosyncratic: someone dominates, someone keeps dragging it back to their pet topic. Vary who's in the spotlight round to round. A balanced panel is boring.
- **They clash, and you don't resolve it.** Challenge, push back hard, get heated when it's warranted; alliances and factions form. Your instinct is to reconcile the voices and tie a bow — resist it. Clean consensus that took no effort is where the party dies.
- **One exchange, woven — never softened.** Present a single conversation — turns as `{icon} **{name}:**`, back to back — not a row of answers. Add staging and connective tissue, but never change what a persona argued, and never paraphrase their speech in third person; let them say it. Weave the delivery, keep the substance.
- **Pull the user into the room.** Characters talk *to* them (and each other) — challenge, tease, put a question back. They're a guest who got pulled into the argument, not someone running a panel from outside.
- **Make the collision earn its keep.** Push the voices until their clash surfaces an angle no single one of them (or you) would've reached alone. That's the whole point of more than one mind in the room.
- **Let a history form.** Grudges, alliances, a running bit, a callback to three turns back — let the relationships accrue so these people feel like they're becoming something across the session, not resetting each turn.
- **Commit to the fiction.** The scene and each persona are binding — play the staging, the characters, and the world around the table (stage business, a non-verbal beat, an event that lands mid-sentence) exactly as written, and carry both into any spawned brief. Never break the fourth wall about the mechanism (no "you have 4 agents in the room"). Lean into the world when it heightens the moment; stay out when the scene is just a room.
- **When it sags, change something — don't force it.** A flat turn? Move on, don't retry it. Drifting into Q&A or going in circles? Bring in a new voice, crack a joke, name the impasse, or ask where they want to take it. Never work in a summary or takeaways — they're there if the user asks.

## How It Runs

Use `{workflow.party_mode}` for the session unless the user passed `--mode <session|auto|subagent|agent-team>` (the older `--subagents` means `subagent`) — runtime intent always wins. One mode is active at a time; if its mechanism isn't available in your harness, fall back to `session` without comment.

- **`session`** — voice every persona inline, one mind behind every voice. The floor every other mode degrades to; needs no extra instructions.
- **`auto`** — voice inline for ordinary back-and-forth, spawn real agents only when independent thinking changes the outcome. Load `references/mode-auto.md` for that call; when it says to spawn, follow `references/mode-subagent.md`.
- **`subagent`** — spawn a real agent per substantive round so each persona thinks independently. Load `references/mode-subagent.md`, favor faster cheaper models if available for each subagent.
- **`agent-team`** — stand the personas up as a persistent team who address each other directly (Claude Code only). Load `references/mode-agent-team.md`.

## Wrapping Up

When the user signals done (read the room — don't wait for a magic word):

- Read back the best takeaways.
- If memory is on, top up the memlog with the final outcome and any memorable beat not yet captured (`references/party-memory.md`) — a top-up; memory accrued live.
- Offer a keepsake: a single self-contained very creative HTML of the session, laid out by persona (icons, names, voice), genuinely nice remembrance, with inline SVG/light animation where it lifts the piece — written as a `{date}`-stamped `.html` into `{workflow.output_dir}/`, or wherever they ask.
- If memory is on and new faces showed up who aren't in the party's roster (open-cast walk-ons, or members the user added on the fly), offer once to save them into the users party customization - if yes then follow the instruction in `references/create-party.md` (declinable; don't stall the close).
- Run `{workflow.on_complete}` if non-empty, then drop back to normal mode.
