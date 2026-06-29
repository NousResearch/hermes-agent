# Creating a Party

A guided authoring flow that turns an idea — a themed cast, a one-off persona, or a pile of raw profile data — into custom party members and groups, written to the user's customize.toml override. The output is configuration; `bmad-customize` does the actual write.

## What you're producing

Sparse `[workflow]` override entries for `bmad-party-mode`:

- `[[workflow.party_members]]` — one per persona: `code`, `name`, `icon`, `title`, `persona`, optional `capabilities`, optional `model`.
- `[[workflow.party_groups]]` — when the personas form a named room: `id`, `name`, an optional freeform `scene`, `members` (codes), and `memory` (`true`/`false`). `members` is optional: leave it off for an open-cast room whose `scene` names a pool the model casts from on the fly. `memory` is whether the group remembers across sessions; ask the user when they don't say, default `false`.
- `default_party` — set only if the user wants this group to load by default.

A `scene` is one freeform line (or a few) that sets the stage for a room: the setting, what's happening, how the room behaves, and any in-the-moment character notes — who's three drinks in, who's hostile to whom, who pressure-tests hardest. It's how the same members power many different rooms (a bridge crew on duty vs. the same crew off-duty in the lounge vs. a hostile buyer panel). Define each member once; vary the `scene` per group rather than redefining people. There's no fixed vocabulary — write it plainly and the model plays it.

The `persona` field is the whole game. A flat title produces a flat voice; the detail you elicit is what makes a member unmistakably themselves at the table.

## Find the shape

Open by understanding what they're building. Three common shapes — stay open, anything that yields distinct voices is fair game:

- **A cast** — a themed ensemble ("the Star Trek TOS bridge crew", "a board of famous investors"). Several members plus a group that holds them.
- **One-offs** — a persona or two added to the collective, no group needed.
- **Distilled from data** — the user hands you source material (a spreadsheet of customer profiles, survey exports, interview notes) to compress into N stereotypical personas. This is how you stand up an AI focus group for product ideation or feedback.
- **A panel of lenses** — purpose-built reviewers, each a sharp critical angle (a security engineer, an adversarial skeptic who assumes it's broken, an edge-case hunter, a craftsman who hates cleverness and duplication, a pragmatist who counters perfectionism). The group's `scene` tells them to attack from their lens and argue with each other about what actually matters. A great adversarial-review or red-team room.
- **Open-cast** — no fixed roster at all. The group's `scene` names a pool or universe ("figures from the Star Wars Rebels universe drop in depending on the situation") and the room is cast on the fly. Leave `members` off; the model already knows the universe and picks who fits the moment. Anchor a face or two by listing them if some should always be present.

Ask which they're after if it isn't obvious, then proceed.

**Persisting a cast already in play.** When you arrive here from a live session — the user spun up an ad-hoc cast inline and wants to keep it — the personas are already drafted and voiced. Don't re-interrogate: capture them as they've been playing, give the group an `id` and name, ask the memory and default questions, and go straight to the write.

## Editing an existing party

When the user wants to change a party that already exists (retune a member's persona, add someone to a group, swap the default), read the current state first so you change rather than clobber: `uv run {project-root}/_bmad/scripts/resolve_customization.py --skill {skill-root} --key workflow` returns the merged `party_members`, `party_groups`, and `default_party`. Show the member or group being touched, capture only the delta with the user, and hand that sparse change to `bmad-customize` — it replaces a `party_members`/`party_groups` entry whose `code`/`id` matches and appends the rest, so an edit is just the changed entry, never a full rewrite.

## Keeping new faces from a session

At the end of a remembered party, the room offers to keep the faces that showed up but aren't in its roster — characters cast from an open-cast scene, or members the user added on the fly. They're already drafted and voiced, so don't re-interrogate: capture each as they played (`code`, `name`, `icon`, a one-line `title`, and a `persona` drawn from how they came across), then add them as `party_members`. For a fixed-roster group, also list their codes in the group's `members` so they return as regulars. For an open-cast room, leave `members` empty — listing any member turns the room into a fixed roster and kills its on-the-fly casting; the saved personas now live in the collective, so the scene still names them and they can return without locking the room down. Hand that sparse delta to `bmad-customize` — for a built-in party with no override yet it creates one; for an existing override it merges the new members in.

## Distill from source data (when provided)

When the user points you at data — a file path, a pasted table, exported profiles — read it and compress it into the requested number of representative personas. Cluster by what actually differentiates behavior (goals, budget, pains, adoption posture), not surface demographics alone. Each cluster becomes one persona with a real name and face. Name your reasoning: tell the user which segments you found and which traits drove the split, so they can correct the cut before you flesh the personas out. If they didn't say how many, propose a number from the spread in the data and let them adjust.

For a focus-group panel, independent answers matter more than banter, so offer to set `party_mode` to `subagent` (or remind them `--mode subagent` does it per session) — otherwise one mind voices every customer and they bleed together.

## Flesh out each persona

Draft, don't interrogate. Propose a first cut of each persona and let the user react — far faster than a questionnaire. Push each one until it has a voice you could pick out blind. The dimensions that earn their place:

- **Identity** — name, a one-line title, an emoji that fits.
- **Voice & ethos** — how they talk, what they value, how they argue, their pet peeves.
- **Agenda** — what they're really after in any conversation; what they push for.
- **Quirks** — the specific, human details (a catchphrase, a bias, a blind spot).
- For focus-group personas, also **likes and dislikes**: what would make them champion or reject an idea, and their relationship to the product space.
- **Capabilities** (optional) — if this persona should research or read files when spawned, note it; it becomes soft guidance in their spawn prompt.

Keep pushing for specificity. "Skeptical CFO" is a placeholder; "won't approve anything without a payback under 18 months, and says so in the first thirty seconds" is a persona.

## Close it out

- Ask straight: **anything else about this party to specify** before you write it — a house dynamic, a missing voice, a member who should lead.
- Ask whether **this party should remember across sessions** (unless the user already said). Yes → `memory = true` on the group; no → `memory = false`. One-offs with no group skip this — memory is a group setting.
- Ask whether **this group should be the default party going forward**. Yes → set `default_party` to the group's id. One-offs with no group can't be a default; skip the ask.

## Write via bmad-customize

**First, check for code collisions.** A custom member whose `code` matches an installed agent silently *overrides* that agent in the collective. Before composing, resolve the collective once — `uv run {skill-root}/scripts/resolve_party.py --project-root {project-root} --skill {skill-root}` — and check each new member's `code` against the returned members. On a collision, surface it ("`analyst` would override the installed Analyst — intended, or pick a different code?") and let the user confirm or rename. One check, not a gate.

Compose the sparse override and hand it to `bmad-customize` to place, confirm, and write — target skill `bmad-party-mode`, `[workflow]` surface. Default to the **user** override (`bmad-party-mode.user.toml`); offer the **team** file when the party is meant to be shared. Hand it the exact entries: the `party_members` tables, any `party_groups` table (including its `memory` flag), and `default_party` if the user opted in. Keep it sparse — only the new entries, never a copy of the base customize.toml. `bmad-customize` shows the TOML, waits for an explicit yes, writes, and verifies the merge; don't write the file yourself.

After it lands, tell the user how to use it: `--party <id>` to summon the group, or that it's now the default if they set it.
