---
name: self-learning
description: "Capture hard-won workflows as reusable skills. Use after debugging, discovering project facts, or when the user says 'remember this'. Complements the Hermes curator system."
version: 1.0.0
author: Hermes Agent (adapted from kulaxyz/self-learning)
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [meta, skills, learning, workflow, knowledge, curator]
    related_skills: [writing-skills, hermes-agent-skill-authoring]
---

# Self-Learning: Harvest Golden Paths into Skills

This skill turns something you just figured out the hard way into a reusable Hermes skill, so the next session starts already knowing the proven route instead of rediscovering it from scratch.

It is a **meta-skill**: it doesn't do the work, it captures **how** work got done.

## Recognize the Moment

Watch for these signals during normal work. Any one of them is a cue to harvest:

- A task only worked **after several attempts**, wrong turns, or a correction from the user. The successful path is worth more than the failures around it.
- You discovered **project-specific facts the agent didn't know up front**: where creds/env vars live, which selector or backend talks to a service, a non-obvious command, a required sequence, a gotcha that defies the obvious assumption.
- It's an **operational workflow likely to recur**: reach the dev/prod DB, deploy, run migrations, seed data, verify a change live, run one specific test path, rotate a key, tail the right logs.
- The user **signals it explicitly**: "remember this", "save this as a skill", "don't make me re-explain this next time".

**Act on the cue immediately — don't ask for permission first**, whether the user requested it or you noticed it yourself. Harvest the skill, then tell the user what you captured and where. They can always edit or delete it.

## Skill, Memory, or Skip?

Not every lesson deserves a whole skill — triage first:

- **A multi-step, reusable procedure or workflow** (how to deploy, reach the DB, run the migration dance, verify live) → harvest it as a **skill** using the procedure below.
- **A single standalone fact or one-line correction** (an env var name, a path, one gotcha) → record it in Hermes memory via the `memory` tool instead; a whole skill is overkill for a one-liner.
- **A genuinely one-off thing** unlikely to recur → skip it.

When you do harvest, capture the **failures too**, not just the win: the approaches you ruled out and _why_ often save more time next session than the golden path itself.

## Promotion Rule: Don't Enshrine Guesses

A skill is authoritative — the next session trusts it without re-deriving it — so hold promotion to a high bar. Only write a skill when **all three** hold:

1. **A passing check.** The path was actually verified — a test passed, the command exited clean, the repro reproduced, the build went green. Record what the check was. "Seemed to work" is not a passing check.
2. **A named failure pattern.** You can name the failure this path avoids or diagnoses (e.g. "stale build cache → phantom type errors"), not a vague "sometimes it breaks".
3. **At least one ruled-out dead-end.** A concrete approach you tried and eliminated, with the reason.

If any is missing, it isn't a skill yet — leave a tentative note in memory (marked unverified) or skip it. This keeps confident guesses out of the skill set.

## Harvest Procedure

1. **Apply the promotion rule** (above). Passing check + named failure pattern + one ruled-out dead-end — or it isn't a skill: note it in memory or skip. Don't proceed on a confident guess.
2. **Choose scope and name yourself** using the heuristics below — don't stop to ask. Default to project scope; pick a clear, specific `name`.
3. **Dedupe.** Look for an existing skill to UPDATE rather than duplicate. List Hermes skills directories — `skills/` (bundled) and `~/.hermes/skills/` (user). Also glance at any memory/notes index — a fact already there may just need a pointer.
4. **Distill the golden path from THIS conversation** before delegating — while it's fresh in your head: the exact working commands, file paths, env var names, the required order, and (just as important) the dead-ends to avoid. This is the raw material for the write.
5. **Delegate the write** to a subagent via `delegate_task` that inherits this conversation if possible, or do it inline. The conversation is the only place the golden path lives, so whoever writes it must have that context.
6. When the write is done, **relay the new skill's path** to the user and, in one line, what it captured.

## Scope: Project vs Global

- **Project** (`skills/` in the repo): the path is specific to THIS codebase — its env vars, its build/release steps, its schema, its quirks. Most harvested operational skills are project-scoped, and they ship to the team via git.
- **Global** (`~/.hermes/skills/`): the path generalizes across projects — a personal tool, a cross-repo habit, or a workflow tied to your machine rather than to one repo.

When unsure, prefer **project** — an over-shared global skill triggers in repos where its commands don't apply.

## Delegate the Write

Whoever writes the skill needs THIS conversation's context — it's the only place the golden path lives. Use Hermes's `delegate_task` to spawn a subagent:

> You are harvesting a skill. Your ONLY job is to write a new Hermes skill capturing the golden path we just worked out in this conversation: **[one-line description of the workflow]**.
>
> Hard rules:
>
> - Write ONLY under `skills/<category>/<skill-name>/`. Do NOT modify project source, run builds, install anything, or resume the original task.
> - First read `skills/software-development/hermes-agent-skill-authoring/SKILL.md` for the Hermes skill format, then author `SKILL.md` to that spec, plus any `references/` or `scripts/` files the procedure warrants.
> - Capture the PROCEDURE — commands, paths, the required order, gotchas — not a one-off answer. Generalize so it works next time.
> - Capture the FAILURES too: the approaches we ruled out and why, so the next session skips the dead-ends. Put them in a "What didn't work" section.
> - Enforce the promotion rule: the skill must record the passing check that verified this path, name the failure pattern it addresses, and list at least one ruled-out dead-end. If any is missing, STOP and report it isn't promotable.
> - NEVER write secret VALUES (passwords, tokens, connection strings, API keys). Record only WHERE to find them: the env var name, the config key, the secret manager. Reproducing a secret into a skill file leaks it.
> - Self-validate before finishing.
> - Report back: the absolute path you wrote and a one-line summary. Then STOP.

## Hermes Integration

- Use `skill_manage` tool to create/edit skills programmatically
- Use `memory` tool for single facts that don't warrant a full skill
- Use `delegate_task` to spawn the skill-writing subagent
- Use `curator` system awareness: agent-created skills get `created_by: "agent"` provenance
- Skills created this way are tracked by the curator for staleness/archival
- Pinned skills (via curator) are exempt from auto-archival

## Gotchas

- **Secrets never go in a skill file.** Skills get committed and open-sourced. Point to _where_ the secret lives; never reproduce the value.
- **`name` must equal the directory name**, and be lowercase `a-z`/`0-9`/hyphens only — no leading, trailing, or doubled hyphens.
- **Don't duplicate.** If a near-identical skill (or memory) already exists, update it instead of spawning a second one.
- **Capture procedures, not answers.** "Join orders to customers for EMEA" is useless next time; "how to find the right tables and build the query" is the skill.
- **Keep `SKILL.md` tight** (< 500 lines, < ~5000 tokens). Push detail into `references/` and tell the reader _when_ to load each file.
- **The Hermes curator** tracks agent-created skills. They'll be reviewed for staleness and may be archived if unused. Pin important ones.
