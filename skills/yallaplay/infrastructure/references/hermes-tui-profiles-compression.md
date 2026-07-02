# Hermes TUI, profiles, skills, and compression notes

Use this reference when troubleshooting Claudio/Hermes operability from the TUI.

## `/skills` visibility and profiles

- `/skills` and `hermes skills list` show skills installed in the **active Hermes profile**, not repo-local `skills/` directories.
- Repo-local Claudio skills under `yallaplay-hermes-agent/skills/yallaplay/<name>/SKILL.md` are source artifacts. They should be exposed to the TUI by symlinking each skill directory into the active profile's `claudio-authored` category, e.g. `~/.hermes/profiles/claudio-lab/skills/claudio-authored/<name> -> <repo>/skills/yallaplay/<name>`.
- After changing installed skills, tell the user to run `/reload-skills` or start a new session.
- Keep profile symlinks intact when the user expects a skill to both ship with the repo and appear in the current TUI.

## TUI legibility style

The TUI can render some fenced blocks or long markdown regions with surprising vertical whitespace in long sessions. For operator-facing answers:

- Prefer compact bullets over fenced code blocks when showing short command lists or profile layouts.
- Avoid big markdown tables unless the table is the deliverable.
- If a renderer issue is suspected, describe it plainly and switch to a lower-friction format rather than repeating the same formatting.

## Context compression expectations

`/compress` compacts conversation history; it does not shrink everything in the next prompt.

Non-compressible or weakly-compressible prompt mass includes:

- system/developer/project instructions,
- tool schemas,
- loaded skills,
- persistent memory/profile text,
- protected head/tail messages,
- recent large tool outputs that are still in the protected tail.

Default compression behavior to remember:

- `compression.threshold` default is around `0.50` of main-model context.
- `compression.target_ratio` default is around `0.20` and controls tail token budget.
- `compression.protect_last_n` default protects recent messages.
- The summary itself can be thousands of tokens because it preserves decisions, files, commands, and blockers.

If the user asks why compression did not reduce context much, check whether recent large tool outputs or loaded skills are still in the protected tail. Practical options are `/new` or `/reset` for a hard reset, or more aggressive settings such as lower `compression.target_ratio` and `compression.protect_last_n` for future sessions.

## Known GitHub issue class

Blank space / odd scroll behavior in the TUI has had upstream issues around virtualized transcript row-height estimation and scroll clamp bounds. When investigating, search for terms like:

- `fix(tui): defer virtual clamp until resumed rows are measured`
- `fix(dashboard): prevent xterm.js viewport blank whitespace after long conversations`
- `fix(tui): strip ANSI before estimating message height`

Treat these as issue classes to search, not permanent negative claims about Hermes.