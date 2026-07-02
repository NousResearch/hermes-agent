# Hermes Profiles and YallaPlay Skill Visibility

Use this reference when a repo-local YallaPlay skill exists but does not appear in `/skills`, or when configuring which Hermes profile should own Claudio behavior.

## Mental Model

Hermes profiles are separate agent installations. A profile has its own config, sessions, memory, skills, gateway setup, cron jobs, plugins/MCPs, and auth/env state.

Repo-local skills under `./skills/yallaplay/` are project source files. They should be symlinked into the active profile because the `/skills` UI and `hermes skills list` read the active profile's installed skill registry, e.g. `~/.hermes/profiles/<profile>/skills/`. Use the `claudio-authored` profile category for a clear visual split from bundled/default skills.

## When To Use Profiles

Use profiles for durable isolation across sessions:

- Work vs personal agents.
- Lab vs production gateway/cron agents.
- Different provider/model stacks.
- Different gateway bots or platform routing.
- Multi-agent roles such as planner/coder/reviewer.
- Client/project isolation where memory, credentials, or tools must not bleed.

Do not create a profile for every repo or task. Use `.hermes.md` / `AGENTS.md` for repo-specific rules, skills for reusable workflows, `/new` or `/reset` for a clean conversation, and worktrees for git isolation.

## YallaPlay Claudio Pattern

Recommended profile split:

- `claudio-lab`: development/testing of Claudio skills, tools, wiki layout, provider config.
- `claudio-prod`: future production-facing Slack/gateway/cron profile, created only when ready for real delivery.
- `default`: generic fallback; avoid storing Claudio-specific assumptions there.

Promote from lab to prod deliberately. Do not auto-sync experimental tools or skills into prod.

## Skill Visibility Procedure

When the user asks why a skill is not visible in `/skills`:

1. Check whether the skill exists in the repo, e.g. `skills/yallaplay/analytics/SKILL.md`.
2. Check the active profile with `hermes profile list` or `/profile`.
3. Check installed skills with `hermes skills list`.
4. If needed, symlink the repo-local skill directory into the active profile, e.g. `~/.hermes/profiles/claudio-lab/skills/claudio-authored/<skill> -> <repo>/skills/yallaplay/<skill>`.
5. Ask/tell the user to run `/reload-skills` in the current TUI, or start a new session.
6. Verify with `hermes skills list` that the skill appears and is enabled.

## Commands

```bash
hermes profile list
hermes profile use claudio-lab
hermes profile show claudio-lab
hermes skills list
```

One-off invocation with a profile:

```bash
hermes --profile claudio-lab
```

Clone a future production profile from lab only after reviewing secrets/tools/gateway settings:

```bash
hermes profile create claudio-prod --clone-from claudio-lab
```

## Pitfalls

- Editing `skills/yallaplay/<name>/SKILL.md` updates repo source and the installed profile skill when the `claudio-authored` profile entry is a symlink.
- Creating or changing a profile skill symlink may still require `/reload-skills` or a new session for the current TUI to see it.
- Profile state lives outside the repo and affects future sessions; treat writes there as operational changes.
- Keep `claudio-prod` boring and locked down; test in `claudio-lab` first.
