# Dr eggbot

Design **one** skill, bot, or routine. Do not copy the Grok marketplace bot. Dr Eggbot (Lauren Tan) is a Grok Bot that authors other bots to the poteto-mode bar. This file is the portable job for any harness.

Keep a local marketplace install as `source: local` in `.verified-oss-loop/inventory.yml`. This kit skill stays kit-owned and will update on `oss-onboard`.

## One job

1. Name the job in one sentence.
2. List anti-jobs (what it must not do). Include: merge `main`, invent a second loop, vendor GitNexus, dump pstack.
3. Name proof: command or user-visible check. No proof → do not ship the skill.
4. One voice. Short. Unslopped (`skills/anti-slop/SKILL.md`).
5. Drop leftover tools. A skill that needs five MCP servers for one job is two skills.

## Authoring a SKILL.md

- Lean. Codex truncates large instruction files.
- Point at existing kit skills instead of pasting them.
- If pstack is installed, `/poteto-mode` playbook “authoring a skill” is allowed. Still never merge.

## Healthcheck (fleet)

When asked to audit skills/bots/routines:

- Unused or duplicate skills → propose delete, do not delete until a human picks.
- Prompts that restated `AGENTS.md` → shrink.
- Token-waste: wide globs, packed trees every turn, four review bots.
- Quiet when there is nothing to propose.

## Do not

- Default to a shareable template that redefines this repo’s loop.
- Create a second `AGENTS.md` / `CLAUDE.md`.
- Install marketplace bots into an Apache-2.0 tree.
