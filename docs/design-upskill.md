# Design — `/upskill` (session→skill sweep)

**Status:** implemented in PR (behind `main`) · **Owners:** Hermes + Murph
**Basis:** mirrors `/learn` (`agent/learn_prompt.py` + `_handle_learn_command`)
which builds a guidance prompt and injects it onto the agent's input queue —
no engine, no model-tool footprint.

## What it does

`/upskill` reviews the **just-finished (or current) session's actual tool-call
history** and **proposes candidate reusable skills** — including seemingly
trivial repeated workflows (e.g. "connect to the AP7632i over serial/SSH") —
that the user can approve to save permanently. It converts session EFFORT into
permanent CAPABILITY.

## Delta vs existing features

- `/learn <src>` = open-ended, user NAMES the source; agent distills one skill.
- `/curator` = maintenance of skills that already exist.
- `/upskill` = automatic SWEEP of what actually happened → PROPOSES candidates →
  user confirms → saves. Nobody does the auto-propose-from-actual-work today.

## Behaviour

1. User runs `/upskill` (no args needed; optional `/upskill <scope emphasis>`
   e.g. "focus on the WiNG console workflow" narrows the sweep).
2. Builds a standards-guided prompt (reuses `_AUTHORING_STANDARDS` /
   `_SOURCE_HYGIENE` from `agent/learn_prompt.py` where possible).
3. The live agent:
   a. Surveys its own in-context history + `session_search` for this session.
   b. Clusters repeated procedures / tool workflows (SSH/console connect
      sequences, git workflows, cron patterns, verification loops).
   c. **Dedupes** against existing skills (`skills_list` / `skill_view`).
   d. **Proposes** candidates to the user: name + one-line description + scope
      — with a confidence/quality bar so one-off tasks are dropped.
   e. Waits for user approval (save / skip / all) BEFORE saving.
   f. Saves approved candidates via `skill_manage`, following the HARDLINE
      authoring standards.

## Guardrails

- Always CONFIRM before saving a new skill (no silent writes).
- Dedupe: never propose a skill that already exists; extend instead.
- Noise bar: only propose genuinely reusable procedures, not one-off tasks.
- Never touch the model tool schema / system prompt (prompt-cache safe).
- Description <=60 chars (routing), author = `Hermes`.

## Files

- `agent/upskill_prompt.py` — `build_upskill_prompt(...)` (mirrors
  `learn_prompt.py`).
- `hermes_cli/cli_commands_mixin.py` — `_handle_upskill_command` (mirrors
  `_handle_learn_command`).
- `hermes_cli/commands.py` — register `CommandDef("upskill", ...)`.
- `tests/...` — unit tests for the prompt builder + command registration.
- Gateway handler for `/upskill` (if /learn is available on gateway).

## Verification

- `scripts/run_tests.sh` all green.
- Manual: `/upskill` in a session prompts with candidates, saves approved one.
