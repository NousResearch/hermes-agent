---
name: runtime-debugging
description: "Runtime debugging umbrella for systematic root-cause work plus Python pdb/debugpy and Node.js inspector workflows."
---

# Runtime Debugging

Use this skill when something fails and you need to reproduce, localize, inspect runtime state, and add a guard so the bug stays fixed.

## Debugging law

Do not patch before understanding. Reproduce → localize → explain root cause → fix the minimal cause → verify → add a regression guard.

## Universal workflow

1. Capture the exact error, command, input, environment, and expected behavior.
2. Reproduce consistently with the smallest command/test.
3. Read the relevant code path and recent changes before editing.
4. Instrument or attach a debugger at the point where state diverges.
5. Fix the cause, not the symptom.
6. Run the failing reproduction, adjacent tests, and any requested verification.

## Python (`pdb` / `debugpy`)

- Use `pdb` for local synchronous inspection and `debugpy`/DAP for long-running apps or editor-style breakpoints.
- For pytest: run the narrow failing test first, then expand to the module/suite.
- Prefer temporary breakpoints or logging; remove debug code before finishing unless it is useful diagnostics.

## Node.js inspector

- Use `node inspect`, `--inspect`, or `SIGUSR1` attach for running Node processes.
- Read inspector URL/port from runtime output; avoid assuming 9229 is free.
- For TypeScript, ensure source maps/tsx invocation match the project.

## Verification evidence

Report the reproduction command that failed before, the command that passes after, and the specific code/behavior change that explains why.
## Support files

- `references/absorbed-skills.md` — list of original skill packages consolidated into this umbrella and where to recover full archived content.
