---
name: hermes-omx-bridge
description: Bridge a chosen Hermes bot/profile to oh-my-codex (omx) for delegated coding work, repo analysis, multi-agent execution, and verified result reporting. Use when a user asks a specific Hermes bot or profile to hand implementation, debugging, tests, PR review, or repository investigation to `omx exec`, `omx ralph`, `omx team`, `omx explore`, or related OMX workflows and then summarize the outcome back to the user.
version: 0.1.0
author: Hermes Agent + oh-my-codex
license: MIT
prerequisites:
  commands: [omx]
metadata:
  hermes:
    tags: [hermes, omx, oh-my-codex, codex, delegation, multi-agent, coding-agent]
    related_skills: [hermes-agent, codex, claude-code, opencode]
    requires_toolsets: [terminal]
---

# Hermes ↔ OMX Bridge

Use this skill when the user wants a particular Hermes bot/profile to act as the orchestrator while oh-my-codex (`omx`) performs the coding work. The selected bot/profile clarifies scope, chooses the lightest OMX workflow, launches it from the target repository, monitors completion, verifies the result, and reports back in the user's language.

## User Request Pattern

The user-facing pattern is:

```text
Ask <bot-or-profile-name> to use OMX for <goal>.
Repository: <repo-path>
Acceptance: <tests, lint, behavior, docs, or report expected>
Constraints: <no unrelated rewrites, preserve existing work, no external publish unless asked>
```

Hermes should keep the chosen bot/profile identity in its own orchestration context. The prompt sent to OMX should focus on the coding task, repository path, constraints, acceptance criteria, and return contract; it should not invent or switch bot identities inside OMX.

## Preconditions

- `omx` is installed and authenticated on the same machine as Hermes.
- The task has a concrete target repository or working directory.
- Hermes has terminal/process tools available.
- Existing repository-local developer/agent guidance is authoritative. Mention that OMX must respect local repo guidance, but do not create, edit, copy, or persist agent configuration files unless the user explicitly asks.
- Do not paste secrets, tokens, or `.env` values into the OMX prompt. Refer to secret locations only when the local repo policy allows it.

Quick checks:

```bash
which omx && omx --version
omx doctor
```

## Workflow Selection

Choose the lightest OMX surface that can complete the job:

| User intent | Use | Notes |
|---|---|---|
| Find files, symbols, repo relationships | `omx explore --prompt "..."` | Read-only first stop for simple lookup. |
| Small/medium implementation, tests, refactor, debug | `omx exec --high "..."` | Default one-owner execution path. |
| Keep iterating until complete | `omx ralph "..."` | Use for acceptance-criteria-driven work that needs a persistent loop. |
| Independent parallel lanes | `omx team <n>:executor "..."` | Use only when work can be partitioned cleanly. |

Prefer `omx exec` unless another mode is clearly better.

For planning-first or repeated QA/test/fix work, prefer an explicit `omx exec --high` or `omx ralph` prompt that asks OMX to plan first or run a test/fix loop. Only use additional OMX workflow names when the local `omx --help` output confirms they are available as direct CLI commands.

## Prompt Contract

Before launching OMX, write a prompt that includes:

1. **Delegating Hermes bot/profile** — keep this in Hermes' orchestration notes, not as an instruction to change OMX identity.
2. **Working directory/repo** — exact path and branch constraints.
3. **Goal** — one concise outcome.
4. **Scope** — files, issue/PR numbers, or components to touch.
5. **Acceptance criteria** — tests, lint/typecheck, required behavior, report format.
6. **Safety constraints** — no forced history rewrites, no secrets, no unrelated rewrites, commit policy.
7. **Return contract** — changed files, verification commands/results, commit SHA/PR if applicable, remaining risks.

Template:

```text
Task: <goal>
Repo: <absolute path>
Scope: <files/issues/components>
Constraints: <no forced history rewrites, preserve behavior, respect repository-local guidance>
Acceptance criteria: <tests/lint/typecheck/docs expected>
Report back with: changed files, verification evidence, commit SHA if committed, remaining risks.
```

## Launch Patterns from Hermes

### One-shot coding task

Use a terminal call from the target repository. Keep the prompt quoted as one shell argument.

```text
terminal(
  command="omx exec --high -o /tmp/omx-result.txt 'Task: Fix failing tests in src/foo.test.ts. Constraints: respect repository-local guidance, do not rewrite history. Acceptance: npm test passes. Report changed files and verification.'",
  workdir="/path/to/repo",
  timeout=900
)
```

Then read the result file and inspect git/test state:

```text
terminal(command="cat /tmp/omx-result.txt && git status --short", workdir="/path/to/repo")
```

### Fully unattended trusted-local execution

Only use approval-bypassing flags when the user or repo policy explicitly allows local autonomous execution.

```text
terminal(
  command="omx exec --madmax --high -o /tmp/omx-result.txt 'Task: ... Acceptance: ...'",
  workdir="/path/to/repo",
  timeout=1800
)
```

### Long-running background task

```text
terminal(
  command="omx ralph 'Close issue #123. Acceptance: tests pass and commit with repo convention.'",
  workdir="/path/to/repo",
  background=true,
  timeout=60
)
# Poll with process tools until complete, then read logs/result artifacts.
```

### Parallel work

Only use team mode when lanes are independent and file ownership can be separated.

```text
terminal(
  command="omx team 3:executor 'Lane A: fix parser tests. Lane B: update docs. Lane C: add regression tests. Avoid shared-file conflicts; report separately.'",
  workdir="/path/to/repo",
  background=true,
  timeout=60
)
```

## Verification After OMX Returns

Hermes must not blindly relay the OMX final message. Verify enough to make the report trustworthy:

1. Check `git status --short` and `git diff --stat`.
2. Confirm expected files changed and unrelated files did not.
3. Run or inspect the verification commands OMX claims it ran when cheap enough.
4. If tests fail or evidence is missing, relaunch OMX with a narrower corrective prompt.
5. Summarize in the user's language.

Report format:

```markdown
Done.

- Delegated via: `omx <mode>`
- Changed files: ...
- Verification: `<command>` → pass/fail summary
- Commit/PR: ... (if any)
- Remaining risks: ...
```

## Failure Handling

- If the shell reports that the `omx` command is unavailable, ask the user to install/run `omx setup`, or fall back to the `codex` skill if available.
- Authentication/model errors → run `omx doctor`; do not retry blindly.
- Approval prompts hang → rerun with the repo-approved autonomy flag, or use a safer narrower prompt.
- Dirty worktree before launch → inspect `git status`; tell OMX to preserve existing changes and avoid overwriting them.
- Conflicting team workers → stop team mode if needed, preserve logs, and relaunch as one-owner `omx exec`.
- Missing evidence → rerun verification locally or ask OMX for a verification-only pass.

## Public Distribution Checklist

Before publishing this skill:

1. Test on a disposable repository with one read-only `omx explore` task and one small `omx exec` task.
2. Confirm examples use placeholders such as `<repo-path>` instead of private machine paths.
3. Keep default examples safe; document `--madmax` as trusted-local only.
4. Publish from the skill folder if Hermes supports registry publishing:

```bash
hermes skills publish "$PWD/skills/hermes-omx-bridge" --to github --repo <owner>/<hub-repo>
```

5. After install, verify discovery:

```bash
hermes skills list | grep hermes-omx-bridge
```
