---
name: copilot-cli
description: Delegate coding tasks to GitHub Copilot CLI.
version: 1.1.0
author: Ken Kuang (@sykuang), Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Coding-Agent, Copilot, GitHub, ACP, Code-Review]
    related_skills: [claude-code, codex, opencode, hermes-agent]
---

# Copilot CLI Skill

Use GitHub Copilot CLI as a coding worker through Hermes. This skill covers
trusted ACP delegation, one-shot print mode, and interactive sessions; it does
not expose ACP transport settings to the model.

## When to Use

- The user explicitly asks for GitHub Copilot CLI.
- A coding task benefits from an isolated subagent.
- A one-shot review or edit is easier through `copilot -p`.
- An iterative task needs Copilot's interactive terminal interface.

## Prerequisites

- Install GitHub Copilot CLI using an official package.
- Authenticate with `copilot login` or a supported GitHub token.
- Verify installation with `terminal(command="copilot --version")`.
- Run coding tasks inside the target repository.
- For ACP delegation, configure the trusted provider in `config.yaml`:

```yaml
delegation:
  provider: copilot-acp
  model: gpt-5.4
```

Choose a model available to the authenticated Copilot account. The operator
owns this configuration; never pass ACP commands or arguments through
`delegate_task`.

## How to Run

### ACP delegation

After configuring `delegation.provider`, call `delegate_task` with only its
model-facing task fields:

```text
delegate_task(
    goal="Fix the token refresh race and add a regression test.",
    context=(
        "Repository: /absolute/path/to/project\n"
        "Run the smallest relevant test.\n"
        "Do not modify the legacy authentication module."
    ),
)
```

For independent work, use batch mode:

```text
delegate_task(
    tasks=[
        {
            "goal": "Fix the authentication regression and add a test.",
            "context": "Repository: /absolute/path/to/project",
        },
        {
            "goal": "Review the API diff for correctness.",
            "context": "Repository: /absolute/path/to/project; report only.",
        },
    ],
)
```

Children inherit the parent's toolset. They do not inherit conversation
history, so include the repository path, constraints, and completion criteria
in `context`.

### One-shot print mode

Use `terminal` for a bounded task:

```text
terminal(
    command="copilot -p 'Review src/auth.py and report concrete bugs with line numbers.' --allow-all-tools --silent",
    workdir="/absolute/path/to/project",
    timeout=180,
)
```

Use `--allow-tool` instead of `--allow-all-tools` when a narrower permission
set is sufficient. Use `--allow-all` only when unrestricted path and URL access
is intentional.

### Interactive mode

Use a PTY for multi-turn work:

```text
terminal(
    command="copilot",
    workdir="/absolute/path/to/project",
    background=true,
    pty=true,
)
process(action="submit", session_id="<id>", data="Implement the fix and run its test.")
process(action="poll", session_id="<id>")
process(action="log", session_id="<id>")
process(action="write", session_id="<id>", data="\x03")
process(action="kill", session_id="<id>")
```

## Quick Reference

| Need | Use |
|------|-----|
| Isolated Hermes subagent | Trusted `copilot-acp` provider + `delegate_task` |
| Bounded scripted task | `copilot -p ... --silent` |
| Iterative session | `copilot` with `background=true, pty=true` |
| Specific model | `--model <id>` |
| Additional directory | `--add-dir <path>` |
| Narrow tool approval | `--allow-tool <pattern>` |
| All tool approvals | `--allow-all-tools` |
| All permissions | `--allow-all` or `--yolo` |
| Resume latest session | `--continue` |
| Resume selected session | `--resume` |

Permission patterns include `shell(git:*)`, `write`, and
`url(https://github.com)`. Deny rules take precedence over allow rules.

## Procedure

1. Confirm the target repository and requested outcome.
2. Verify Copilot CLI is installed and authenticated.
3. Pick ACP delegation, print mode, or interactive mode.
4. Give Copilot the repository path, constraints, and a concrete completion
   check.
5. Inspect the resulting diff or report.
6. Run the smallest relevant project test.
7. Stop any interactive background session.
8. Report the changed files and remaining limitations.

## Pitfalls

- Do not pass `acp_command`, `acp_args`, or per-task toolsets to
  `delegate_task`. Hermes intentionally keeps transport and toolset selection
  outside model control.
- ACP permission requests are denied by the Hermes ACP client. Operator-owned
  Copilot permissions must be configured before delegation.
- Non-interactive print mode needs pre-approved tools or it cannot proceed.
- `--allow-all-tools` does not grant unrestricted paths or URLs.
- Files outside the working directory need `--add-dir` or explicit path
  permission.
- Slash commands are interactive; describe the task normally in print mode.
- Interactive mode requires `pty=true`.
- Parallel agents must not share a working tree when both can edit files.

## Verification

Installation smoke test:

```text
terminal(command="copilot --version")
```

Print-mode smoke test:

```text
terminal(
    command="copilot -p 'Reply exactly COPILOT_SMOKE_OK' --allow-all-tools --silent",
    timeout=120,
)
```

ACP smoke test, after configuring `delegation.provider`:

```text
delegate_task(
    goal="Reply exactly COPILOT_ACP_OK without changing files.",
    context="This is a read-only connectivity check.",
)
```

Success means the command exits normally, the expected marker is returned, and
no files change during either smoke test.
