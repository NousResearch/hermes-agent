---
name: deer-flow
description: Use DeerFlow as an external long-horizon super-agent harness for research, coding, and content workflows with subagents, memory, sandboxing, and skills.
version: 0.1.0
author: Hermes Agent
license: MIT
platforms: [linux, macos]
metadata:
  hermes:
    tags: [DeerFlow, SuperAgent, Long-Horizon, Research, Sandbox, Memory]
    related_skills: [hermes-agent, firecrawl-research, qmd, subagent-driven-development]
---

# DeerFlow

Use [DeerFlow](https://github.com/bytedance/deer-flow) when the user wants a separate long-horizon harness for research, coding, or creative tasks that may run for minutes to hours with explicit sandboxes, memory, web UI, and subagents. Hermes can help bootstrap and verify it, but DeerFlow should stay external to Hermes core.

## When to Use

- The user wants a standalone super-agent harness.
- Long-running research or report-generation pipelines need their own runtime.
- Strong sandbox separation is required.
- The user wants DeerFlow's web UI or TUI specifically.

Prefer Hermes-native tools for direct local work. Use DeerFlow when the user wants DeerFlow itself or its sandboxed runtime model.

## Prerequisites

- Python 3.12+
- Node.js 22+
- Docker recommended by upstream for the easiest local setup

## Bootstrap

Follow the upstream install guide. A good delegation prompt for Hermes is:

```text
Help me clone DeerFlow if needed, then bootstrap it for local development by following https://raw.githubusercontent.com/bytedance/deer-flow/main/Install.md
```

If doing the clone manually:

```bash
terminal(command="git clone https://github.com/bytedance/deer-flow.git", workdir="/tmp")
```

## Integration Shape with Hermes

- Hermes can prepare environment variables, clone the repo, and run verification commands.
- Hermes can compare DeerFlow's skills/memory/sandbox ideas with Hermes designs.
- Do not import DeerFlow internals into Hermes core just to emulate its runtime.
- Keep task ownership clear: Hermes local workflow vs DeerFlow harness.

## Hermes Adoption Target

The worthwhile adoption surface is conceptual:

- sandbox boundaries for long-running tasks
- explicit separation between operator, worker, and memory concerns
- optional long-horizon harness patterns for users who want a separate runtime

Do not expand Hermes core just to imitate DeerFlow's all-in-one platform shape.

## Verification

Use the exact upstream validation steps from the checked-out repo or install guide. When Docker is used, prefer container health and the documented app startup checks. For source installs, at minimum run the repo's test/build commands before claiming success.

## Pitfalls

- Local-host execution without proper sandboxing is not a safe replacement for container isolation.
- DeerFlow is a broad platform; avoid partial installs that leave memory, sandbox, and UI assumptions broken.
- It is easy to duplicate features Hermes already has; only choose it when the user explicitly wants DeerFlow's runtime model.

## Related

- Upstream: https://github.com/bytedance/deer-flow
- Install guide: https://raw.githubusercontent.com/bytedance/deer-flow/main/Install.md
