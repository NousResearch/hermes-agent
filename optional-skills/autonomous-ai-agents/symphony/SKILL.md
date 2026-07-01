---
name: symphony
description: Use OpenAI Symphony as an external work-management layer that turns tracker items into isolated autonomous implementation runs with proof of work.
version: 0.1.0
author: Hermes Agent
license: MIT
platforms: [linux, macos]
metadata:
  hermes:
    tags: [Symphony, Work-Management, Linear, Autonomous-Runs, Proof-of-Work]
    related_skills: [hermes-agent, subagent-driven-development, using-git-worktrees]
---

# Symphony

Use [OpenAI Symphony](https://github.com/openai/symphony) when the user wants work items from a tracker such as Linear turned into isolated autonomous implementation runs with attached proof of work. Symphony is explicitly a low-key engineering preview and should be treated as an external system or design reference, not something to merge into Hermes core.

## When to Use

- The team wants issue-tracker-driven autonomous implementation runs.
- The user wants proof artifacts such as CI status, PR review feedback, complexity analysis, or walkthrough evidence attached to each run.
- The user is already adopting an external harness-engineering style workflow.

Prefer Hermes-native `delegate_task`, cron, and repo workflows for local-first work. Use Symphony when the user specifically wants Symphony's work-management model.

## Adoption Paths

### 1. Build from the spec

Symphony's repo explicitly supports implementing the system from `SPEC.md`:

```text
Implement Symphony according to the following spec: https://github.com/openai/symphony/blob/main/SPEC.md
```

### 2. Use the experimental Elixir reference implementation

```text
Set up Symphony for my repository based on https://github.com/openai/symphony/blob/main/elixir/README.md
```

## Hermes Integration Shape

- Hermes can read the spec, inspect the reference implementation, and adapt ideas into repo-local workflows.
- Hermes should not claim to be Symphony or silently recreate its entire control plane in core.
- Good adoption targets are patterns: isolated runs, proof-of-work artifacts, tracker-driven dispatch.

## Hermes Adoption Target

This is primarily a pattern library for Hermes:

- tracker item -> isolated run -> proof artifact is worth documenting
- spec-first implementation guidance is useful for advanced users building their own orchestration layer
- the preview implementation should stay external and optional
- avoid presenting Hermes as a drop-in Symphony replacement

## Verification

When working with the reference implementation, use the upstream validation commands from the Elixir README. Repository summaries indicate checks such as:

```bash
terminal(command="make -C elixir all", workdir="/path/to/symphony", timeout=900)
terminal(command="mix test", workdir="/path/to/symphony/elixir", timeout=900)
```

## Pitfalls

- Symphony is marked as an engineering preview for trusted environments.
- The spec-first nature means many users will need to implement or adapt it rather than install a polished product.
- Do not assume a drop-in Hermes plugin exists.

## Related

- Upstream: https://github.com/openai/symphony
- Spec: https://github.com/openai/symphony/blob/main/SPEC.md
