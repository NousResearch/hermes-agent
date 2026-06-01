---
name: hermes-memory-plugin-integration
description: "Connect Hermes plugins to memory safely."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [hermes, plugins, memory, ebbinghaus, testing]
    related_skills: [ebbinghaus-memory, hermes-agent-skill-authoring]
---

# Hermes Memory Plugin Integration Skill

Use this skill when a Hermes plugin should read from or write to an agent memory system without leaking private context. It is for plugin integration design, implementation review, and tests around memory-aware generated output.

This skill does not make memory mandatory for plugin operation. A plugin should keep its normal behavior when memory is disabled or unavailable unless the user explicitly requests a hard dependency.

## When to Use

- Use when modifying a Hermes plugin that should use Ebbinghaus recall, sleep, consolidation, or writeback.
- Use when social posts, briefings, cron outputs, or personal-agent embodiments need durable continuity.
- Use when reviewing whether plugin prompts clearly separate trusted memory from untrusted public content.
- Do not use for ordinary session memory questions that do not involve plugin code.

## Prerequisites

- The target plugin has a documented settings surface for optional features.
- The memory provider exposes a stable tool or local store for recall and writeback.
- Tests can create a temporary memory fixture instead of touching a real user database.
- `terminal`, `read_file`, `search_files`, and `patch` are available for code inspection and edits.

## How to Run

Begin with the memory substrate:

1. Prefer an official provider API or Hermes tool when one is available.
2. If only a local store exists, bridge through it defensively and keep the path configurable.
3. Keep the plugin usable when memory is missing, disabled, or malformed.

For Ebbinghaus-backed prototypes, use `HERMES_HOME` or `get_hermes_home()` to resolve state. Do not hardcode the default profile path.

## Quick Reference

| Concern | Rule |
| --- | --- |
| Recall | Keep snippets bounded, ranked, and truncated. |
| Prompting | Label memory as trusted context and public input as untrusted. |
| Writeback | Store concise artifacts with source and tags. |
| Secrets | Never store credentials, cookies, tokens, or raw auth state. |
| Tests | Use temp SQLite or provider mocks, never real memory stores. |

## Procedure

1. Find the plugin's generation or delivery path and identify where memory can be optional context.
2. Add configuration for enablement, provider or DB path, and recall limit.
3. Implement bounded recall with graceful failure when memory is unavailable.
4. Add prompt boundaries so private memory cannot be mistaken for user-approved public text.
5. Write generated artifacts back only after a draft or live action is created, with clear provenance tags.
6. Add targeted tests for enabled recall, disabled recall, missing-store behavior, and writeback.

## Pitfalls

- Do not make publishing, cron delivery, or UX-critical paths fail because optional memory is offline.
- Do not add heavy dependencies just for simple ranking if token overlap is enough.
- Do not let memory alignment bypass dry-run, approval, or live-action gates.
- Do not write tests against the user's real `HERMES_HOME`.
- Do not store private session dumps as public artifact memory.

## Verification

- Memory disabled: the plugin behaves exactly as before.
- Memory missing or malformed: the plugin continues without context.
- Memory enabled: relevant snippets enter the prompt with clear boundaries.
- Writeback stores concise content, source, tags, and timestamps.
- Tests run with temporary state and do not touch real user memory.

## References

- `references/lm-twitterer-ebbinghaus-bridge.md`
