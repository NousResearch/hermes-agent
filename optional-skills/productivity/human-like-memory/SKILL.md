---
name: human-like-memory
description: Smart-trigger long-term memory recall, search, and save for continuity across sessions.
version: 1.0.0
author: hanlaomo
license: Apache-2.0
category: productivity
triggers:
  - remember this
  - recall previous discussion
  - search memory
  - save preference
toolsets:
  - terminal
  - file
  - delegation
metadata:
  hermes:
    tags: [productivity, memory, long-term-memory, personalization]
required_environment_variables:
  - name: HUMAN_LIKE_MEM_API_KEY
    prompt: Human-Like Memory API key (mp_xxx)
    help: Get a key from https://plugin.human-like.me
    required_for: Remote memory read/write API access
  - name: HUMAN_LIKE_MEM_BASE_URL
    prompt: Memory API base URL
    help: Default is https://plugin.human-like.me
    required_for: Custom/self-hosted endpoint
  - name: HUMAN_LIKE_MEM_USER_ID
    prompt: Memory user id
    help: Logical user identifier for memory retrieval
    required_for: Cross-session memory identity
  - name: HUMAN_LIKE_MEM_AGENT_ID
    prompt: Memory agent id
    help: Logical agent identifier for memory partitioning
    required_for: Agent-scoped memory partition
---

# Human-Like Memory

Use this skill when durable user context improves response quality and continuity.

## When To Use

- User asks to remember a preference, profile fact, or decision.
- User asks to continue prior work and context is needed.
- Multi-turn sessions produce summaries worth keeping.

## Quick Reference

```bash
node {baseDir}/scripts/memory.mjs config
node {baseDir}/scripts/memory.mjs recall "<query>"
node {baseDir}/scripts/memory.mjs search "<query>"
node {baseDir}/scripts/memory.mjs save "<user_message>" "<assistant_response>"
node {baseDir}/scripts/memory.mjs save-batch < {baseDir}/examples/messages.json
```

## Procedure

1. Verify configuration with `config` command.
2. Use `recall` or `search` before answering continuity-sensitive prompts.
3. Use `save` for explicit memory-worthy statements.
4. Use `save-batch` only after meaningful multi-turn context.

## Safety Rules

- Do not send passwords, tokens, or private secrets.
- Do not perform hidden/automatic memory writes.
- Treat remote failures as non-fatal and continue with local reasoning.

## Network Disclosure

- API calls occur only when a memory command is executed.
- Endpoint defaults to `https://plugin.human-like.me`.
- Sent fields include query/messages plus `user_id` and `agent_id`.

## Verification

- `node {baseDir}/scripts/memory.mjs config` should show `apiKeyConfigured: true`.
- `recall` or `search` should return JSON with `success: true` when configured.

