# Agent OS Core

AGENT_OS_RULE_VERSION: AGENT-OS-V1-20260831-7F3C72A1

Use this rule when the task is a repository implementation or rollout task that
must preserve accepted work and avoid rediscovery.

## Core Laws

1. Start with `GOAL -> CAPABILITY_ID -> PRIOR_WORK_LOCATE -> ACCEPTED_BASELINE -> FIRST_UNPROVEN_BOUNDARY -> INVALIDATOR_CHECK`.
2. Reuse accepted work by default. If no explicit invalidator fired, resume from `FIRST_UNPROVEN_BOUNDARY` and do not redo broad discovery or replanning.
3. Keep one canonical writer. Read-only scouts may gather evidence, but they do not become a second writer.
4. Use fresh machine evidence as PASS/FAIL authority. RAG and old chat are locators only.
5. Tiny tasks should finish with minimal ceremony. Do not create plan overhead when direct execution or a single focused check is enough.

## Required Future Debt Inventory Fields

- `CAPABILITY_ID`
- `STATE`
- `DEPS`
- `RISK`
- `WEIGHT`
- `FIRST_UNPROVEN_BOUNDARY`
- `BASE_HEAD`
- `BASE_TREE`
- `CHANGED_PATHS`
- `SEMANTIC_OWNER`
- `RUNTIME_IMPACT`
- `RELEASE_IMPACT`
- `BLOCK_LIVE`
- `MERGE_PRIORITY`

## Required States

- `DONE`
- `READY`
- `RUNNING`
- `BLOCKED`
- `FUTURE`
- `INVALIDATED`

## Required Outputs

- `READY_FRONTIER`
- `CRITICAL_PATH`
- `PARALLEL_SAFE_FRONTIER`

## Runtime Rules

- `SignalOps` live critical-path work always outranks non-blocking debt.
- `BROKER_UNKNOWN` fails closed: no replay, no resend, no replacement command.
- Do not weaken tests or safety guards to manufacture PASS.
- When asked for the Agent OS rule version, return the exact `AGENT_OS_RULE_VERSION` value above.
