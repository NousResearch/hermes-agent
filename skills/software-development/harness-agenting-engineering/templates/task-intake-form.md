# Harness Task Intake Form

> Fill this before submitting a non-trivial task to Hermes / an LLM / a coding agent. Leave fields blank only when genuinely not applicable.

## 0. Submission mode

- Task title:
- Workspace / repo:
- Desired mode:
  - [ ] Analysis only
  - [ ] Plan only
  - [ ] Implement changes
  - [ ] Review existing changes
  - [ ] Debug / root cause
  - [ ] Integrate external tool via Plugin / MCP / Command
- Permission level:
  - [ ] Read-only
  - [ ] May write files
  - [ ] May run tests/builds
  - [ ] May install dependencies
  - [ ] May touch config
  - [ ] May call external APIs
  - [ ] Needs confirmation before side effects

## 1. Spec — what counts as correct?

- Problem / request:
- Business or user outcome:
- Acceptance criteria:
  1.
  2.
  3.
- Non-goals / out of scope:
- Constraints:
  - Time:
  - Compatibility:
  - Performance:
  - Cost / token budget:
  - Security / privacy:
- Known examples / references:
- What should not change:

## 2. Context routing — what must the agent read first?

- Relevant docs / rules:
  - [ ] AGENTS.md
  - [ ] CONTRIBUTING.md
  - [ ] docs/CONTRACTS.md or equivalent
  - [ ] RFC / design doc:
  - [ ] TESTING.md / test docs:
- Relevant files / modules:
- Relevant prior sessions / memories / skills:
- External docs or APIs needed:
- MCP servers / tools that should be available:
- Tools that should be disabled or avoided:

## 3. Risk surface

Check anything touched by the task:

- [ ] Secrets / API keys / credentials
- [ ] Auth / OAuth / license / payment
- [ ] Shell commands / destructive filesystem ops
- [ ] Network calls / webhooks / external APIs
- [ ] Plugin / MCP / provider integration
- [ ] Browser automation / cookies / sessions
- [ ] Database / persistent state
- [ ] User data / PII
- [ ] UI / frontend state / replay / streaming
- [ ] Build / packaging / release artifacts
- [ ] No special risk identified

Required negative checks:
-

## 4. Extension choice

If this task adds a capability, choose one:

- [ ] Built-in code change — because:
- [ ] Plugin — because it is a reusable Hermes extension or provider/tool/hook package.
- [ ] MCP — because an external server/tool already exists or should stay outside Hermes core.
- [ ] Command / script — because this is a repeatable operator action or quality gate.
- [ ] Skill — because this is a reusable multi-step workflow.
- [ ] Rule / AGENTS.md — because this is a short always-on constraint.
- [ ] Contract doc / tests — because this changes a public invariant.

Chosen extension surface:
Reason:
Rollback plan:

## 5. Implementation plan

- Smallest useful change:
- Files likely touched:
- Tests likely touched:
- Dependencies to add, if any:
- Migration / compatibility concerns:
- Expected output artifact:

## 6. Verification evidence required before done

- Automated tests:
- Lint / static checks:
- Manual verification:
- UI evidence / screenshots / browser notes:
- Security / negative tests:
- Performance / token budget evidence:
- Logs / diagnostics to include:

## 7. Retention decision

After completion, should Hermes update any of these?

- [ ] Skill
- [ ] AGENTS.md / rule
- [ ] Contract doc
- [ ] Test fixture / harness gate
- [ ] MCP / plugin config docs
- [ ] Durable memory, only if stable and non-secret
- [ ] No retention needed

Notes:

## 8. Final prompt to submit

Use this condensed prompt after filling the form:

```text
Please execute this task under Harness / Agenting Engineering discipline.

Task: <title>
Workspace/repo: <workspace>
Mode and permissions: <mode + permission level>
Spec: <problem, acceptance criteria, non-goals, constraints>
Context routing: <docs/files/skills/MCP/tools to inspect first>
Risk surface: <checked risks + required negative checks>
Extension choice: <Plugin/MCP/Command/Skill/Rule/Contract/Built-in + reason>
Implementation expectation: <smallest useful change + likely files>
Verification required: <tests, lint, manual evidence, security checks>
Retention: <skill/rule/contract/memory decision>

Do not claim completion without evidence. If required context is missing and cannot be retrieved, ask a targeted clarification before implementation.
```
