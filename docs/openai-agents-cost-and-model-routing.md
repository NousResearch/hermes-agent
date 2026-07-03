# OpenAI / GitHub / Hermes Routing Policy for SDK Hardening

Status: local policy, deterministic-gate enforced  
Scope: Hermes OpenAI Agents SDK hardening and pre-NAS migration readiness

## Goal

Keep the pro build streamlined around the trusted stack Scott approved:

```text
Hermes = orchestrator and runtime shell
OpenAI = model/SDK/evals provider
GitHub = repo/issues/PR/CI control plane
Hermes Kanban = local operational task board
```

No additional model/provider ecosystem is introduced just because it exists.

## Provider policy

Allowed primary stack for this work:

- `Hermes` for orchestration, memory/skills, terminal/file tools, gateway, cron, and Kanban.
- `OpenAI` for GPT-5.5 orchestration support, OpenAI Agents SDK lanes, and bounded live smokes.
- `GitHub` for Git remote, issues, PRs, CI, branch protection, and review history.

Blocked / not authorized for this track:

- Chinese-origin models/providers/tools/packages/repos.
- Anthropic migration or Claude routing for this track unless Scott explicitly changes direction.
- Local Ollama/local LLM routing unless Scott explicitly reauthorizes it.
- New hosted PM/services/providers when GitHub + Hermes Kanban can carry the workflow.

## Model roles

| Role | Default |
|---|---|
| Top-level orchestration | Hermes on OpenAI GPT-5.5 |
| SDK subordinate lanes | OpenAI Agents SDK bounded review/execute/verify/architecture lanes |
| Mechanical checks | deterministic Python scripts, git, pytest, JSON schemas |
| PM tracking | repo-local manifest, GitHub Issues when writable target exists, Hermes Kanban board |
| Live smoke | bounded OpenAI SDK calls only when deterministic gates already pass |

## Cost discipline

1. Run deterministic local gates before live SDK calls.
2. Keep SDK lanes bounded by `max_turns`, `max_tokens`, and compact-output constraints.
3. No recurring LLM jobs, cron digests, or autonomous Kanban dispatch without explicit scope.
4. Every live smoke must produce a receipt path and SHA-256.
5. Do not hardcode unverified pricing. If pricing is configured, label it as an estimate.
6. If spend/runaway behavior is suspected, freeze active loops first and attribute token usage before adding automation.

## Tool-surface discipline

The SDK bridge does **not** inherit Hermes terminal/file/browser/memory tools. Hermes stays responsible for side effects. SDK workers remain subordinate proof/review/drafting lanes until tool guardrails, approval boundaries, and receipts are designed.

## Promotion rule

A capability can move from local-only to external/runtime only when all are true:

- local quality gate passes;
- proof bundle generated and hashed;
- project tracking manifest updated;
- source freshness manifest current for SDK assumptions touched;
- explicit scope exists for the external side effect;
- rollback/blocked state is recorded.

## Current external blockers

- GitHub origin `NousResearch/hermes-agent` is visible but current account has `viewerPermission=READ`, so upstream issue/PR creation needs a writable fork/target confirmation.
- `projectmanager` profile exists but has no configured model in the current profile list; Kanban cards should remain blocked unless profile routing is configured or we execute them directly in this governed session.
- Gateway/fresh-session proof requires restart and `/new` before claiming Telegram runtime has loaded the latest SDK retry code.
