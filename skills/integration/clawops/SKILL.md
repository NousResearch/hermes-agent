---
name: clawops
description: Use when the user mentions ClawOps, 爪控中心, OpenClaw 營運中樞, or asks Grace to route work to the OpenClaw multi-agent operations hub.
version: 1.0.0
author: local
license: MIT
metadata:
  hermes:
    tags: [Integration, OpenClaw, ClawOps, operations, multi-agent]
    related_skills: [openclaw-delegation]
---

# ClawOps

## Overview

ClawOps is the operating name for the OpenClaw multi-agent operations hub.
Hermes-Grace is the chief-of-staff entry point; ClawOps is the operations hub
Grace routes to when the user asks for coordinated agent work across course
creation, course marketing, secondhand commerce, and ingrids.app marketing.

Use the name **ClawOps** in Telegram-facing replies. Accept these aliases:

- `ClawOps`
- `爪控中心`
- `OpenClaw 營運中樞`
- `OpenClaw 多代理營運中樞`
- `龍蝦團隊營運中樞`

## When to Use

Use this skill when the user asks Grace to:

- explain ClawOps responsibilities or model assignments
- route a task to ClawOps
- ask which agent should handle a business operations task
- summarize the OpenClaw Agent Team operating model
- work on Hahow course planning, course marketing, secondhand sales, or
  ingrids.app marketing through the multi-agent hub

Do not ask the user to choose between routing and execution commands. Grace is
the chief-of-staff interface: when the user assigns work to Grace, Grace should
hand it to ClawOps, let ClawOps select the agent/model policy, produce the
agent result, and stop before any external side effect that requires approval.

## Grace and ClawOps Collaboration Mechanism

Do not answer ClawOps responsibility/model questions as a generic lookup table
or as generic Strategy/Execution/Review/Ops categories. The intended mechanism
is collaborative:

1. Grace receives the user's natural-language request and preserves business
   context.
2. Grace hands the framed task to ClawOps.
3. ClawOps routes by project, task type, risk level, agent, model policy, and
   approval requirement.
4. The assigned ClawOps agent produces the answer using the ClawOps Registry as
   grounding context.
5. Grace returns the synthesized answer to the user and holds any external
   action behind an approval id.

For responsibility/model questions, the model should still answer as an agent,
but the factual source must be the ClawOps Registry and routing rules. Do not
invent agents or substitute generic categories when registry entries are
available.

## Operating Vocabulary

| Layer | Name | Meaning |
|---|---|---|
| User-facing assistant | Hermes-Grace | Receives Telegram/CLI requests and coordinates next steps. |
| Operations hub | ClawOps | OpenClaw multi-agent operations hub. |
| Agent group | OpenClaw Agent Team | Specialized agents managed through ClawOps. |
| Router | ClawOps Orchestrator | Decides project, task type, risk level, agent, and model policy. |
| Registry | ClawOps Registry | Source of truth for agent roles, primary models, fallback models, and approval rules. |

Preferred Telegram phrasing:

```text
Grace，請找 ClawOps 協助執行。
Grace，請讓 ClawOps 判斷要派哪個 agent。
Grace，把這個任務交給 ClawOps，需要審核的地方先不要發布。
Grace，請整理 ClawOps 的責任分工和使用模型。
```

The live Hermes bridge also exposes:

```text
/clawops <request>
/clawops-run <request>
/clawops-approve <approval_id>
```

Natural-language Telegram messages that mention `ClawOps`, `爪控中心`, or
`OpenClaw 營運中樞` may be rewritten to `/clawops <request>` before normal
agent dispatch.

Use `/clawops` as the single normal entry point. It performs routing, selects
the assigned ClawOps agent, calls the configured model, and returns the agent
result to Grace. `/clawops-run` is only a backwards-compatible alias for older
operator habits.

External actions are separate from model execution. Public posting, outbound
email, data changes, deploys, credential changes, and other side effects must
remain pending until the user explicitly approves them.

When a ClawOps result requires approval, Grace should surface the approval id
and tell the user that execution waits for:

```text
/clawops-approve <approval_id>
```

`/clawops-approve` executes only allowlisted action handlers. Unsupported
actions must remain blocked even after confirmation until a specific connector
has been implemented and allowlisted.

## Source of Truth

When available, answer from these workspace files:

- `/Users/kj/my_agent_team/docs/projects/hub-ops/agent-registry.yaml`
- `/Users/kj/my_agent_team/docs/projects/hub-ops/routing-rules.yaml`
- `/Users/kj/my_agent_team/docs/architecture/hermes-grace-openclaw-operations-hub-v1.md`

If these files are not readable in the current runtime, use the matrix below
and say that the answer is from the ClawOps v1 operating design, not live
runtime verification.

## Agent and Model Matrix

| Agent | Responsibility | Primary model | Fallback model |
|---|---|---|---|
| Hermes-Grace | User entry point, chief-of-staff triage, task framing | `codex` | `gemini-3.1-pro` |
| ClawOps Orchestrator | Routing, result synthesis, final coordination | `codex` | `gemini-3.1-pro` |
| Strategy Agent | Business positioning, offers, audience strategy | `codex` | `gemini-3.1-pro` |
| Course Designer Agent | Hahow course planning, syllabus, lesson design | `codex` | `gemini-3.1-pro` |
| Content Creator Agent | Social posts, scripts, article drafts | `gemini-2.5-flash` | `codex` |
| Marketing Operator Agent | Campaign planning, channel calendars, launches | `gemini-2.5-flash` | `codex` |
| CRM Sales Agent | Lead segmentation, follow-up drafts, sales templates | `gemini-2.5-flash` | `codex` |
| Secondhand Commerce Agent | Resale listings, inventory workflow, buyer replies | `gemini-2.5-flash` | `codex` |
| Ingrids Product Marketing Agent | ingrids.app positioning, SEO, demo scripts | `codex` | `gemini-3.1-pro` |
| Risk Review Agent | Finance, crypto, trading, claim, and publication review | `codex` | `gemini-3.1-pro` |
| Analytics Agent | Weekly reports, metrics, anomaly summaries | `gemini-2.5-flash` | `codex` |
| DevOps Integration Agent | Config, deployment, tests, bridges | `codex` | `gemini-3.1-pro` |

## Routing Rules

Use `codex` for:

- code, config, deployment, repository, credential, or shell work
- business strategy, course design, and ingrids.app product marketing
- approval gates
- financial, crypto, trading, or performance claims
- final synthesis across multiple agents
- correcting conflicts between lightweight-model drafts and source evidence

Use `gemini-3.1-pro` as fallback for:

- business strategy
- course design
- long-form product marketing
- structured planning that still needs approval before publication

Use `gemini-2.5-flash` for:

- high-volume drafts
- summaries
- classification
- CRM templates
- low-risk internal variants

## Safety Boundary

ClawOps v1 should not automatically publish, send outbound sales messages,
commit transactions, deploy code, touch credentials, or make financial claims.

Model calls are allowed for drafting, analysis, routing, and agent execution.
Treat model calls as internal ClawOps work, not public execution. Do not send
secrets, credentials, private customer data, or transaction instructions into
ClawOps model prompts.

Require approval for:

- public content
- Hahow proposal or publication assets
- ingrids.app trading, backtest, performance, or product claims
- outbound CRM/sales messages
- secondhand transaction commitments
- config, repository, deployment, push, destructive, or credential changes

## Answer Pattern

When the user asks what ClawOps is, answer directly:

```text
ClawOps 是 Grace 背後的 OpenClaw 多代理營運中樞。Grace 是入口與總特助；ClawOps 負責判斷任務類型、選擇 agent、套用主要/備援模型、保留審核邊界，並把結果回報給 Grace。
```

When the user asks why Grace gave an older bridge answer, explain:

```text
那是目前 live Hermes ↔ OpenClaw bridge 的已部署狀態；ClawOps 是新的 v1 營運中樞設計。除非 clawops skill 或相關 routing context 已安裝到 Grace 的 live profile，Grace 會優先回答它目前能驗證的 bridge/dry-run 架構。
```

When asked to assign a task, respond as Grace and hand it to ClawOps:

```text
我會交給 ClawOps 執行，外部發布或資料變更會先等你確認。
- project:
- task_type:
- assigned_agent:
- primary_model:
- fallback_model:
- approval_required:
```

Current runtime scope:

- `/clawops` performs routing first, then calls the assigned model for the
  ClawOps agent result. Codex-primary agents use the configured Hermes
  `codex_app_server` runtime; Gemini-primary agents use the configured Gemini
  endpoint. It reports the assigned agent, approval requirement, and actual
  model used when aliases are required.
- `/clawops-run` is a backwards-compatible alias for `/clawops`.
- `/clawops-approve <approval_id>` executes a pending approval record through
  the action executor. The current executor supports allowlisted handlers only;
  unsupported publish/send/update/deploy action types are blocked until their
  connectors are explicitly implemented.
- It does not publish, send messages, trade, deploy, or perform external side
  effects until the user explicitly approves those actions.

Do not say a task has been executed unless a tool call or runtime action
actually completed.
