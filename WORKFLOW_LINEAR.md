# Linear as a Hermes Interface

**Status:** Canonical workflow contract v0.1  
**Applies to:** Hermes gateway `Platform.LINEAR`, the DEC `Hermes Linear Interface` project, and any future Linear-backed Hermes control plane  
**Last updated:** 2026-04-28

## Doctrine

Linear is durable intent. It has three distinct lanes:

1. **Inbox/chat** — one-off conversations that should behave like a normal Hermes session rooted in a Linear issue. No Kanban task is created by default.
2. **Execution** — shaped work that should become durable Kanban execution with retries, decomposition, worker profiles, and terminal event mapping.
3. **Status/control** — a persistent issue for asking Hermes about system/Kanban/Linear state.

A Linear issue is a control surface. Its state, labels, description, and comments are not just project-management metadata; together they form the prompt envelope, permission boundary, execution state, audit log, and human review loop for Hermes.

The gateway must optimize for three things:

1. **Durability** — the user's intent survives context loss, gateway restarts, model switches, and long-running work.
2. **Steerability** — a human can redirect work without losing the thread or spawning a parallel ghost task.
3. **Safety** — state, labels, and explicit permissions constrain the agent more strongly than any ambiguous natural-language request.

If a request is too vague to survive being read tomorrow by a different agent run, it is not ready to be an autonomous Linear task.

## Normative language

- **MUST** means required for a conforming Linear-Hermes interface.
- **SHOULD** means the default unless there is a clear reason not to.
- **MAY** means optional extension.
- **Human** means Anton or another explicitly authorized Linear user.
- **Hermes** means the active Hermes gateway agent, not a fictional second operator.

## Objects

| Object | Contract |
|---|---|
| Issue | Durable task container and Hermes session root. One issue equals one long-lived Hermes thread. |
| Issue description | Initial prompt plus durable context. It should be stable enough for a fresh session to start from it. |
| Comment | Steering input. A comment is either a new turn, a slash command, or a structured control directive. |
| Workpad | Hermes' canonical output block. It records status, result, verification, changed files, safety notes, and next action. |
| State | Execution gate. State decides whether Hermes should ignore, run, wait for review, or stop. |
| Labels | Routing and policy defaults. Labels define route, model preference, risk, permission, domain, and surface membership. |
| Session | Hermes gateway session keyed by the Linear issue id, with the issue identifier as thread id. |

## Issue lifecycle

### 1. Inbox / chat

A human creates an issue specifically for a one-off conversation.

- State: `Inbox` or another configured inbox state.
- Hermes behavior: route directly to the Linear issue's Hermes session; do **not** create a Kanban task by default.
- State mutation: leave the issue state unchanged unless the response contains an explicit Linear state directive or Anton asks to turn it into work.
- Use case: quick questions, status checks, "think with me" threads, and small one-off requests where durable execution machinery is unnecessary.

### 2. Capture

A human creates or receives an issue.

- State: `Triage`, `Backlog`, or equivalent non-active state.
- Hermes behavior: **ignore** except for explicitly addressed comments if the adapter is configured to process them.
- Required human action: clarify outcome, constraints, permission, and done criteria.

### 3. Shape

The issue is turned from a note into an executable control surface.

The description SHOULD contain:

```markdown
## Outcome
What must be true when this is done.

## Context
Why this matters, relevant files, links, prior decisions, and product boundaries.

## Constraints
Hard constraints: no token leakage, no prod deploy, do not restart gateway, preserve API shape, etc.

## Done means
Concrete acceptance criteria and verification requirements.

## Initial instruction
The first thing Hermes should do when the issue becomes active.
```

If the issue cannot answer these questions, Hermes SHOULD plan or ask a tight clarification instead of improvising:

- What is the outcome?
- What constraints matter?
- What counts as done?
- Should Hermes act, plan, review, research, or wait?
- What state should the issue move to afterward?

### 4. Activate

A human moves the issue into the executable queue.

Canonical start state:

- `Todo`

Hermes behavior:

- `Backlog` is inert. Hermes MUST NOT start work from Backlog.
- On the first `Todo` poll, Hermes receives the issue description as the initial turn.
- When processing begins, Hermes moves the issue to `Working`.
- Later allowed comments become follow-up turns in the same issue session only when the state machine permits them.
- Hermes MUST treat the issue title, state, team, project, URL, and labels as part of the prompt envelope. Project is context, not an activation boundary.

### 5. Execute

Hermes performs the requested route within the issue's permission boundary.

During execution Hermes MUST:

- Preserve the Linear issue as the audit trail.
- Use the newest explicit steering comment as the immediate instruction.
- Treat older issue description constraints as still binding unless explicitly revoked by an authorized human.
- Avoid external side effects unless the issue grants them.
- Report uncertainty rather than silently widening scope.

### 6. Review

Hermes returns a workpad and the issue enters human review.

Canonical review state:

- `Ready for Review`

Hermes behavior:

- Successful runs SHOULD end in `Ready for Review`.
- New human comments in `Ready for Review` are review notes, not automatic execution. A human moves the issue back to `Todo` to run another turn.
- Hermes-authored workpad comments MUST NOT be re-ingested as user input.
- If Hermes needs Anton's response before it can proceed, it MUST ask the question and move the issue to `Blocked`.

### 7. Rework

A human requests changes.

- State: move back to `Todo` when the next agent turn should run.
- Hermes behavior: run another turn in the same session, preserving the workpad history and prior constraints.
- The newest comment wins for immediate route/model preference, but safety constraints accumulate.

### 8. Complete or stop

Terminal states:

- `Done`
- `Canceled`
- `Duplicate`

Hermes behavior:

- MUST NOT start new runs from terminal states.
- MAY respond only if an authorized human explicitly revives the issue by moving it back to an active state.

## State machine

| State family | Example Linear state | Hermes behavior | Allowed transition |
|---|---|---|---|
| Inbox/chat | `Inbox` | Direct Hermes session rooted in the issue. No Kanban task by default. | Leave unchanged; human may move to `Todo` to execute. |
| Intake | `Triage` | Do not run automatically. | Human shapes issue. |
| Dormant | `Backlog` | Preserve durable intent; no run. | Human moves to `Todo`. |
| Start gate | `Todo` | Run initial issue prompt once; process queued human instruction. | Gateway moves to `Working` on processing start. |
| Active | `Working` | Agent is currently processing. | Gateway to `Ready for Review` or `Blocked`. |
| Blocked | `Blocked` | Wait for Anton's answer. A new authorized comment may resume the same issue and move it to `Working`. | Human answers; gateway resumes. |
| Review | `Ready for Review` | Wait. Do not auto-run comments as work. | Human to `Todo`, `Done`, or `Canceled`. |
| Terminal | `Done`, `Canceled`, `Duplicate` | Ignore. | Human revives by moving to `Todo`. |

State is an execution gate, not decoration. If state and comment conflict, state wins unless the comment is an explicit state-change command from an authorized human.

## Comment steering contract

Comments are for steering, not ambient chatter. A good comment changes the next run's behavior.

### Structured comment header

Use this shape when precision matters:

```text
Route: build | research | review | plan | ops | wait
Model preference: default | fast | deep | claude | codex | gemini | local
Risk: low | medium | high | external | prod
Permission: read-only | edit local branch only | edit this repo only | external write allowed | deploy allowed
State after: Ready for Review | Blocked | Done | leave unchanged

Instruction:
<the actual request>

Done means:
- <acceptance criterion>
- <verification command or evidence>
```

Field names are case-insensitive. Free-form comments are allowed, but Hermes MUST infer the safest route and permission when fields are absent.

### Precedence

When multiple sources conflict, use this order:

1. Safety deny rules and terminal states.
2. Explicit comment fields in the newest authorized human comment.
3. Restrictive labels on the issue.
4. Issue description constraints.
5. Project/team defaults.
6. Gateway defaults.

Restrictions only ratchet tighter unless an authorized human clearly relaxes them. A new `Route: build` comment does not erase an older `Do not restart the gateway` constraint.

### Slash commands

Linear comments MAY contain gateway slash commands when the adapter routes them:

- `/status` — ask for current session status.
- `/stop` — stop the active run.
- `/queue <text>` — run text after the current turn.
- `/steer <text>` — intra-turn guidance if busy steering is supported.
- `/new` or `/reset` — reset this issue's Hermes session.
- `/approve` / `/deny` — approval flow for gated tool actions, if available.

Normal comments SHOULD be turn-boundary input. If a comment arrives while Hermes is already running, the gateway MUST either steer or queue it according to configured busy-input mode; it MUST NOT silently drop it.

### Good steering examples

```text
Route: plan
Permission: read-only

Start with a minimal implementation plan. Do not edit code yet.
```

```text
Route: build
Model preference: codex
Risk: low
Permission: edit local Hermes branch only

Implement the Linear workpad ignore sentinel. Add a regression test. Do not restart the live gateway.

Done means:
- changed files listed
- exact pytest command reported
- no token values in output
```

```text
Route: review
Permission: read-only

Inspect the Linear adapter and tell me the next three hardening steps before I use this daily.
```

Bad comment:

```text
Can you handle this?
```

That is not a control surface. It forces Hermes to guess route, scope, permission, and done criteria.

## Workpad contract

Hermes' canonical output to Linear is a workpad comment. It MUST start with this exact heading:

```markdown
## Hermes Workpad
```

The current Linear adapter ignores comments whose body starts with `## Hermes Workpad`; this prevents Hermes from eating its own output.

### Required workpad format

```markdown
## Hermes Workpad

```yaml
contract: linear-hermes/v1
status: human_review        # running | human_review | blocked | failed | done
issue: DEC-6
route: build                # plan | research | build | review | ops | wait
model: provider/model-or-default
risk: low                   # low | medium | high | external | prod
permission_used: edit-local # read-only | edit-local | external-write | deploy
state_after: Ready for Review
```

### Outcome
What changed or what was decided.

### Work performed
- Concrete action 1
- Concrete action 2

### Changed files
- `path/to/file` — why it changed
- None

### Verification
- `command run`
- Result: PASS / FAIL / NOT RUN
- Evidence or failure reason

### Safety notes
- Secrets exposed: no
- External writes: Linear comment only
- Live services restarted: no

### Next action
The single next human or Hermes action.
```

If Hermes did not run verification, it MUST say `NOT RUN` and why. If work is blocked, it MUST state the blocker and the smallest unblock action.

### Workpad rules

- One workpad per substantive Hermes run.
- No raw transcripts.
- No secret values.
- No invented test results.
- No claims of deployment, delivery, or external writes without a verifiable handle.
- If output is long, put summary in the workpad and write files/links for the rest.

## Label contract

Labels are policy defaults. Comments can route a specific turn, but labels define the issue's standing posture.

### Surface labels

| Label | Meaning |
|---|---|
| `hermes` | Issue is eligible for Hermes handling. |
| `gateway` | Touches Hermes gateway behavior or adapter contracts. |
| `symphony` | Part of the Linear control-plane / Symphony orchestration layer. |

### Route labels

| Label | Meaning | Default permission |
|---|---|---|
| `route:plan` | Produce plan/design; no implementation. | read-only |
| `route:research` | Gather external/internal evidence and cite it. | read-only |
| `route:build` | Modify local repo files and verify. | requires edit permission |
| `route:review` | Adversarial audit of existing artifact/code. | read-only |
| `route:ops` | Services, deploys, credentials, restarts, admin surfaces. | explicit human approval |
| `route:wait` | Do not act until a human comments. | none |

### Model preference labels

| Label | Meaning |
|---|---|
| `model:default` | Use gateway default model/provider. |
| `model:fast` | Prefer low-latency, low-cost routing for simple work. |
| `model:deep` | Prefer highest reasoning budget for architecture/review. |
| `model:claude` | Prefer Claude-family coding/reasoning model if available. |
| `model:codex` | Prefer Codex/OpenAI coding route if available. |
| `model:gemini` | Prefer Gemini route if available. |
| `model:local` | Prefer local/offline model route if configured. |

Model labels are preferences, not guarantees. If unavailable, Hermes MUST fall back visibly and record the actual model in the workpad when known.

### Risk labels

| Label | Meaning | Required behavior |
|---|---|---|
| `risk:low` | Local, reversible, no external effect. | Proceed within permission. |
| `risk:medium` | Could break local dev flow or user-facing behavior. | Verify; report changed files. |
| `risk:high` | Large refactor, destructive action, sensitive data, or ambiguous blast radius. | Plan first; require review. |
| `risk:external` | Writes outside local filesystem or Linear. | Explicit human authorization. |
| `risk:prod` | Production deploy/restart/data mutation. | Explicit approval immediately before action. |

### Permission labels

| Label | Meaning |
|---|---|
| `perm:read-only` | No file writes or external writes. |
| `perm:edit-local` | May edit local checkout/worktree only. |
| `perm:edit-branch` | May create branch/commit locally. |
| `perm:external-write` | May write to approved external system named in issue. |
| `perm:deploy` | May deploy/restart only with explicit final approval. |

If multiple permission labels exist, the most restrictive one wins unless a newer authorized comment clearly changes it.

## Safety rules

### Default stance

Read-only is the default. Hermes may only widen scope when Linear provides a clear permission boundary.

### Secrets

- Never paste API keys, tokens, passwords, cookies, private keys, or OAuth credentials into Linear.
- If a comment contains token-like text, Hermes MUST treat it as contaminated input: do not repeat it; tell the human to move it to the secret store.
- Env var names are allowed; values are not.

### External side effects

Hermes MUST NOT perform these without explicit permission in the issue or newest authorized comment:

- Send email, Slack, Discord, Telegram, X/Twitter, or public comments outside the originating Linear issue.
- Deploy, restart live services, rotate credentials, mutate databases, or change DNS.
- Merge PRs, push branches, publish packages, or alter production config.
- Spend money, order goods, book rides, or perform account actions.

### Live gateway changes

For Hermes gateway work:

- Do not restart the live gateway unless the issue explicitly grants it.
- Prefer local branch/worktree verification first.
- If a config or bootstrap-file change requires restart, say so in the workpad rather than silently restarting.

### Filesystem safety

- Prefer reversible edits.
- Use trash rather than destructive deletes when possible.
- For code changes, isolate work in the current branch or an explicit worktree.
- Report every changed file.

### Tool and model honesty

Hermes MUST NOT claim:

- tests passed if they were not run;
- deployment succeeded without a URL/status/commit/release id;
- a model/provider was used unless known;
- a Linear state changed unless the API call succeeded or the human changed it.

## Current adapter compatibility

The current Linear adapter already implements the transport core:

- Polls configured Linear issues by team/project/state.
- Treats an issue as an initial Hermes message when first seen in an active state.
- Treats allowed comments as follow-up messages.
- Sends Hermes replies as Linear comments.
- Uses `SessionSource.platform = linear`, `chat_id = issue UUID`, and `thread_id = issue identifier`.
- Tracks seen issue/comment ids in `~/.hermes/linear/state.json`.
- Ignores Hermes workpad comments starting with `## Hermes Workpad`.

The contract above defines the stricter operating layer that routing, labels, state mutation, and future automation should converge on.

## Operator checklist

Before activating an issue:

- [ ] Outcome is explicit.
- [ ] Constraints include safety boundaries.
- [ ] Done criteria are testable.
- [ ] Route label or `Route:` field exists.
- [ ] Risk and permission are clear.
- [ ] The next desired state is clear.

Before marking done:

- [ ] Workpad exists.
- [ ] Changed files are listed or `None`.
- [ ] Verification is present and honest.
- [ ] External side effects are recorded with handles.
- [ ] No secrets are present in the issue or workpad.
