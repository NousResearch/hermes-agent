# Hermes Orchestrator Roadmap — Phase 7+

> **For Hermes:** Use the controller/worker/reviewer loop for each phase. Write a task packet, create an isolated worktree, delegate implementation to Claude Code only after the packet is clear, run targeted tests, run independent spec + quality review, import to main only after approval, and commit each phase separately.

**Goal:** Continue evolving Hermes into Woo's front-desk / concierge / butler orchestrator: the main Hermes remains accountable and conversational while focused task/workers handle long or specialized work behind the scenes.

**Current baseline:**

```text
eddabc597 feat(agent): add orchestration status observatory
```

**Current built substrate:**

```text
Phase 1  /busy integrated foundation
Phase 2  structured PendingTurnItem / PendingTurnQueue
Phase 2.1 CLI pending-input producer/drain ordering hardening
Phase 3  Focused Task Registry substrate
Phase 4  Worker Lane substrate
Phase 5  Conservative Follow-up Router
Phase 6  Orchestration Status Observatory
```

**North-star UX:**

```text
Woo speaks naturally in Telegram/CLI
→ Hermes identifies task or follow-up
→ Hermes records task/follow-up state
→ Worker lanes execute focused work
→ Woo can ask what is running
→ Hermes shows /tasks /agents style status when useful
→ Worker result returns to Hermes
→ Hermes reviews, applies late follow-ups, and synthesizes final answer
```

---

## 0. Product Principles for Remaining Phases

### 0.1 Main Hermes remains the front desk

Hermes should not become a thin command launcher. It should:

- understand the user's intent and context;
- decide whether direct answer, foreground work, or background worker is appropriate;
- keep the user relationship and accountability;
- inspect/review worker results before delivery;
- incorporate late follow-ups naturally into the final answer.

### 0.2 Commands are useful, but natural language is primary

`/tasks` and `/agents` are useful observability shortcuts, especially for debugging and confidence. But the core UX should also support:

```text
지금 뭐 하고 있어?
돌고 있는 작업 있어?
이것도 방금 작업에 추가해줘
새 작업으로 따로 해줘
그건 취소하지 말고 기존 작업에 붙여
```

### 0.3 Conservative first, automation later

For routing and cancellation:

- status query is safe;
- append-only follow-up is usually safe;
- correction note is safe if append-only;
- new task is safe only when explicit;
- cooperative cancel needs an unambiguous target;
- force kill requires a separate phase and confirmation;
- LLM classifier should be advisory only until deterministic gates are strong.

### 0.4 No fake availability

Do not claim Hermes is always available if the current implementation still blocks the active parent turn. Real availability requires gateway/runtime wiring and worker lanes that continue independently of the foreground response.

---

## Phase 7 — Runtime Orchestration Context + Minimal Status Commands

**Goal:** Provide a session/runtime place where CLI/gateway can hold or access `TaskRegistry` and `WorkerLaneRegistry`, then expose minimal `/tasks` and `/agents` status surfaces using Phase 6 formatters.

**Why now:** Phase 6 intentionally did not wire `/tasks` and `/agents` because there was no live registry for commands to read. This phase creates the smallest safe runtime context so observability becomes real rather than a formatter library only.

**Likely new/modified files:**

```text
agent/orchestration_runtime.py          # if a small runtime container is useful
agent/orchestration_status.py           # only if small adapter hooks are needed
hermes_cli/commands.py                  # register /tasks and /agents if not present
cli.py                                  # minimal command handlers if runtime context exists
gateway/run.py                          # optional minimal command handler, behind safe empty-state behavior
tests/agent/test_orchestration_runtime.py
tests/cli/test_orchestration_status_commands.py
tests/gateway/test_orchestration_status_commands.py
```

**Do:**

- Create a small `OrchestrationRuntime` or equivalent container if needed:

```python
@dataclass
class OrchestrationRuntime:
    task_registry: TaskRegistry
    worker_registry: WorkerLaneRegistry
```

- Add a safe empty state for `/tasks` and `/agents`:

```text
No active tasks are currently registered.
No active workers are currently registered.
```

- Use `format_tasks(...)` and `format_agents(...)` from Phase 6.
- Keep runtime in-memory only unless an existing session store integration is trivial and tested.
- Make command handlers read-only.

**Do not:**

- start workers from `/tasks` or `/agents`;
- implement gateway natural-language auto-routing yet;
- create durable DB;
- add force cancel;
- expose a complex task management UI.

**Acceptance:**

- `/tasks` and `/agents` are discoverable in command registry/help if wired.
- Empty state is graceful.
- A test-injected runtime with tasks/workers formats correctly.
- No production behavior changes except read-only status commands.
- Existing Phase 1-6 targeted suites remain green.

---

## Phase 8 — Gateway Minimal Natural-Language Status Routing

**Goal:** Let Telegram/gateway answer safe status questions such as “지금 뭐 하고 있어?” by calling the Phase 6 status formatter against the Phase 7 runtime.

**Why:** This is the first step toward Manus-like conversational smoothness: the user should not need to remember `/tasks`.

**Likely files:**

```text
gateway/run.py
gateway/session.py
agent/followup_router.py or agent/orchestration_status.py
tests/gateway/test_orchestration_status_routing.py
```

**Do:**

- Use `looks_like_orchestration_status_query(text)` as a deterministic predicate.
- Only answer status; do not mutate tasks/workers.
- Restrict to safe, obvious status-query texts.
- If no runtime is available, return graceful empty state.
- Keep behind a config flag if the gateway path is high risk:

```yaml
orchestration:
  status_queries_enabled: false   # default initially if needed
```

**Do not:**

- auto-route append/correction/cancel/new task yet;
- run an LLM classifier;
- change normal chat behavior broadly;
- hijack unrelated questions like calorie estimates or pharma research.

**Acceptance:**

- “지금 뭐 하고 있어?” returns overview when active tasks/workers exist.
- Ordinary unrelated messages still go to normal agent conversation.
- Slash commands remain boundaries.
- Tests cover Korean and English status queries, empty state, and unrelated messages.

---

## Phase 9 — Append-Only Follow-up Attachment in Gateway

**Goal:** When exactly one active task exists in the chat/session, safe append/correction follow-ups can be recorded to the task without interrupting or replacing worker execution.

**Why:** This is the core Manus-like smoothness: Woo can keep adding requirements while work is running.

**Likely files:**

```text
gateway/run.py
gateway/platforms/base.py
agent/followup_router.py
agent/task_registry.py
tests/gateway/test_followup_attachment.py
```

**Do:**

- Convert incoming message to `PendingTurnItem`.
- Use `FollowupRouter` for deterministic classification.
- For a single active task:
  - append plain text follow-up;
  - record correction as append-only note/follow-up;
  - defer command/media/ambiguous cases.
- Preserve original message order and metadata.
- Send a concise acknowledgement only when useful:

```text
방금 작업에 추가해둘게요.
```

**Do not:**

- mutate worker prompt destructively;
- force re-run worker;
- cancel or create new task unless explicit;
- use LLM classifier yet.

**Acceptance:**

- “이것도 포함해줘” attaches to single active task.
- “아니 그건 빼고…” is captured as correction note/follow-up.
- Multiple active tasks cause conservative defer/ask, not guessing.
- Commands/media are not swallowed.

---

## Phase 10 — Worker Result Synthesis Contract

**Goal:** When a worker completes, Hermes main synthesizes the final user-facing answer from original task + follow-ups + worker result + artifacts + errors.

**Why:** The user should receive a coherent final result, not raw worker output. This is the “concierge” accountability layer.

**Likely new file:**

```text
agent/orchestration_synthesis.py
```

**Inputs:**

```text
FocusedTask
Pending follow-ups / notes
WorkerResult
Artifacts
Original user goal
Review/test metadata if available
```

**Do:**

- Define a `SynthesisPacket` / `SynthesisResult` data structure.
- Mark which follow-ups were applied, deferred, or require user decision.
- Include artifact verification status.
- Produce concise Telegram-friendly final answer blocks.
- Support failure synthesis:

```text
작업 실패 / 원인 / 다음 선택지 / 재시도 가능 여부
```

**Do not:**

- blindly forward worker output;
- auto-publish artifacts without verification;
- run new workers automatically on failure unless explicitly scoped.

**Acceptance:**

- Worker raw result is transformed into final summary.
- Late follow-ups appear in the final synthesis.
- Artifact paths are verified before mention.
- Error result produces actionable recovery options.

---

## Phase 11 — Background Worker Dispatch: Claude Code Lane MVP

**Goal:** Turn the worker-lane substrate into a practical Claude Code background worker for repo-local implementation tasks.

**Why:** This is where Hermes becomes materially closer to Manus-like orchestration for code work: main Hermes can acknowledge, spawn work, and later synthesize/review.

**Likely files:**

```text
agent/worker_lanes.py
agent/claude_code_lane.py              # likely new
cli.py or gateway/run.py               # only minimal dispatch hook if needed
tests/agent/test_claude_code_lane.py
```

**Do:**

- Implement a lane that launches Claude Code print-mode in an isolated worktree or specified cwd.
- Store worker id, status, stdout/stderr path, result summary, exit code.
- Ensure no direct commit/import by the worker unless explicitly requested.
- Support cooperative cancel if process still running.
- Require task packet / goal / workdir metadata.

**Do not:**

- make all delegation background by default;
- expose unsafe arbitrary shell process killing;
- auto-import worker diffs without review;
- skip task packet/review gates for Hermes repo work.

**Acceptance:**

- A test/dummy Claude Code command can run through the lane.
- Status is visible through Phase 6/7 observatory.
- Completion produces a worker result that Phase 10 can synthesize.
- Cancellation is cooperative and safe.

---

## Phase 12 — Review/Import Automation for Code Workers

**Goal:** Codify the pattern used in Phases 1-6: worker diff → targeted tests → spec review → quality review → import → commit.

**Why:** Hermes should be able to manage code-work lifecycle reliably without re-inventing the manual sequence each time.

**Likely new file:**

```text
agent/code_workflow.py
```

**Do:**

- Represent a code-work phase packet.
- Track diff artifact path, tests run, review verdicts, import status, commit sha.
- Provide helper functions or templates for:
  - saving diff;
  - running targeted tests;
  - recording reviewer results;
  - generating final report.

**Do not:**

- auto-merge without approvals;
- hide test failures;
- push to upstream remote automatically.

**Acceptance:**

- Phase-style code work can be represented as a structured lifecycle record.
- Failed review blocks import.
- Accepted review enables import/commit with traceable metadata.

---

## Phase 13 — Conservative New Task Creation from Natural Language

**Goal:** Let explicit natural-language “new task” requests create a focused task record through the gateway/CLI runtime.

**Examples:**

```text
이건 새 작업으로 따로 해줘
별도 리서치로 돌려줘
new task: analyze CELMoD market
```

**Do:**

- Use deterministic `FollowupRouter` new-task classification.
- Create `FocusedTask` with origin/session/user goal.
- Return a concise acknowledgement and task id/title.
- No automatic worker dispatch unless configured/explicit.

**Do not:**

- infer new task from ambiguous topic shift;
- auto-fanout to many workers;
- launch costly workers without a policy gate.

**Acceptance:**

- Explicit new task creates task.
- Ambiguous text asks/defers.
- `/tasks` shows the new task.

---

## Phase 14 — Cooperative Cancel UX

**Goal:** Support safe cancellation requests for a single unambiguous task/worker.

**Do:**

- Route “멈춰/취소/cancel” only when target is unambiguous.
- Mark task `cancelled` or `blocked/cancel_requested` according to worker capability.
- Call `WorkerLane.cancel(worker_id)` for cooperative cancellation.
- Report whether cancellation was requested, completed, or not possible.

**Do not:**

- force kill processes;
- cancel when multiple active tasks exist without asking;
- treat negated phrases as cancel:

```text
취소하지 마
don't stop
not cancel
```

**Acceptance:**

- Single active worker receives cooperative cancel.
- Multiple active workers asks which one.
- Negated cancel phrases are safe.

---

## Phase 15 — Optional LLM-Assisted Ambiguity Classifier

**Goal:** Add an advisory classifier only for ambiguous follow-ups that deterministic rules refuse to route.

**Do:**

- Keep deterministic router as the safety gate.
- LLM can suggest `append/status/new_task/ambiguous`, but destructive actions require deterministic confirmation or user confirmation.
- Include confidence and rationale.
- Log classifier decisions for audit.

**Do not:**

- let LLM classifier cancel/force kill/create costly workers alone;
- replace deterministic hard boundaries for commands/media;
- silently route ambiguous messages.

**Acceptance:**

- Ambiguous text can produce a suggested route.
- Low confidence asks user.
- Deterministic safety rules override LLM.

---

## Phase 16 — Durable Runtime Persistence

**Goal:** Persist active tasks/workers/routing state across gateway restarts when the runtime is mature enough to need it.

**Do:**

- Start with append-only JSONL or SQLite only after schema is stable.
- Persist task registry and worker metadata, not secrets/raw payloads.
- On restart, mark unknown in-flight workers explicitly:

```text
recovered | unknown | failed | needs_user_review
```

**Do not:**

- serialize `PendingTurnItem.raw`;
- persist credentials or command outputs with secrets;
- promise live process recovery if only metadata is recovered.

**Acceptance:**

- Restart shows prior task state clearly.
- In-flight non-recoverable workers are marked unknown/needs-review.
- No raw/secrets persisted.

---

## Phase 17 — Ralph Runtime Naming Layer

**Goal:** Only after runtime, routing, observability, synthesis, and persistence are coherent, introduce Ralph as a focused worker/task unit abstraction if still useful.

**Definition candidate:**

```text
Ralph = a named focused-agent unit bound to one FocusedTask, one WorkerLane handle, follow-up routing, status visibility, and synthesis contract.
```

**Do:**

- Make Ralph a user-comprehensible abstraction, not a premature class name.
- Keep Hermes as accountable orchestrator above Ralph.
- Support task-scoped personality/tooling only if needed.

**Do not:**

- introduce Ralph before it simplifies user experience;
- make Ralph compete with Hermes main;
- hide worker provenance.

**Acceptance:**

- User can understand “Ralph is working on X” and see status.
- Hermes can synthesize Ralph output into final answer.
- No loss of accountability.

---

## Recommended Execution Order

Short version:

```text
7. Runtime context + /tasks /agents minimal status
8. Gateway natural-language status query
9. Append-only follow-up attachment
10. Worker result synthesis contract
11. Claude Code background worker lane MVP
12. Review/import automation for code workers
13. Explicit new-task creation
14. Cooperative cancel UX
15. Optional LLM ambiguity classifier
16. Durable runtime persistence
17. Ralph runtime naming layer, only if it now simplifies UX
```

Practical checkpoint groups:

```text
Visibility block:
  Phase 7-8

Smooth conversation block:
  Phase 9-10

Real worker execution block:
  Phase 11-12

Control/recovery block:
  Phase 13-16

Product naming/runtime block:
  Phase 17
```

---

## Immediate Next Step

Do **not** jump directly to Ralph runtime.

Next task packet should be:

```text
Claude Code Task Packet — Phase 7 Runtime Orchestration Context + Minimal Status Commands
```

This phase should be small and concrete:

```text
- Create/inject a minimal runtime context for TaskRegistry + WorkerLaneRegistry.
- Wire read-only /tasks and /agents only if handlers can access that context safely.
- Empty state must be graceful.
- No natural-language gateway auto-routing yet.
- No worker dispatch/cancel/new-task behavior yet.
```

This is the right bridge between the completed Phase 6 formatter and the Manus/Claude Agent View-like UX Woo is asking for.
