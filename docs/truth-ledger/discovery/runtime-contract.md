# T2 — post_llm_call runtime contract (CLI, gateway, chat -q, cron, subagent, kanban)

Date: 2026-07-17 (updated by t_ceae60bd)
Task: t_3ddb7659, t_ceae60bd
Workspace: /Users/hermes/.hermes/hermes-agent/.worktrees/truth-ledger-option-2

## Scope

Freeze what `post_llm_call` can prove today for:
- CLI interactive
- gateway turns
- `chat -q`
- cron jobs
- delegated subagents
- kanban workers

Focus fields:
- success/completion
- top-level vs child
- speaker/chat/thread identity
- stable turn identity inputs for idempotency

## Canonical emission point

`post_llm_call` is emitted in one place:
- `agent/turn_finalizer.py:365-378`

Current guard:
- fires when `final_response` is truthy and `interrupted` is false (`agent/turn_finalizer.py:365`)

Current kwargs sent to hook:
- Existing fields (unchanged):
  - `session_id`
  - `task_id`
  - `turn_id`
  - `user_message`
  - `assistant_response`
  - `conversation_history`
  - `model`
  - `platform`
- Additive R1 fields (backward-compatible kwargs enrichment):
  - Eligibility: `completed`, `failed`, `interrupted`, `turn_exit_reason`
  - Origin: `delegate_depth`, `is_subagent`, `parent_session_id`, `kanban_task_id`
  - Identity/lane: `speaker_id`, `conversation_id`, `chat_id`, `thread_id`, `chat_type`

No additional metadata is auto-injected by plugin manager (`hermes_cli/plugins.py:1890-1925`); metadata is supplied at the emission site in `turn_finalizer`.

## Turn/task identity construction

Turn identity is generated in `build_turn_context`:
- `effective_task_id = task_id or uuid4()` (`agent/turn_context.py:216-217`)
- `turn_id = f"{session_id}:{effective_task_id}:{random8}"` (`agent/turn_context.py:218`)

Implication:
- `turn_id` is unique per turn, not deterministic.
- Idempotency should key primarily on `(profile, session_id, turn_id)` (or equivalent) rather than message text.

## Runtime path trace by mode

### 1) CLI interactive

- Agent platform is fixed to `"cli"` (`hermes_cli/cli_agent_setup_mixin.py:372`).
- Turn call passes `task_id=self.session_id` (`cli.py:12341-12346`).

What hook can prove:
- stable session-scoped task identity (`task_id == session_id`)
- platform is CLI

What it cannot prove:
- speaker/chat/thread identity (none in hook kwargs)
- explicit success/completion (only inferred by guard)

### 2) Gateway

- Gateway binds session contextvars with platform/chat/thread/user (`gateway/run.py:14860-14891`).
- Gateway agent is created with platform + user/chat/thread fields (`gateway/run.py:18171-18200`).
- Gateway turn call passes `task_id=session_id` (`gateway/run.py:18777-18790`).

What hook can prove from kwargs alone:
- platform string
- session_id/task_id/turn_id

What it cannot prove from kwargs alone:
- speaker_id/user_id
- chat_id/conversation_id
- thread_id

Note:
- Those IDs exist in runtime contextvars for gateway execution, but they are not part of the `post_llm_call` kwarg contract today.

### 3) `chat -q`

Two distinct execution shapes:

A) Default `chat -q` (non-quiet)
- Code path calls `cli.chat(...)` (`cli.py:16259`), which uses `task_id=self.session_id` (`cli.py:12345`).

B) `chat -q --quiet` / `-Q`
- Direct call omits `task_id` (`cli.py:16162-16165`), so `effective_task_id` becomes random UUID (`agent/turn_context.py:216`).

What hook can prove:
- platform CLI
- turn uniqueness

What varies by flag:
- `task_id` stability differs between non-quiet vs quiet one-shot path.

### 4) Cron

- Cron agent platform is `"cron"` and runs with `skip_memory=True` (`cron/scheduler.py:3044-3073`).
- Cron call is `agent.run_conversation(prompt)` without explicit `task_id` (`cron/scheduler.py:3104`).
- Cron intentionally clears `HERMES_SESSION_*` identity context for run execution (`cron/scheduler.py:2692-2717`).

What hook can prove:
- platform cron
- session_id and generated turn_id

What it cannot prove:
- speaker/chat/thread identity (intentionally absent)
- stable external conversation identity

### 5) Delegated subagents

- Child agent platform is `"subagent"` (`tools/delegate_tool.py:1318`).
- Child call explicitly passes `task_id=child_task_id` (`tools/delegate_tool.py:1921-1924`), where child_task_id is the subagent id fallback (`tools/delegate_tool.py:1880`).

What hook can prove:
- this is subagent context via platform
- subagent-scoped task identity

What it cannot prove from kwargs:
- parent/child linkage metadata (depth, parent ids are on agent internals, not in post hook kwargs)

### 6) Kanban workers

- Dispatcher spawn command is `... chat -q "work kanban task <id>"` without `--quiet` (`hermes_cli/kanban_db.py:7668-7669`, `7795-7797`).
- Worker env includes `HERMES_KANBAN_TASK=<card-id>` (`hermes_cli/kanban_db.py:7711`).
- Because this path is non-quiet `chat -q`, it uses `cli.chat(...)` and passes `task_id=self.session_id` (same as CLI interactive).

What hook can prove from kwargs:
- platform CLI
- session_id/task_id/turn_id

What it cannot prove from kwargs:
- kanban card id (`HERMES_KANBAN_TASK` is env/runtime, not a hook kwarg)
- top-level worker role vs normal CLI one-shot

## Success contract analysis (critical)

Emission guard remains:
- `if final_response and not interrupted` (`agent/turn_finalizer.py`)

R1 closes the ambiguity by passing explicit eligibility fields to kwargs:
- `completed`
- `failed`
- `interrupted`
- `turn_exit_reason`

Result: consumers no longer need to infer success from the hook firing condition.

## Frozen contract table

Current `post_llm_call` kwarg contract can reliably provide:
- Base fields: `session_id`, `turn_id`, `platform`, `task_id`, `assistant_response`
- Eligibility: `completed`, `failed`, `interrupted`, `turn_exit_reason`
- Origin: `delegate_depth`, `is_subagent`, `parent_session_id`, `kanban_task_id`
- Identity/lane when available: `speaker_id`, `conversation_id`, `chat_id`, `thread_id`, `chat_type`

Unknown identity remains explicit (`None` values), enabling fail-closed admission logic.

## Go / No-Go for user identity and preference facts

Decision: GO for identity-aware admission gating at hook-contract level, with fail-closed behavior for unknown identity.

Reason:
- R1 adds explicit eligibility and identity/origin metadata without removing legacy fields.
- Unknown identity remains explicit (`None`) rather than inferred.
- Consumers can require both positive eligibility and non-null identity fields before activating user-scoped facts.

Compatibility note:
- Existing `post_llm_call` callbacks remain compatible because the change is additive kwargs only; legacy fields are unchanged.

## Acceptance check against T2 goals

- No assumption left that `post_llm_call` implies success.
- No assumption left that current kwargs imply top-level origin.
- Speaker/chat/thread identity is explicit when present and explicit `None` when unavailable.
- Turn identity inputs and variability are frozen by call path.
- Explicit GO decision recorded for identity/preference gating at the runtime-contract level, with fail-closed unknown identity semantics.