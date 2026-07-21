# Ultraresearch Synthesis: Hermes verified goal-learning control plane

Workers: 4 · Waves: 3 · Sources: 12 · Verifications: 2 focused runs + independent review

## Executive summary

Hermes already persisted session goals, durable wait resumption, verification
evidence, and explicitly confirmed outcome receipts. The missing usable seam
was a production caller for the pull-only reusable-receipt query. The selected
extension adds `/goal outcomes` and `/goal learning` across CLI, gateway, and
TUI, rather than adding a second task database or automatic memory system.
[wave-1-codebase-goal-lifecycle.md](wave-1-codebase-goal-lifecycle.md)
[wave-1-codebase-learning-evidence.md](wave-1-codebase-learning-evidence.md)

The surface is deliberately constrained: it filters to the active session and
canonical workspace, rechecks currently passing evidence, fails closed when no
workspace can be identified, and only reports receipt id/timestamp metadata.
It neither stores raw goal text nor injects prompts, memories, skills, tool
calls, or external events. [verify-goal-outcomes.md](verify-goal-outcomes.md)

## Findings by theme

### Durable goals and verified learning

- Current Hermes follows a useful minimal separation: session goal state in
  `state_meta` and verification/receipt evidence in its own SQLite ledger.
  [wave-1-codebase-goal-lifecycle.md](wave-1-codebase-goal-lifecycle.md)
- Confirmed receipts require human confirmation plus fresh passing evidence;
  later workspace edits invalidate older learning candidates. This is a
  stronger boundary than a model-completion label. [wave-1-codebase-learning-evidence.md](wave-1-codebase-learning-evidence.md)
- The new control plane keeps retrieval explicit. It therefore exposes a
  learning capability without conflating it with automatic self-modification.
  [verify-goal-outcomes.md](verify-goal-outcomes.md)

### Architecture evidence

- Magentic-One separates task and progress ledgers; A2A represents explicit
  task state and idempotent delivery. These support an observable control plane
  but do not require a new framework in Hermes at this stage.
  [Magentic-One](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/magentic-one.html)
  [A2A specification](https://a2a-protocol.org/latest/specification/)
- Durable resume must treat external effects as idempotent, not assume exactly
  once delivery. Hermes's existing leased wait continuation matches this
  constraint. [LangGraph interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts)
  [Temporal workflow execution](https://docs.temporal.io/workflow-execution)
- Human approval and persisted state are data boundaries. The selected feature
  does not create a tool approval bypass or serialize new sensitive state.
  [OpenAI Agents HITL](https://openai.github.io/openai-agents-python/human_in_the_loop/)
  [OWASP agentic security guide](https://genai.owasp.org/download/49059/)

### Evaluation and observability

- Deterministic outcome/trajectory checks should precede LLM-rubric evaluation
  for critical behavior. The focused regression suite asserts session scoping,
  fail-closed workspace handling, and the three transport paths.
  [Google ADK evaluation](https://adk.dev/evaluate/)
  [WebArena-Verified](https://github.com/ServiceNow/webarena-verified)
  [verify-goal-outcomes.md](verify-goal-outcomes.md)

## Codebase findings

- `agent/verification_evidence.py`: optional session-scoped receipt retrieval
  now fails closed without a canonical workspace root.
- `hermes_cli/goals.py`: one shared read-only outcome formatter keeps CLI,
  gateway, and TUI semantics aligned.
- `hermes_cli/cli_commands_mixin.py`, `gateway/slash_commands.py`, and
  `tui_gateway/server.py`: add `/goal outcomes` and `/goal learning`.
- Tests cover ledger isolation plus each command transport.

## Verified claims

| Claim | Verdict | Evidence |
| --- | --- | --- |
| Session-scoped outcomes cannot fall back across unrecognised workspaces. | CONFIRMED | `tests/agent/test_verification_evidence.py`; `verify-goal-outcomes.md` |
| CLI, gateway, and TUI dispatch the shared read-only summary. | CONFIRMED | `tests/hermes_cli/test_goals.py`, `tests/gateway/test_goal_max_turns_config.py`, `tests/tui_gateway/test_goal_command.py` |
| Focused post-repair suite is green. | CONFIRMED | 158 passed / 0 failed in `verify-goal-outcomes.md` |
| Independent review found no merge blocker. | CONFIRMED | Wave 3 reviewer PASS recorded in `expansion-log.md` |

## Contradictions and resolution

The external architecture survey described a larger `GoalLedger v0`. The
codebase survey found that Hermes already has the relevant ownership domains
and that duplicating them would create recovery conflicts. The bounded
read-only outcome surface was selected because it closes a demonstrated gap
without adding automatic learning or a competing task store.

## Gaps

- Terminal commands that edit files may not update the workspace freshness
  marker; no E2E proof justified changing that behavior in this scope.
- Long-term outcome-receipt retention and generic task-ledger interoperability
  remain separate design work.
- The review-expanded run observed two unrelated Windows failures in
  `tests/tools/test_approval.py` after 310 passing tests (POSIX `/tmp`
  expectation and unavailable symlink privilege). No approval code changed.

## Expansion trace

- Wave 1: goal lifecycle, evidence lifecycle, architecture, and parent source
  cross-checks identified the unused pull-only receipt seam.
- Wave 2: integration review required direct CLI/gateway tests.
- Wave 3: review remediation fixed a test-boundary error and a workspace-root
  fail-open edge; the final 158-test run and independent PASS closed all
  implementation leads.
