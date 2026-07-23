# ULR: goal-learning contract integrity

## Decision

Hermes may retain a verified, human-confirmed outcome as a reusable learning
candidate only when the receipt binds the final completion criteria that were
actually evaluated and its verification evidence remains fresh.

This change adds a nullable, versioned SHA-256 completion-contract digest to
the append-only outcome receipt. The digest covers the canonical completion
contract and the ordered subgoals, but stores no raw goal, criteria, or subgoal
text. Legacy receipts remain readable during the schema v4-to-v5 migration.

## Workspace-evidence boundary

Attempted foreground terminal commands now mark the workspace stale before a
normal terminal result is recorded and before timeout or final execution-error
returns. The rule intentionally treats all shell commands as possible edits:
terminal writes cannot be safely inferred from command text. A recognized
verification command immediately follows the marker with new evidence, so it
can establish a fresh passing state. Background process provenance is
explicitly not changed in this increment.

## Why this is a narrow AGI/agent foundation increment

The capability does not give Hermes autonomous self-modification or prompt
injection. It strengthens the auditable loop of goal -> evaluated criteria ->
human confirmation -> reusable evidence, while preserving current session and
workspace boundaries.

## Research inputs

- OpenAI Agents SDK documents persistent sessions, human-in-the-loop pauses,
  and tracing as separate mechanisms: [sessions](https://openai.github.io/openai-agents-python/sessions/),
  [human in the loop](https://openai.github.io/openai-agents-python/human_in_the_loop/),
  and [tracing](https://openai.github.io/openai-agents-python/tracing/).
- Anthropic recommends robust long-running harnesses and evaluation systems:
  [effective harnesses](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)
  and [agent evaluations](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents).
- The design inference for Hermes is to bind reusable learning to stable,
  reviewable evaluation provenance rather than to let outcomes automatically
  alter agent behavior.

## Verification

`scripts/run_tests.sh tests/agent/test_verification_evidence.py
tests/hermes_cli/test_goals.py tests/tools/test_terminal_verification_freshness.py
-q` passed 147 tests. The terminal regression covers successful and failing
foreground commands, timeout and final execution-error returns, stale prior
evidence, and a later fresh verification.
