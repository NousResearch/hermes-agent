# Ultraresearch Synthesis: verified goal-learning proposal freshness

Workers: 5 attempted · Waves: 3 · Verifications: 4 executed commands

## Executive summary

Hermes already treated goal outcomes as explicit, confirmed, fresh-evidence
candidates rather than auto-learning.  The remaining immediate gap was the
later application of a staged memory or skill proposal: queue claims prevented
duplicate approval but did not bind a memory proposal to the reviewed state,
and skill replay lost trusted background provenance.

The selected implementation adds a narrow V2 memory proposal contract.  It
hashes the exact staged record, captures a raw-byte target revision under the
existing memory lock, and rechecks that revision under the same lock before
mutation.  Legacy, modified, malformed, and stale records become terminal
no-ops; ordinary apply failures retain retry behavior.  Approved skills now
replay the original, allow-listed provenance so existing action-sink guards
remain active.  [verify-memory-proposal-freshness.md](verify-memory-proposal-freshness.md)

## Findings by theme

### Goal-directed learning boundary

- Confirmed goal outcomes remain pull-only, same-session/workspace scoped, and
  fresh-evidence gated; this patch does not inject them into prompts or memory.
  [wave-1-approval-agent-architecture.md](wave-1-approval-agent-architecture.md)
- Memory state can only change from a fresh reviewed target revision, reducing
  stale proposal and memory-poisoning risk without introducing self-modifying
  learning. [wave-1-memory-freshness.md](wave-1-memory-freshness.md)

### AI-agent operating safety

- A staged background skill preserves its host-bound provenance at replay, so
  ownership, pin, bundled, and external guards still run. [wave-1-skill-integrity.md](wave-1-skill-integrity.md)
- Full skill static review artifacts and CAS need action-specific designs and
  are explicitly deferred rather than partially claimed. [wave-2-terminal-and-verification.md](wave-2-terminal-and-verification.md)

## Codebase changes

- `tools/write_approval.py`: canonical digest support for versioned proposals.
- `tools/memory_tool.py`: target revision capture, same-lock precondition
  verification, and fail-closed record replay.
- `hermes_cli/write_approval_commands.py`: terminal memory proposal handling.
- `tools/skill_manager_tool.py`: allow-listed replay provenance restoration.
- `agent/learning_mutations.py`: journey memory writes join the shared lock.

## Verified claims

See [verify-memory-proposal-freshness.md](verify-memory-proposal-freshness.md).

## Gaps and scope boundaries

- Advisory locks cannot control a hostile external writer after the locked
  comparison.  The implementation detects state observed before apply and
  serializes Hermes-internal writers.
- The full Skills V2 static review artifact/CAS and append-only terminal
  decision receipt ledger remain separate work; neither is represented as
  complete here.

## Expansion trace

- Wave 1: memory freshness, skill integrity, approval/agent architecture.
- Wave 2: terminal lifecycle and portable outcome verification.
- Wave 3: implementation integration, parent readback, and executed tests.
- Convergence: no unchecked in-scope lead remains.
