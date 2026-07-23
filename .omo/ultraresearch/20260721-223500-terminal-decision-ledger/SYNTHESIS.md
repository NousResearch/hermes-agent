# Ultraresearch Synthesis: Immutable approval-decision receipts

Workers: 8 · Waves: 3 · Sources: 5 primary external sources plus repository
code/history · Verifications: 1 focused baseline execution suite.

## Executive summary

Hermes's pending memory and skill approval queue already prevents duplicate
apply through an atomic claim and retains retry only for non-terminal failures.
It does not preserve a durable record once a terminal claim is cleaned up.
The smallest compatible extension is a profile-scoped immutable
`approval_decision_receipts` table inside the existing
`verification_evidence.db`.

The receipt must be separate from `outcome_receipts`: outcome confirmation
updates its row and can eventually become a learning candidate, while approval
is only authorization/audit provenance. Receipt rows contain no raw proposal
payload, summary, session, or actor: current shared CLI/gateway/TUI paths do
not possess a universally trustworthy identity. Decision rows record the
subsystem, pending ID, proposal digest, decision/outcome, safe failure code,
and time.

## Findings by theme

### Lifecycle and crash safety

- The only emission boundary is a claimed record with terminal outcome:
  apply success, explicit rejection, or memory's terminal no-op. Requeueable
  failures emit no receipt. Evidence: `tools/write_approval.py:217,342,375,407`
  and `hermes_cli/write_approval_commands.py:108-210`.
- Rejections persist receipt before claim cleanup. Approvals first apply; a
  successful or terminal-no-op result then persists receipt and cleans up.
  Receipt failure holds the claim with a non-reapply recovery message.
- The queue filesystem and SQLite cannot share a transaction. The design
  explicitly favors at-most-once mutation over automatic recovery.

### Persistence and audit boundary

- `verification_evidence.db` is profile scoped, WAL backed, and covered by
  WAL-safe full and quick backups. Evidence:
  `agent/verification_evidence.py:62-151`, `hermes_cli/backup.py:788`.
- `outcome_receipts` is mutable on confirmation, so a separate table with
  unique proposal identity and UPDATE/DELETE rejection triggers is necessary.
- No doctor change is justified: doctor has a state-DB-specific recovery
  purpose and no corrective policy for evidence DB corruption.

### Goals and learning

- Current learning reuse requires explicit outcome confirmation and fresh
  passing verification. Approval receipts neither alter `GoalState` nor become
  reusable outcome evidence. Evidence: `agent/verification_evidence.py:752-1010`.
- Focused baseline checks for root/session isolation and stale evidence passed;
  the reported failure came from a missing WSL distribution before tests ran.
  See `verify-outcome-baseline.md`.

## External sources (accessed 2026-07-21)

1. OpenTelemetry event conventions — meaningful state changes and outcomes:
   <https://opentelemetry.io/docs/specs/semconv/general/events/>.
2. OpenAI, Running Codex safely — links intent, approval decisions, tool
   results, and policy outcomes in agent-native telemetry:
   <https://openai.com/index/running-codex-safely/>.
3. NIST AI RMF resources — traceability, testing, evaluation, verification and
   validation as risk evidence: <https://airc.nist.gov/>.
4. NIST Generative AI Profile:
   <https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf>.
5. OpenAI, Practices for Governing Agentic AI Systems — accountability and
   operational safety practices: <https://openai.com/index/practices-for-governing-agentic-ai-systems/>.

## Verified claims

| Claim | Verdict | Evidence |
| --- | --- | --- |
| Outcome root/session baseline fails on current main | Refuted | `verify-outcome-baseline.md` |
| Generic terminal approval queue is safe to merge into proposal ledger | Refuted | `wave-3-terminal-command-scope.md` |
| Existing outcome receipt row is safe for immutable approval audit | Refuted | `wave-2-receipt-schema.md` |
| Separate immutable table in evidence DB is compatible with backup/profile scope | Confirmed | `wave-1-codebase-persistence.md`, `wave-2-receipt-schema.md` |

## Gaps and boundaries

- Per-user attribution needs a future explicit decision-context contract.
- Generic terminal/tool approval receipt correlation is a separate capability.
- Receipt persistence failure after mutation remains an intentionally held,
  manually reconciled claim; no automatic replay is permitted.

## Expansion trace

Wave 1 mapped lifecycle, persistence, goal-learning, and external guidance.
Wave 2 verified the outcome baseline, receipt schema, and health scope. Wave 3
closed held-claim, attribution, and generic terminal-command boundaries.
Convergence: all leads investigated or explicitly scoped out.
