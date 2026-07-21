# Wave 1 — External audit and agent evaluation grounding

## Findings

- OpenTelemetry defines events as meaningful point-in-time occurrences,
  including state changes and outcomes. A terminal approval decision is such an
  event, so a receipt should use a stable event name, timestamp, and structured
  attributes rather than an ad-hoc prose log. Source:
  <https://opentelemetry.io/docs/specs/semconv/general/events/>.
- OpenAI's production Codex safety report describes agent-native logs that link
  user intent, approval decisions, tool results, and policy outcomes. Hermes
  should therefore capture provenance and terminal decision status, but must
  not equate approval with a successful external mutation. Source:
  <https://openai.com/index/running-codex-safely/>.
- NIST's AI RMF resources frame transparency, documentation, testing,
  evaluation, verification, and validation as operational risk-management
  evidence. This supports a local, inspectable, privacy-preserving receipt
  record rather than outbound telemetry. Sources:
  <https://airc.nist.gov/> and
  <https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf>.

## Design constraints derived

1. Append only after a terminal decision is known.
2. Record only non-secret, local provenance required to link decision,
   original proposal, and execution result.
3. Keep receipt data as audit provenance; verified outcome gates remain the
   exclusive source of learning eligibility.
4. Exercise receipt creation through the real CLI approval/rejection path.

## EXPAND

- LEAD: Receipt schema and crash behavior need repository-specific proof — WHY: external guidance does not define Hermes's durable boundary — ANGLE: codebase persistence worker.
- LEAD: Verified outcome promotion must remain separate from receipt audit records — WHY: approval is not evidence of real-world success — ANGLE: goal-learning worker.
