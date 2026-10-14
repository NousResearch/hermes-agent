"""hermes_durability — embeddable durable-execution layer for Hermes.

Three guarantees, stdlib-``sqlite3`` only (per the pyproject scope rule that
core deps must be universal):

  * crash-safe sessions   — fsynced append-only journal (hash chain, torn-tail
                            detection) + replay recovery (``journal``/``runtime``)
  * exactly-once outbound — transactional outbox committed atomically with the
                            journal, idempotency-keyed delivery, retry/DLQ
                            (``outbox``)
  * egress guardrails     — mandatory redaction at the outbound platform
                            boundary, wired at the gateway/cron/tool send choke
                            points (``egress``), extensible via the
                            ``outbound_message`` middleware kind

The journal/outbox engine is deliberately framework-agnostic so other agent
runtimes can embed it; the Hermes-specific wiring lives in ``egress`` and the
call sites it names.
"""

from hermes_durability.egress import EgressBlocked, guard_outbound_text
from hermes_durability.guardrail import Envelope, Guardrail, Rule, Verdict
from hermes_durability.journal import (ASSISTANT_MESSAGE, COMPACTION_COMPLETE,
                                       COMPACTION_SNAPSHOT, GUARDRAIL_AUDIT,
                                       Journal, JournalTransaction,
                                       OUTBOX_DELIVERED, OUTBOX_ENQUEUED,
                                       Record, SESSION_START,
                                       TOOL_CALL_INVOKED, TOOL_CALL_RESULT,
                                       TXN_COMMIT, USER_MESSAGE)
from hermes_durability.outbox import Adapter, OutboxWorker, RetryPolicy
from hermes_durability.runtime import DurableRuntime

__all__ = [
    "DurableRuntime", "Journal", "JournalTransaction", "Record",
    "OutboxWorker", "RetryPolicy", "Adapter",
    "Guardrail", "Rule", "Envelope", "Verdict",
    "EgressBlocked", "guard_outbound_text",
    "SESSION_START", "USER_MESSAGE", "ASSISTANT_MESSAGE",
    "TOOL_CALL_INVOKED", "TOOL_CALL_RESULT", "OUTBOX_ENQUEUED",
    "OUTBOX_DELIVERED", "TXN_COMMIT", "COMPACTION_SNAPSHOT",
    "COMPACTION_COMPLETE", "GUARDRAIL_AUDIT",
]
