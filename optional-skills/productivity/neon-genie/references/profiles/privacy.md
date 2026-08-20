# Privacy Profile

Always load with `core`. Default to `local_only`; private facts stay local and become DataRequests. Before any host-tool egress, call `RUNE.PRIVACY_EGRESS_CHECK(payload, destination, purpose, privacy_context)`. Only `ALLOW` or `REDACT_THEN_ALLOW` can be represented as an external action. `BLOCK` and `REQUEST_CONSENT` never fall through to execution. Record only classifications and provenance, never raw secrets or private payloads.

Briefs may set `privacy.mode` and purpose-bound `privacy.consents` (never a global disable). Configuration must flow into receipt, envelope, packets, and local learning-ledger disclosure. `REDACT_THEN_ALLOW` yields a minimized `safe_query` without persisting private source content. Every external action records provider, tool class, destination, purpose, source class, classification, decision, redaction status, and UTC `recorded_at`. Host/provider retention remains `NOT_COMPUTABLE`.
