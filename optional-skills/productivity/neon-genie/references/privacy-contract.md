# Privacy Contract 1.0.0

Neon Genie repository telemetry is disabled, it does not train models or construct advertising profiles from run content, and artifacts remain at the selected output path. Host, OS, Hermes, model, search, and MCP provider retention or processing are `NOT_COMPUTABLE` unless separately established. `research.enabled=false` and `local_only` produce no Neon-Genie-initiated external actions. Delete artifacts by deleting the selected output directory.

## Egress

Egress decisions are exactly `ALLOW`, `REDACT_THEN_ALLOW`, `REQUEST_CONSENT`, or `BLOCK`.

| Condition | Decision |
|-----------|----------|
| `mode=local_only` or `egress.allowed=false` | `BLOCK` |
| Missing destination or purpose | `BLOCK` |
| Secret/credential patterns (API keys, bearer, private keys, connection strings, GitHub tokens, card-like values) | `BLOCK` |
| Private customer lists / multi-email without purpose-bound consent | `REQUEST_CONSENT` |
| Private markers with matching purpose-bound consent, or redaction-needed public mixed query | `REDACT_THEN_ALLOW` |
| Clean public query with approved destination (if allowlisted) | `ALLOW` |

`REDACT_THEN_ALLOW` must produce a minimized `safe_query` and must **not** persist private source content.

## Consent

Purpose-bound consent records only. Forbidden scopes include `global`, `global_disable`, `disable_privacy`, `all`, `unrestricted`, `bypass`. Unsupported `contract_version` values raise `NG-PRIVACY-011`.

## External action records

Required fields: `provider`, `tool_class`, `destination`, `purpose`, `source_class`, `classification`, `decision`, `redaction_status`, `recorded_at` (UTC Zulu). Only `ALLOW` / `REDACT_THEN_ALLOW` may be recorded as executed external actions.

## Learning ledger

Local operator path only (default `out/neon-genie/learning-ledger.jsonl`). Never auto-applied to the skill corpus. Must not store raw secrets or private source payloads. Entries disclose `privacy_mode` and local-only learning flags when recorded via `scripts/record_learning.py`.

## Profiles

Always load `core, privacy` for routed runs. Public facts may use host research when egress is allowed; private facts become DataRequests.
