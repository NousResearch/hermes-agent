# P0 Threat Model / Pre-Mortem

This register freezes the highest-value failure modes before runtime work
starts. The scope is documentation and replay readiness, not production
mitigation.

| Risk | Failure condition | Evidence to detect it | Mitigation | Rollback |
|---|---|---|---|---|
| Schema drift | P0-T02/T03 schemas diverge from current `cron/*.py` contracts | validator fails on examples or a new field disappears without update | keep schemas permissive for legacy fields; pin required fields only | edit schema docs only |
| Evidence leakage | audit/evidence examples accidentally contain prompts, tokens, or receipts | grep finds auth-like strings, full prompts, or unredacted payloads | redact by default; use synthetic fixtures only | delete the offending fixture/file |
| Transition ambiguity | two states claim the same trigger or a rule ID is reused | state-machine validator sees duplicate rule IDs or conflicting transitions | freeze rule IDs in one file and fail closed on conflict | rename rule ID in docs only |
| False stale-running reset | a live run is misread as abandoned | replay fixture shows `running` with live owner evidence | require max_runtime plus owner-liveness proof before `reset_job` | keep verdict as `suspect` |
| Unsafe repair escalation | storage repair is treated as a routine auto-fix | fixture or review shows L2/L3 repair without human approval | keep `repair_state_store` outside automatic action paths | force `human_required` |
| Replay drift | fixtures no longer match the documented expected verdict/action | validator reports a mismatch or a missing expected block | validate every fixture against an explicit expected verdict block | regenerate fixture and update hashes |
| Unresolved runtime decisions | P0 looks complete but section-21 decisions stay implicit | review finds missing owner/auth/approver notes | record open items as `manual_only` or `UNAVAILABLE` in docs | add a decision note, do not infer runtime behavior |

Frozen decisions for P0:

- `control_policy` is documented as an optional job extension, not a runtime
  migration requirement.
- `state_store` L2/L3 repair remains human-only.
- Control API authentication is intentionally left open for later phases.
- External notification owner mapping is out of scope for P0.
