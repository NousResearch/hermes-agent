# gateway-deferred-restart-g3-audit Specification

## ADDED Requirements

### Requirement: Cron prohibition on the deferred restart channel (G3)
The deferred restart command SHALL be prohibited in cron execution contexts at
three layers. Layer 1 — definition paths: cron job create/edit/import/re-enable
SHALL refuse payloads that reference the deferred restart entrypoint, using
mention-level detection (no inert-body stripping: a heredoc that merely carries
the text is refused too), per-carrier scanning (prompt, monitor source and
referenced scripts each scanned independently, with combined-query splitting and
scripts resolved under the ACTIVE home binding). Layer 2 — dispatch: the real
dispatch entry SHALL re-check every payload fail-closed before any user code or
model runs, including pre-existing jobs. Layer 3 — runtime control point: the
CLI entrypoint SHALL refuse (exit 7) invocations carrying trusted cron
provenance (`HERMES_CRON_SESSION` truthy or the external-worker marker) before
any request file write or signal. Normal chat turns remain allowed subject to
the companion change's authorization, capability and cooldown rules.

#### Scenario: cron job definition rejected
- WHEN creating or editing a cron job whose payload references `hermes gateway defer-restart`, in any carrier or via dynamic assembly inside a generated script
- THEN the definition is refused with a clear error

#### Scenario: pre-execution fail-closed
- WHEN a pre-existing cron job payload containing the deferred restart command is dispatched
- THEN the execution is booked failed with the named reason and zero user-code/model side effects occur

#### Scenario: dynamic assembly reaches the entrypoint and is refused
- WHEN a cron-context bash chain assembles `hermes gateway defer-restart` from variables and executes it
- THEN the entrypoint refuses with exit 7 and no request file is written and no signal is sent

#### Scenario: non-cron control proceeds
- WHEN the identical chain runs without cron provenance
- THEN the request is published and the target is signaled per the companion protocol

#### Scenario: existing guard behavior unchanged
- WHEN any previously blocked command shape is submitted (restart/stop/uninstall forms, launchctl forms, kill forms, spliced/continuation/compound variants)
- THEN the existing G1/G2 scanner behavior is unchanged (verdicts byte-identical)

### Requirement: Audit logging with a degraded, explicit guarantee
Every applicable decision or replay event SHALL attempt one JSONL append to a
protected `audit.jsonl` (request id, bounded actor reference ≤256 chars,
timestamp, target generation, service identity, outcome). A decision audit
SHALL fire only after the authoritative result record is persisted. Append
failures SHALL be reported (log warning) and SHALL NOT alter the authoritative
decision — the result record remains the sole authoritative outcome. The audit
chain MAY have gaps under storage failure; completeness is explicitly NOT
guaranteed. The appender SHALL check actual written length (a short write is
never success), SHALL isolate an incomplete tail left by a crashed writer before
the next append, and SHALL open the file without following symlinks after
verifying it is a regular file owned by the current user (over-open pre-existing
permissions tightened to 0600). actor_ref is attribution only and never an
authorization input.

#### Scenario: decision audited after persistence
- WHEN a decision is persisted to the result record
- THEN exactly one audit event with that outcome is appended afterwards

#### Scenario: duplicate replay audited
- WHEN a request id is replayed and answered duplicate
- THEN a duplicate audit event is appended (time and actor of the replay), and the original result/claim remain unchanged

#### Scenario: append failure reported, decision unaffected
- WHEN the audit append raises an OSError
- THEN a warning is reported and the authoritative decision stands

#### Scenario: short write completed
- WHEN os.write returns fewer bytes than the record length
- THEN the appender continues writing until the record is complete, or reports failure — it never treats the short write as success

#### Scenario: crashed tail isolated
- WHEN the audit file ends with a half-written record from a crashed writer
- THEN the next append isolates the partial line first, keeping earlier complete events and the new event individually readable

#### Scenario: unsafe file refused or repaired
- WHEN the audit path is a symlink, a non-regular file, or owned by another user
- THEN the append is refused and reported without writing through the path
- AND WHEN the pre-existing file has permissions beyond owner-only
- THEN permissions are tightened to 0600 before the append proceeds

#### Scenario: gap in the chain is honest
- WHEN a replay event's append fails under storage failure
- THEN the system logs the failure and does not reconstruct or fake the missing record, and no claim of complete coverage is made
