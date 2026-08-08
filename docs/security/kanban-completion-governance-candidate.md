# Governed Kanban completion — 2B patch candidate

Status: **candidate only; not installed, not activated**

This document describes the provider-free 2B candidate. It is not a deployment
instruction and does not authorize a live database migration, permission change,
service install, gateway restart, worker dispatch, provider inference, or E2E run.

## Security invariant

A governed task may become `done` only when all of the following are true in the
same `BEGIN IMMEDIATE` transaction as the final task-state CAS:

1. exactly one hash-valid policy and activation record exists;
2. activation is enabled and the kill switch is false;
3. the authoritative SQLite path equals the activated board path;
4. the caller source and runtime profile are allowlisted;
5. a structured completion context binds caller profile, native task and native run;
6. task status is `running`, `expected_run_id` is present, and the native run matches;
7. one hash-valid task/profile/prompt/task-envelope binding exists;
8. the result is valid JSON and passes the hash-bound Draft 2020-12 result schema;
9. result task, agent, prompt, approval, QA gate and evidence match the binding;
10. no open question remains and every evidence item is `verified`;
11. a transaction-local trigger permit exists for the exact task/result pair;
12. the task CAS, closing run, completed event and immutable receipt all commit;
13. receipt failure rolls back task/run/event/permit state and removes any newly
    staged scratch artifact copy.

Presence of any policy, activation or binding row opts the board into governance.
Partial, malformed, unreadable or inconsistent state denies completion. A board
with no governance rows retains legacy behavior.

## Implemented candidate components

- `hermes_cli/kanban_completion_guard.py`
  - strict policy/activation/binding parser;
  - `CompletionContext` and semantic result-envelope validator;
  - receipt construction and insertion;
  - worker isolation policy loader;
  - fail-closed completed-result immutability.
- `hermes_cli/kanban_db.py`
  - governance tables, transaction-local permits and persistent receipts;
  - triggers against direct completion, result replacement and reopening;
  - guard execution inside the completion write transaction before the CAS;
  - atomic receipt/event/run/task mutation;
  - governed `edit_completed_task_result()` denial;
  - governed dispatcher isolation pin.
- `hermes_cli/kanban_completion_broker.py`
  - strict newline-delimited JSON protocol over a Unix socket;
  - Linux `SO_PEERCRED` UID authentication;
  - UID-to-profile allowlist;
  - broker-generated `CompletionContext(source="broker")`;
  - no host-artifact-copy support for lower-privilege clients;
  - same transactional kernel as in-process callers.
- `tools/kanban_tools.py`
  - worker completion uses the broker when
    `HERMES_KANBAN_BROKER_SOCKET` is pinned by policy;
  - no direct DB connection occurs on that completion path.
- `tools/terminal_tool.py` and `tools/environments/docker.py`
  - isolated workers are forced to Docker regardless of inherited/profile config;
  - only the task workspace is mounted;
  - network, arbitrary volumes, forwarded environment, credentials, skills,
    cache mounts, egress proxy mounts, host UID, persistence and reuse are disabled.
- `plugins/kanban/systemd/`
  - non-installed system service and strict example configuration.

## Trust boundaries

### Transactional kernel

All official in-process completion paths converge on `complete_task()`. Governed
CLI, dashboard or direct-Python callers without a trusted `CompletionContext`
are denied. The model tool constructs a worker context only from dispatcher-pinned
environment; the broker constructs one only after `SO_PEERCRED` authorization.

### SQLite triggers

Triggers are defense in depth against accidental raw SQL. They are **not** the
ultimate boundary because a process that owns the DB can drop triggers or forge
permit rows.

### DB owner and OS identity

For hostile-worker safety, the DB, WAL and SHM files must be owned by a dedicated
`hermes-kanban-broker` Unix user and inaccessible to Coordinator and worker UIDs.
Each specialist profile must have a distinct UID (or an equally strong container /
mandatory-access-control identity); otherwise a shared UID can impersonate another
profile at the broker socket.

The socket group grants only broker access, never DB-directory access. The systemd
candidate uses a broker-owned `0700` state directory and a group-accessible `0660`
Unix socket. The Coordinator Telegram gateway remains a separate service and is
not disabled or reconfigured by this design.

## Non-live activation sequence requiring separate approval

The following is intentionally **NOT RUN**:

1. Freeze and independently QA the exact candidate diff.
2. Back up and hash-read-back the live DB, WAL/SHM state, policy artifacts and
   current ownership/modes.
3. Stop all writers in a maintenance window.
4. Create dedicated broker and per-profile OS identities.
5. Copy the DB to a staging path, run schema initialization, load policy,
   activation and task bindings with activation still disabled / kill switch true.
6. Install the reviewed broker binary/config/unit and apply broker-only DB ownership.
7. Prove unauthorized UIDs cannot open the DB/WAL/SHM but can reach only their
   authorized socket profile.
8. Start the broker, run `ping`/negative synthetic checks without task completion.
9. Run approved synthetic staging completion and rollback rehearsal.
10. Obtain independent Security and QA PASS for the frozen diff and evidence.
11. Obtain explicit human approval for the exact live diff, identities, DB hash,
    migration, service unit and activation change.
12. Only then enable the policy and disengage the kill switch in one controlled
    activation transaction.

## Rollback plan

Rollback must happen in the approved maintenance window and fail closed:

1. Engage kill switch and disable governed completion; stop dispatcher/broker
   completion intake without stopping the Coordinator Telegram conversation surface.
2. Stop all Kanban writers and verify no process holds the DB/WAL/SHM files.
3. Archive and hash the failed state for forensics; do not overwrite evidence.
4. Restore the pre-migration DB/WAL/SHM backup as one consistent set and read back
   its hashes and `PRAGMA integrity_check` result.
5. Restore the recorded DB owner, group and mode; remove only the reviewed broker
   socket/unit/config artifacts installed by the activation change.
6. Restore the pre-candidate Hermes source/package and verify its hash.
7. Keep activation disabled and kill switch engaged.
8. Start only the previously running services; verify the Coordinator Telegram
   gateway without sending a message.
9. Do not resume specialist dispatch until a new QA/Security decision and explicit
   human approval.

## Known candidate limits

- The broker currently exposes completion only. Other write operations (claim,
  heartbeat, comment, block, create) need equivalent broker endpoints before a
  full separate-UID worker can operate with all direct DB access removed.
- Artifact promotion is intentionally rejected by the broker until a separately
  owned staging protocol is designed.
- Docker availability/image provenance and the complete per-profile UID rollout
  remain deployment prerequisites and are not proven by unit tests.
- No live migration, real Unix-user permission test, container execution,
  provider-backed E2E, gateway restart or rollback rehearsal has been run.
