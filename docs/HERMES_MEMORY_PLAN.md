# Hermes Memory Plan

Date: 2026-05-20

## Current Memory Diagnosis

Hermes currently has multiple memory layers:

- Built-in curated agent memory: `~/.hermes/memories/MEMORY.md`.
- Built-in user preference memory: `~/.hermes/memories/USER.md`.
- Session history and search: `~/.hermes/state.db` and `~/.hermes/sessions/`.
- Structured external memory: active `holographic` provider using
  `~/.hermes/memory_store.db`.
- Response/cache stores: `~/.hermes/response_store.db`.
- Profile-specific memory stores under `~/.hermes/profiles/*`.

Observed issue:

- Curated markdown memory is near capacity.
- Holographic facts can drift when built-in markdown entries are replaced or
  removed.
- Session retention is strong but creates privacy and deletion obligations.

## Memory Tiering

### Session Memory

Use for:

- Active task state.
- Recent failed attempts.
- Temporary notes.
- Logs and one-time runtime facts.
- Short-term coordination.

Backing stores:

- `state.db`
- `sessions/*.jsonl`

Do not promote everything from session memory into curated memory.

### Curated Agent Memory

Use `MEMORY.md` only for stable facts that should shape many future sessions:

- Durable repo/runtime paths.
- Known recovery commands.
- Long-lived architecture decisions.
- Reusable environment quirks.
- Stable operational conventions.

### Curated User Memory

Use `USER.md` only for stable user preferences:

- Communication style.
- Delivery expectations.
- Confirmation preferences.
- Long-lived project priorities.
- Privacy boundaries.

Avoid:

- One-off tasks.
- Raw personal data.
- Secrets.
- Long transcripts.
- “Always do X” entries that should really be gated by context.

### Structured Long-Term Memory

Use holographic facts for:

- Entity/project facts.
- Searchable preferences.
- Tool lessons.
- Cross-entity reasoning.

Actions must support add, search, probe, reason, contradict, update, remove,
and feedback.

### Skills

Use skills for reusable procedures:

- Review workflow.
- Testing workflow.
- Tool-building workflow.
- Release workflow.
- Safe research/report workflows.

If the content is a checklist or procedure, make it a skill, not a memory line.

## Retrieval Strategy

Use retrieval source by question:

- Current runtime health: verify live with `hermes doctor`, `hermes gateway
  status`, launchd, and health-loop files.
- Prior work: use `session_search`.
- User preferences: use `USER.md` and structured fact probe.
- Project facts: use structured search/probe, then verify files if drift-prone.
- Tool behavior: check docs/tests/source first, memory second.

Memory-derived claims should be treated as potentially stale when they describe
live services, credentials, account state, installed tools, or provider status.

## Write Policy

Only write durable memory when:

- The user explicitly asks to remember something.
- A preference is repeated or corrected.
- A stable project/runtime fact is likely useful for at least 30 days.
- A recurring failure has a verified recovery path.

Before writing:

- Check whether an existing entry can be replaced/compacted.
- Avoid duplicating between `USER.md`, `MEMORY.md`, and structured facts.
- Never include secrets, tokens, private keys, raw chat IDs, or private payloads.

## Compaction Plan

Immediate target:

- Use `hermes memory audit --json --redact` to confirm current capacity,
  store coverage, and deletion domains without exposing memory contents.
- Compact `MEMORY.md` and `USER.md` to around 70% of configured capacity only
  after explicit approval to mutate private persistent memory.
- Merge overlapping Hermes runtime notes.
- Move procedural material into skills/docs.
- Keep only stable preferences and durable operational facts.

Validation:

- Run `hermes memory audit --json --redact`.
- Run `hermes memory status`.
- Confirm new memory writes have headroom.
- Search structured facts for duplicates/stale replacements.

Current blocker:

- The Phase 3 implementation added the read-only audit and deletion-readiness
  scaffold. It did not compact live memory files because that would rewrite
  private persistent runtime state without a user-specific memory-edit request.

## Post-Campaign Explicit Approval Plan

Status: approved default-file draft applied. The post-campaign cleanup passes
used separate read-and-draft and apply approvals, touched only the default
eligible memory files, and did not read, rewrite, compact, delete, backfill, or
reconcile additional private memory stores.

### Required Approval Gates

Private memory compaction must use two separate typed approvals in the same
run where the work is performed:

1. Read-and-draft approval:
   `APPROVE HERMES PRIVATE MEMORY COMPACTION DRAFT`
2. Apply approval:
   `APPLY HERMES PRIVATE MEMORY COMPACTION`

The first approval allows a future run to read the approved private memory
files and create local private draft artifacts. It does not allow rewriting
live memory. The second approval is required after the operator has reviewed
the draft summary and selected the files to apply.

Approval expires at the end of the current Codex run. Prior chat history,
general encouragement, or broad permission to continue does not count.

Current compaction status:

- `2026-05-21`: the read-and-draft approval phrase was provided in the current
  run.
- A private read-only draft was created at
  `/tmp/hermes-memory-compaction-draft-20260521T114154Z`.
- Draft artifacts are private by default (`0700` parent directory and `0600`
  files).
- `2026-05-21`: the apply approval phrase was provided in the current run.
- Owner-only backups were created at
  `/tmp/hermes-memory-compaction-apply-20260521T114955Z`.
- The reviewed draft was applied only to:
  - `~/.hermes/memories/MEMORY.md`
  - `~/.hermes/memories/USER.md`
- Post-apply validation confirmed live files match the reviewed draft, backups
  match the pre-apply bytes, and all live and backup files are `0600`.
- Additional private memory stores remain out of scope and untouched unless a
  separate explicit approval is provided.

### Eligible Scope

Default eligible files after read-and-draft approval:

- `~/.hermes/memories/MEMORY.md`
- `~/.hermes/memories/USER.md`

Additional stores require separate explicit approval:

- `~/.hermes/memory_store.db`
- `~/.hermes/state.db`
- `~/.hermes/sessions/`
- `~/.hermes/response_store.db`
- `~/.hermes/profiles/*`

Out of scope by default:

- provider facts
- logs
- caches
- backups and snapshots
- screenshots
- audio and video
- documents
- credentials, tokens, keys, auth files, and Keychain values

### Safe Future Workflow

1. Re-read this plan, `docs/HERMES_SECURITY_MODEL.md`, and
   `docs/HERMES_TESTING_PLAN.md`.
2. Run metadata-only audit first:

   ```bash
   ./venv/bin/python -m hermes_cli.main memory audit --json --redact
   ./venv/bin/python -m hermes_cli.main memory status
   ```

3. Confirm the user typed
   `APPROVE HERMES PRIVATE MEMORY COMPACTION DRAFT` in the current run.
4. Create private backup and draft artifacts only under owner-only
   permissions (`0700` directories, `0600` files). Keep private diffs out of
   the repo unless they are fully sanitized.
5. Build a draft compaction that:
   - merges duplicate Hermes runtime facts
   - removes stale one-off task details
   - moves procedural material to docs or skills
   - preserves stable user preferences and recovery facts
   - preserves source provenance when useful
   - never adds secrets, raw tokens, private keys, raw chat IDs, or private
     payloads
6. Present only a redacted summary in chat. Do not paste raw private memory
   contents.
7. Wait for the second exact approval:
   `APPLY HERMES PRIVATE MEMORY COMPACTION`.
8. Apply only the reviewed draft for the approved files.
9. Validate with memory audit/status, targeted scans, and rollback check.
10. Update this plan and the build log with applied files, validation, known
    issues, and rollback location. Do not include private content.

### Rollback Requirements

- Before any approved apply step, create an owner-only backup of each target
  file.
- Record the backup path in the build log without printing file contents.
- Confirm rollback can restore the exact pre-apply bytes.
- If validation fails, restore from the private backup or leave the reviewed
  draft unapplied and document the blocker.

## Deletion And Privacy Rules

A forget/delete request must check every layer:

- `~/.hermes/memories/MEMORY.md`
- `~/.hermes/memories/USER.md`
- `~/.hermes/memory_store.db`
- `~/.hermes/state.db`
- `~/.hermes/sessions/`
- `~/.hermes/response_store.db`
- profile memory stores
- logs
- screenshots/media/cache
- checkpoints/snapshots/backups

Deletion should:

- Remove or replace markdown entries.
- Remove structured facts by ID.
- Delete or prune session records.
- Vacuum/checkpoint SQLite stores after large deletes.
- Document what was deleted and what may remain in backups.

## Planned Upgrades

Phase 3 work:

- Add memory capacity report. Implemented as `hermes memory audit`.
- Add markdown-to-holographic reconciliation. Scaffolded as an audit
  requirement; provider-specific mutation remains blocked until approved.
- Add forget-request checklist. Implemented in `hermes memory audit`.
- Add tests for replace/remove mirroring if implementation changes.
- Add privacy-mode recommendations for user-profile writes.

## Non-Negotiables

- Do not store secrets.
- Do not write raw private data unless the user explicitly requests it and it is
  necessary.
- Do not treat memory as current truth for live runtime state.
- Do not silently preserve stale facts after user deletion.
