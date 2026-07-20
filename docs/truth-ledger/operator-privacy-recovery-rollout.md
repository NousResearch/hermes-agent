# Truth Ledger operator, privacy, recovery, and rollout runbook (T16)

Date: 2026-07-19
Scope: current implementation in `plugins/truth-ledger/` on branch `feat/truth-ledger-option-2`.

## 1) What this build does today (and does not do)

Implemented now:
- Opt-in plugin registration via `plugins/truth-ledger/plugin.yaml` and `register()` hooks in `plugins/truth-ledger/__init__.py`.
- Hook coverage:
  - `post_llm_call` capture path: eligible-turn check -> source envelope build -> sanitize -> schema validate -> durable spool enqueue.
  - `on_session_start` recovery path: stale `spool/processing` records are moved back to `pending` or quarantined.
- Operator slash command: `/truth-ledger status|review|rebuild|retract|export` via `plugins/truth-ledger/commands.py`.
- Safety defaults:
  - `rebuild`, `retract`, `export` are dry-run unless `--apply` is present.
  - `retract` is append-only (new `retract` event), never history rewrite.

Not implemented yet (important for rollout expectations):
- No bounded runtime drain from `spool/pending` into extraction/reconciliation/ledger during normal hook lifecycle.
- No automatic promotion into curated memory (`USER.md`, `MEMORY.md`) or GBrain.
- No automatic retention cleanup or hard-delete workflow.
- No public config surface for truth-ledger thresholds/overrides yet.

Implemented stability fixes from the final reduced-MVP acceptance pass:
- In-process envelope dedupe is lock-protected and FIFO-bounded to 1,024 keys.
- Dead-letter review resolves the canonical nested `flow.dead_letter_reason` field.

## 2) Install and opt-in

Truth Ledger is bundled, but disabled by default.

Enable plugin (current profile):
- `hermes plugins enable truth-ledger`

If your runtime does not yet include this plugin as bundled, install it as a user plugin first:
- `mkdir -p ${HERMES_HOME}/plugins`
- `cp -R /path/to/hermes-agent/plugins/truth-ledger ${HERMES_HOME}/plugins/truth-ledger`
- then run `hermes plugins enable truth-ledger`

Verify plugin enablement:
- `hermes plugins list | grep -i truth-ledger`
- In a Hermes session, run `/truth-ledger status --json`.

Disable plugin:
- `hermes plugins disable truth-ledger`

Behavior after disable:
- New capture stops immediately.
- Existing data under `${HERMES_HOME}/truth-ledger/` is preserved (no deletion).

## 3) Configuration and provider behavior

Current config surface:
- Opt-in gate only (`plugins.enabled` / `plugins.disabled`).
- Storage root resolved from `get_hermes_home()` -> `${HERMES_HOME}/truth-ledger`.

Provider/model behavior:
- The extractor module defines optional override fields (`override_mode`, `provider_override`, `model_override`) in `ExtractorSettings`.
- No end-user truth-ledger config wiring is exposed yet for these settings.
- Operator should treat provider selection for extraction as implementation-internal until a config contract lands.

## 4) Data layout, ownership, and permissions

Current on-disk layout from `TruthSpool` + commands/projection:

`${HERMES_HOME}/truth-ledger/`
- `spool/pending/*.json` (spool records)
- `spool/processing/*.json` (claimed records)
- `spool/dead-letter/*.json` (dead-lettered spool records)
- `spool/payloads/*.json` (source envelopes referenced by spool records)
- `ledger/YYYY-MM.jsonl` (append-only lifecycle events)
- `views/current.jsonl` (derived active-state projection)
- `views/review.jsonl` (optional review queue file if present)
- `state/index.sqlite` (derived index if present)
- `backups/current-*.jsonl` (created by `rebuild --apply` when prior `current.jsonl` exists)
- `exports/truth-ledger-export-*.tar.gz` (created by `export --apply`)
- `errors/errors.jsonl` and corrupt-tail quarantine files (projection recovery)

Permissions:
- Directories attempt `0700`.
- Files attempt `0600`.
- Applied best-effort in spool/projection/commands codepaths.

Ownership model:
- Canonical history: `ledger/*.jsonl` (append-only).
- Derived/rebuildable: `views/current.jsonl`, `state/index.sqlite`.
- Ephemeral queue: `spool/pending`, `spool/processing`.

## 5) Eligibility and omission gates

`on_post_llm_call` eligibility requires all of:
- `completed=True`
- `failed=False`
- `interrupted=False`
- `turn_exit_reason` starts with `text_response(`
- not a kanban worker turn (`kanban_task_id` absent)
- not subagent (`is_subagent=False`, `delegate_depth==0`)
- non-empty `session_id` and `turn_id`

Omissions by design in current capture:
- No raw `conversation_history` field persisted to source envelopes.
- No direct persistence of chain-of-thought fields.
- If eligibility fails, capture is skipped.

Current dedupe behavior:
- In-process dedupe `_SEEN_ENVELOPES` is lock-protected and FIFO-bounded to 1,024 envelope keys.
- Durable spool/ledger idempotency remains the cross-process/restart safety boundary.

## 6) Privacy, redaction, and do-not-remember boundaries

Privacy controls present:
- Source envelope is sanitized with `sanitize_payload(...)` before spool write.
- Envelope and spool records are schema-validated before acceptance.
- Extractor input blocks are rebuilt from minimal sanitized envelope subset.
- Error text is redacted/truncated in dead-letter payload generation paths.

Do-not-remember boundaries:
- Plugin does not write `USER.md`, `MEMORY.md`, or GBrain.
- Promotion is manual/review-gated outside this plugin.
- This runbook should be read as "capture + operator tooling," not "auto-memory sync."

## 7) Operator commands and safe usage

All command examples are slash commands in a Hermes session:

1) Status
- `/truth-ledger status`
- `/truth-ledger status --json`

2) Review queue and dead letters
- `/truth-ledger review`
- `/truth-ledger review --limit 50 --json`

3) Rebuild current projection
- Dry-run: `/truth-ledger rebuild --json`
- Apply: `/truth-ledger rebuild --apply --json`
- Apply mode creates backup of existing `views/current.jsonl` before replacement.

4) Retract a fact (append-only)
- Dry-run: `/truth-ledger retract fact_<id> --json`
- Apply: `/truth-ledger retract fact_<id> --apply --json`
- Invalid fact IDs are rejected (`invalid_fact_id`).

5) Export snapshot bundle
- Dry-run: `/truth-ledger export --json`
- Apply default local path: `/truth-ledger export --apply --json`
- Apply explicit local path: `/truth-ledger export --apply --destination /absolute/path/export.tar.gz --json`
- Current implementation caveat: destination validation converts to `Path(...)` before checking for `://`. URI-like values may normalize to local path syntax. Use explicit absolute local paths only.

## 8) Backup, export, recovery, and rebuild playbooks

Backup/export (operator-safe):
1. Run dry-run export:
   - `/truth-ledger export --json`
2. If includes/path are correct, run apply:
   - `/truth-ledger export --apply --json`
3. Record returned `sha256` and `size_bytes`.

Rebuild current projection:
1. Run dry-run rebuild and inspect `before_sha256`.
2. Run apply rebuild.
3. Confirm `after_sha256` changed as expected and `backup_path` exists.

Stale processing recovery:
- Automatic recovery runs on each `on_session_start` via `recover_stale_processing(stale_seconds=900)`.
- If queue appears stuck, restarting Hermes session triggers the recovery hook.

Corrupt ledger tail handling during projection rebuild:
- Projection reader quarantines malformed terminal fragments into `errors/projection-corrupt-tail-*.jsonl` and rebuilds from valid prefix.

## 9) Retention and deletion policy (current state)

Current behavior:
- No scheduled retention cleanup.
- No automatic hard delete.
- Dead-letter spool records persist until explicit operator cleanup.
- Payload file is deleted when a record is acked or dead-lettered through spool flows.

Operator implication:
- Treat retention/deletion as explicit governance work, not implicit runtime behavior.

## 10) Rollout phases and promotion gate

Recommended low-risk rollout:
1. Enable plugin in one canary profile.
2. Verify `status` and `review` outputs only.
3. Exercise `rebuild --json` and `export --json` dry-runs before any apply actions.
4. Run one controlled `export --apply` and validate tarball hash/permissions.
5. Keep promotion to curated memory/GBrain disabled pending separate review gate.

Promotion gate (must remain explicit):
- No automatic path from truth-ledger events to `USER.md` / `MEMORY.md` / GBrain.
- Any future promotion workflow must be separately approved, tested, and documented.

## 11) Disable and uninstall

Disable only (recommended):
- `hermes plugins disable truth-ledger`
- Leaves data intact for later audit/re-enable.

Uninstall guidance:
- For bundled plugin in this repo: operationally prefer disable over file removal.
- For user-installed plugin copies under `${HERMES_HOME}/plugins/...`: disable first, then remove plugin directory only after confirmed export/backup.

## 12) Troubleshooting

Plugin appears missing:
- Check opt-in state (`hermes plugins list`, `hermes plugins enable truth-ledger`).
- Use plugin debug listing:
  - `HERMES_PLUGINS_DEBUG=1 hermes plugins list`

`/truth-ledger` command unavailable:
- Verify plugin is enabled in the active profile.
- Verify session is running with that profile.

`review` shows dead-letter reason `unknown`:
- The current implementation checks `flow.dead_letter_reason`; `unknown` now indicates a legacy or malformed record without a recognized reason field.

In-process dedupe reaches 1,024 entries:
- This is the expected FIFO bound. Durable spool/ledger idempotency remains the cross-process and restart boundary after older keys are evicted.

Queue growth without ledger activity:
- Expected with current implementation gap: runtime hooks enqueue/recover but do not yet drain pending envelopes into extraction/reconciliation/ledger automatically.

Permission issues writing under truth-ledger root:
- Confirm profile `HERMES_HOME` location and filesystem ownership.
- Check parent directory permissions and that plugin can create `0700/0600` artifacts.

## 13) Verification checklist for a fresh operator

- [ ] Plugin enabled via `hermes plugins enable truth-ledger`.
- [ ] `/truth-ledger status --json` returns `ok: true`.
- [ ] `/truth-ledger export --json` dry-run returns expected local path/includes.
- [ ] `/truth-ledger rebuild --json` dry-run returns `before_sha256` field.
- [ ] `/truth-ledger retract <fact_id> --json` dry-run succeeds only for valid fact id pattern.
- [ ] Disable path works and stops new capture (`hermes plugins disable truth-ledger`).
- [ ] No direct writes to curated memory/GBrain are observed.
