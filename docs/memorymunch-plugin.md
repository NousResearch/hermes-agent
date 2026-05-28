# MemoryMunch Hermes Plugin

MemoryMunch is vendored in this repository at `contrib/plugins/memorymunch` so the
runtime plugin can be reviewed, tested, committed, tagged, and restored after
Hermes updates.

## Known-good state

This commit line treats the vendored plugin as a good working MemoryMunch state:

- Curator and Gateway are prompt-facing lanes.
- Capture and Janitor are post-turn background lanes for speed.
- Live capture writes DB + vault.
- Janitor live mutation is enabled and reports archive outcomes explicitly.
- Runtime telemetry is split into three categories:
  1. prompt lanes: Curator/Gateway in-turn;
  2. background write lanes: Capture/Janitor post-turn;
  3. status/reporting surfaces: checker, proof telemetry, and ledger events.

## Telemetry truth rules

Do not read `turn_completed.live_db_write=false` as final write truth. That row is
local turn ledger metadata. Final write truth comes from later events:

- Capture: `live_capture_completed` with `live_db_write=true` and
  `live_vault_write=true`.
- Janitor: `janitor_cycle_completed` with `status=APPLIED` and the archive fields
  below.

Janitor archive telemetry has explicit outcomes:

- `vault_moved_true` — a matching Obsidian vault atom file moved to `_archived`.
- `no_archive_actions_requested` — Janitor applied/reviewed but had no vault move
  request for that cycle.
- `no_vault_file_to_move` — DB/edge cleanup applied but no matching vault file was
  present.
- `vault_move_failed` — the vault move itself failed.

## Session-bleed guard / current intent rule

Canonical compact line:

```text
current_session=yes|no; current_intent only from live_user_msg in active_session_id; no=history_only, never live intent/task/state
```

Gateway rule: old atoms can inform background evidence only. Current intent/task/state is never inferred from another session, even when that old atom came from a live user message in its own original session.

## Runtime and vendored paths

- Runtime plugin: `/home/alcoo/.hermes/plugins/memorymunch`
- Vendored plugin: `/home/alcoo/.hermes/hermes-agent/contrib/plugins/memorymunch`
- Runtime status checker: `/home/alcoo/.hermes/scripts/memorymunch_operational_status.py`
- Vendored status checker: `/home/alcoo/.hermes/hermes-agent/scripts/memorymunch_operational_status.py`

## Verification

From the repository root:

```bash
python -m py_compile contrib/plugins/memorymunch/__init__.py scripts/memorymunch_operational_status.py
python -m pytest tests/scripts/test_memorymunch_operational_status.py tests/run_agent/test_memorymunch_openclaw_parity.py -q -o 'addopts='
python scripts/memorymunch_operational_status.py
```

The operational checker may return non-zero when unrelated fleet gates are stale,
but its MemoryMunch fields must show:

- runtime plugin matches vendored plugin;
- hardwired live writes are present;
- three-lane telemetry is present;
- latest capture and Janitor truth is read from post-turn events.
