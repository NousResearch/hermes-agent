# google_drive_pipeline

CLI-backed Google Drive artifact publishing on top of the existing
`google-workspace` skill primitives.

This plugin does not add new Google API capabilities by itself. Instead, it
turns the lower-level Drive commands from `google_api.py` into a higher-level
workflow for publishing artifacts:

- resolve a target folder
- create the folder if missing
- upload a local file
- optionally apply sharing
- return a canonical Drive link
- persist publish records for later inspection/reuse

## Why this exists

`feat(google-workspace): Drive write ops + Docs/Sheets create/append` added the
raw Drive write commands:

- `drive search`
- `drive get`
- `drive upload`
- `drive create-folder`
- `drive share`

Those commands are the toolbox. This plugin is the operator workflow layer on
top of that toolbox.

## Commands

```bash
hermes google-drive-pipeline resolve-folder --folder-name Reports

hermes google-drive-pipeline publish ./report.pdf \
  --folder-name Reports \
  --create-missing-folder

hermes google-drive-pipeline publish ./report.pdf \
  --folder-name Reports \
  --duplicate-policy reuse

hermes google-drive-pipeline publish ./report.pdf \
  --folder-name Reports \
  --share-type anyone \
  --share-role reader

hermes google-drive-pipeline list
hermes google-drive-pipeline show RECORD_ID
```

## Duplicate policies

- `fail`: error if a same-named file already exists in the target folder
- `reuse`: reuse an existing matching publish record or same-named Drive file
- `version`: upload with a timestamped filename when a conflict exists

## Store

The plugin stores publish records in:

```text
$HERMES_HOME/google_drive_pipeline_store.json
```

It keeps:

- publish records
- source-key index for `reuse`
- folder cache for resolved folders

## Requirements

- `google_drive_pipeline` must be enabled in `plugins.enabled`
- Google Workspace auth must already exist in the current Hermes profile
- The underlying `google-workspace` setup/token is still the source of truth

## Out of scope

- Gmail / Calendar orchestration
- Docs / Sheets publish flows
- background scheduling
- webhook/event ingestion

This plugin is intentionally narrow: Drive artifact logistics only.
