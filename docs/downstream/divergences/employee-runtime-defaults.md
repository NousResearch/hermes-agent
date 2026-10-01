# Employee runtime defaults

- Status: active
- Scope: native config defaults, gateway display/config, compression setup, STT selection
- Introduced: employee runtime defaults implementation

## Downstream intent

Default to quiet Telegram delivery, medium reasoning, 85% compression with three
real user messages retained, approvals off, no restart/transcript echoes, and
local transcription. Explicit operator preferences still win. An unrelated cloud
key must never make the default local transcription upload audio.

## Reconciliation

Preserve these defaults when absorbing upstream changes to config readers,
setup, examples and gateway tiers. Keep the native presence-sensitive loader,
Telegram adapter and approval/compaction mechanics. Do not restore the STT
exception that treats a default local selection as cloud autodetection.

## Validation

Run `tests/hermes_cli/test_employee_runtime_defaults.py`, gateway display/config,
compression-default and transcription tests through `scripts/run_tests.sh`.
Exercise defaults and explicit overrides across profiles A → B → A.
