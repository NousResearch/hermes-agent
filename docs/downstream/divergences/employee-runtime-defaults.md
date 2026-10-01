# Employee runtime defaults

- Status: active
- Scope: native config defaults, gateway display/config, compression setup, STT selection
- Introduced: employee runtime defaults implementation

## Downstream intent

Default to quiet Telegram delivery, medium reasoning, 85% compression with three
real user messages retained, execution approvals and session/MCP confirmations
off, no restart/transcript echoes, unwrapped cron deliveries, and
local transcription. Explicit operator preferences still win. An unrelated cloud
key must never make the default local transcription upload audio.

## Reconciliation

Preserve these defaults when absorbing upstream changes to config readers,
setup, examples and gateway tiers. Keep the native presence-sensitive loader,
Telegram adapter and approval/compaction mechanics. Do not restore the STT
exception that treats a default local selection as cloud autodetection.
Keep missing confirmation settings off in raw gateway readers and client defaults;
explicit opt-ins still prompt. Explicit MCP reloads retain native cache invalidation.

## Validation

Run `tests/hermes_cli/test_employee_runtime_defaults.py`, cron delivery wrapping, gateway display/config,
compression-default and transcription tests through `scripts/run_tests.sh`.
Exercise defaults and explicit overrides across profiles A → B → A.
