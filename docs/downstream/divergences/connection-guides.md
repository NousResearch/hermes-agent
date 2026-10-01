# Connection guides

- Status: active
- Scope: `guides/connections`, `agent/employee_prompt.py`, guide packaging
- Introduced: PR #2

## Downstream intent

Skills are disabled, but service setup must remain discoverable. The connection
pointer opens a top-level guide, native-derived Google/email/GitHub subguides,
and the native MCP reference. Other services use official documentation without
requiring another guide. Service manuals hold verified access/usage patterns,
not copied setup instructions or reusable secrets.

## Reconciliation

Absorb native setup/helper improvements; preserve guide discovery and manual
rules. Keep Google helpers byte-identical to their native source unless a
specific runtime mismatch requires an explicit change. Do not add auth wrappers
or restore skill discovery. Retain native credential/configuration locations.

## Validation

Run employee-knowledge, system-prompt and connection-guide tests. Verify packaged
helpers execute outside the source tree. Compare native helper copies with Git.
See [guide contracts](../../specs/guides.md).
