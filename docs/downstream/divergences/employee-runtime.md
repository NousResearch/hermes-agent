# Employee runtime

- Status: active
- Scope: agent prompts/context/review, tool resolution, responsibilities, memory, messaging and deployment
- Introduced: employee implementation

## Downstream intent

Preserve the [employee contracts](../../specs/employee.md): employee wording and
guides, fixed tool surface, file-owned responsibilities, per-person authored
memory after the current message, Hindsight background memory, and review that
consolidates knowledge. Native adapters, delegation, steering and administration
remain the owners. Confirmed outbound context enters only at the next turn.

## Reconciliation

Retain native fixes and merge these integration points directly. Do not restore
skills or alternate memory/browser surfaces through configuration. Preserve
cached prompts and exact historical API sidecars, including multimodal messages.
Keep Hindsight's pinned policy and Codex authentication route; hosted billing and
credential gateways do not belong here. Absorb equivalent upstream behavior
instead of duplicating it.

## Validation

Run employee tests under `tests/agent`, `tests/responsibilities`,
`tests/plugins`, and `tests/deploy` through the native test runner.
[Deployment acceptance](../../../deploy/railway/README.md) requires a future server.

## Platform scope

The employee runtime targets the specified Linux deployment and macOS development;
WSL2 supplies the Linux runtime on Windows. Its copied responsibility filesystem
uses POSIX directory handles and does not support native Windows. The root README
marks retained upstream Windows instructions as upstream reference. Porting that
filesystem requires separate native Windows implementation and acceptance testing.
