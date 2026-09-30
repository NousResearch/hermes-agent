# Employee runtime

- Status: active
- Scope: prompts, context, review, tools, responsibilities, memory, messaging and deployment
- Introduced: employee implementation

## Downstream intent

Preserve [employee contracts](../../specs/employee.md): fixed tools, responsibility
packages, per-person memory, Hindsight, service manuals and knowledge review.
Native adapters, delegation, scratch/cache and administration remain owners.
Confirmed deliveries enter the next turn.

## Reconciliation

Port native Hermes self-reference as a guide with the native prompt pointer;
preserve [runtime adaptations](../../specs/guides.md) and connection documentation.
Do not restore excluded surfaces or hosted credential gateways. Preserve prompt
caches, replay bytes, profile scope (including worker threads), Codex auth and
Hindsight policy. Messaging needs a leaf toolset for native child restrictions.
Absorb equivalent upstream behavior directly.

## Validation

Run employee tests in `tests/agent`, `tests/responsibilities`, `tests/plugins`
and `tests/deploy` through the native runner. [Deployment acceptance](../../../deploy/railway/README.md)
requires a server. Linux/WSL2 and macOS are supported; responsibility filesystem
operations require POSIX. Native Windows needs separate implementation and tests.
