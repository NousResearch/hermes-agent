# Employee runtime

- Status: active
- Scope: prompts, context, review, tools, responsibilities, memory, messaging and deployment
- Introduced: employee implementation

## Downstream intent

Preserve [employee contracts](../../specs/employee.md): employee wording and guides,
fixed tools, file-owned responsibilities, per-person memory after the current
message, Hindsight, and knowledge consolidation. Native adapters, delegation,
steering, scratch/cache handling and administration remain owners. Confirmed
deliveries enter next turn.

## Reconciliation

Keep integration points direct. Do not restore skills, alternate memory/browser
surfaces, hosted billing or credential gateways. Preserve cached prompts,
historical API sidecars, Hindsight policy and Codex authentication. Employee prompt
assembly follows native agent-home precedence, including bare worker threads.
Messaging must retain a leaf toolset so native delegation restrictions apply.
Absorb equivalent upstream behavior rather than duplicating it.

## Validation

Run employee tests in `tests/agent`, `tests/responsibilities`, `tests/plugins`,
and `tests/deploy` through the native runner. [Deployment acceptance](../../../deploy/railway/README.md)
requires a future server.

## Platform scope

Linux deployment and macOS development are supported. Responsibility filesystem
operations require POSIX; Windows uses WSL2. Native Windows needs separate
implementation and acceptance testing. Retained Windows instructions are upstream reference.
