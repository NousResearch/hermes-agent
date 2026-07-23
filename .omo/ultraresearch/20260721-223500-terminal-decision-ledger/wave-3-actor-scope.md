# Wave 3 — Decision attribution scope

## Finding

The shared approval handler has four direct CLI/gateway producers; TUI and
desktop route through the CLI path. No shared, trustworthy actor/session
identity is supplied at the handler boundary. Runtime session keys are neither
verified user identity nor valid across every producer.

## Decision

Receipt rows are profile-scoped and unattributed. They record only subsystem,
pending ID, proposal digest, terminal decision/outcome, safe code, and time.
The profile-scoped database path supplies the honest isolation boundary.

## Closure

Do not infer or persist session/user/actor data. Per-user attribution requires
a future explicit `decision_context` contract across CLI, gateway, TUI, and
desktop, with group/thread semantics verified end-to-end.

## EXPAND

none — automatic actor/session inference is a dead end under the current
shared-handler contract.
