# Scoped agent guidance

- Status: active
- Scope: `AGENTS.md`, nested `AGENTS.md` files, `ARCHITECTURE.md`, and `docs/`
- Introduced: scoped agent documentation reorganization

## Downstream intent

Root `AGENTS.md` stays compact: global rules and required-reading routes only.
Durable detail belongs in `ARCHITECTURE.md` or focused `docs/`; subtree-only
rules may use a nested `AGENTS.md`. Upstream can add large sections to root, so
taking either whole file would erase downstream structure or new upstream rules.

## Reconciliation

Route each new upstream rule to the narrowest authoritative location. Preserve
downstream product framing, operating posture, checkout/review rules, and doc
routes. Do not resolve this scope wholesale with ours or theirs.

## Validation

- Account for each substantive upstream rule.
- Confirm root guidance stays concise and its links resolve.
- Confirm downstream-only rules remain intact.
