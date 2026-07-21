# Wave 2 — Evidence database health scope

## Finding

Do not add a `hermes doctor` check in this bounded change. The evidence database
is already captured through WAL-safe full backup and quick snapshot paths. The
existing doctor database repair is specific to a historical `state.db` FTS
write-corruption incident; no recovery action is defined for the evidence DB.
A generic `quick_check` would not establish receipt immutability, claim
ownership, or decision idempotency.

## Closure

The correct evidence is behavior-level SQLite and command-path tests. Revisit a
read-only doctor warning only after an actual evidence-DB corruption incident
or when the ledger becomes an approval execution source of truth.

## EXPAND

none — doctor, backup, runtime, tests, and history were covered; no justified
health-scope change remains.
