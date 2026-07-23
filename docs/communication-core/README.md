# Communication Core

Communication Core is the account-explicit domain for personal communication
in Hermes. It is reached through `hermes communication` and the
`manage-communications` skill; it does not add a model tool or mutate the
conversation prompt/tool prefix.

References:

- [Architecture and ownership](architecture.md)
- [Schema and entities](schema.md)
- [Account isolation and routing](isolation-routing.md)
- [Adapter capability matrix](adapters.md)
- [Configuration](configuration.md)
- [Facebook and legacy migration](facebook-migration.md)
- [CLI and skill usage](usage.md)
- [Privacy and approval policy](privacy-safety.md)
- [Operations runbook](operations.md)
- [Rollback](rollback.md)
- [Requirement-to-evidence matrix](requirements-evidence.md)

Production sending is not part of this completion scope. Facebook write
actions and production outbox workers remain disabled. The only executable
delivery adapter is the in-memory fake test sink.
