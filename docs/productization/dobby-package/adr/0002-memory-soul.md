# ADR 0002: Native Memory, SOUL.md, and Consent

Status: Accepted for Stage 3 foundation.

## Context

Hermes already has native memory surfaces: `SOUL.md`, built-in memory files,
SessionDB, and the `session_search` tool. The product must claim memory while
remaining self-hosted and not bundling a Honcho server.

## Decision

The package will use native Hermes memory by default:

- `SOUL.md` is the editable operator-facing identity file.
- `memories/MEMORY.md` and `memories/USER.md` hold built-in durable notes.
- SessionDB stores conversation history for `session_search`.
- Durable memory writes require explicit consent in the package onboarding.
- Memory commands must include status, consent, export, forget, and delete.
- Honcho remains an optional external plugin outside the default package path.

## Consent Model

Initial state should be conservative:

- Volatile session context is allowed.
- Durable memory write is disabled until the operator consents.
- Reads from existing package-owned memory are allowed only inside the selected
  `HERMES_HOME`.
- Importing an existing Hermes home is out of scope for the default quickstart.

Expected command shape for later implementation:

```text
/memory status
/memory consent on
/memory consent off
/memory export
/memory forget <query-or-id>
/memory delete all
```

Exact command names may change if the existing command system has a stronger
convention, but the capability set and evidence gates should remain.

## Evidence Gates

- Temp-home tests prove consent off prevents durable memory writes.
- Export produces a readable bundle from only package-owned memory surfaces.
- Forget can remove a targeted memory without deleting unrelated state.
- Delete all removes package-owned memory and session-search data after an
  explicit confirmation token.
- No test requires Honcho, cloud memory, or copied personal runtime state.
