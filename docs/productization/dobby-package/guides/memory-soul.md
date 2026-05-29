# Memory And SOUL

Reader: an operator configuring Dobby's identity and memory policy. Next
action: decide what the bot may remember, enable consent when ready, and know
how to export, forget, or delete package-owned memory.

## Native Memory Surface

V1 uses native Hermes memory inside the selected `HERMES_HOME`:

- `SOUL.md` defines the bot's durable identity and tone.
- `memories/MEMORY.md` stores concise operational facts.
- `memories/USER.md` stores user preferences and profile notes.
- SessionDB stores conversation history used by `session_search`.

Use a fresh package-owned home. Do not import a personal Hermes home for the
default quickstart.

## SOUL.md

`SOUL.md` is loaded from `HERMES_HOME` only. It should describe the operator
persona, response style, and safe boundaries.

Good content:

```markdown
You are Dobby, a concise Discord operator for ACME staging.
Ask before reading attachments or acting on repositories.
Prefer status, evidence, and rollback steps over speculation.
Never print secrets.
```

Avoid:

- Secrets, tokens, private URLs, or customer records.
- Project-specific instructions that belong in `AGENTS.md`.
- Claims that the bot can access systems that are not configured.

## Consent Model

Default state:

- Volatile session context is allowed.
- Durable memory writes are off until the operator consents.
- Reads are limited to package-owned memory in this `HERMES_HOME`.
- Existing personal sessions are not imported.

Expected command shape:

```text
/memory status
/memory consent on
/memory consent off
/memory export
/memory forget <query-or-id>
/memory delete all
```

Exact command names may follow the final command router, but the capability set
must remain.

## Export

Export should produce a readable archive from package-owned memory only:

- `SOUL.md`
- `memories/MEMORY.md`
- `memories/USER.md`
- Session search metadata or selected transcript summaries
- A manifest with export time, source `HERMES_HOME`, and redaction status

Exports should not include provider API keys, Discord tokens, webhook secrets,
logs, or unrelated home directories.

## Forget And Delete

Use targeted forget when a single memory is wrong or no longer consented:

```text
/memory forget "ACME uses staging region west"
```

Use delete-all only for package-owned memory reset. It should require explicit
confirmation and should not delete unrelated host data.

Rollback does not delete memory by default. Incident response should preserve
redacted evidence first, then delete only when the operator explicitly chooses
privacy cleanup.

## session_search

`session_search` recalls past package conversations from SessionDB. Treat it as
on-demand recall, not always-in-context memory.

Safe use:

- Ask it to find a prior staging incident summary.
- Use it to recover the last rollback checklist.
- Export or delete its package-owned data through the memory controls.

Unsafe use:

- Importing another user's historical sessions.
- Searching personal Hermes sessions from outside this package home.
- Treating recalled content as current without checking freshness.

## Honcho And Mem0 PoC Caveats

Honcho and Mem0 are optional external proof-of-concept integrations. They are
not part of the default V1 install path.

Before enabling either one, require a separate design review covering:

- What data leaves the package home.
- Consent, export, forget, and delete behavior.
- Provider credentials and retention.
- Failure behavior if the provider is unavailable.
- How to disable the provider without breaking native memory.

Native memory must remain usable without these integrations.
