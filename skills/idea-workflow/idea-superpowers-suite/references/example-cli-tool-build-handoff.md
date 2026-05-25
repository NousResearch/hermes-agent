# DevLog CLI Agent Build Handoff

## Mission

Build a small command-line tool that lets a developer keep a local timestamped work log from the terminal.

## Product Vision

A fast, boring, reliable CLI for capturing short development notes without opening a project-management app.

## User Experience Goals

- Add a note in one command.
- List recent notes quickly.
- Search notes by text.
- Store data locally in a human-readable file.

## Non-Negotiable Requirements

- Works offline.
- Uses local storage only.
- Has `add`, `list`, and `search` commands.
- Includes automated tests for command behavior.

## Out of Scope

- Sync.
- Accounts.
- Web UI.
- Team sharing.

## Technical Architecture

- CLI app in the target repo's primary language.
- Local JSONL or SQLite storage.
- Thin command parser layer over a small storage/service module.

## Data Model

Log entry:
- `id`
- `timestamp`
- `project` optional
- `text`
- `tags` optional

## Integrations

None for v1.

## Implementation Phases

1. Create CLI skeleton and storage module.
2. Implement `add`.
3. Implement `list` with limit option.
4. Implement `search`.
5. Add tests and docs.

## Build Tasks

- Inspect existing repo conventions.
- Add storage tests first.
- Add command tests for `add`, `list`, and `search`.
- Implement minimal behavior.
- Document examples in README.

## Testing Requirements

- Unit tests for storage read/write.
- CLI tests for successful commands.
- Test missing/empty log file behavior.

## Verification Commands / Checks

- Run project test suite.
- Manually run: `devlog add "fixed auth bug"`.
- Manually run: `devlog list`.
- Manually run: `devlog search auth`.

## Acceptance Criteria

- User can add a note from terminal.
- User can list recent notes.
- User can search existing notes.
- Data persists across runs.

## Done Means

- Tests pass.
- Manual commands work.
- README includes usage examples.
- No network or account requirement exists.

## Known Risks

- File path conventions may differ by OS.
- Search can be simple substring search for v1.

## Open Questions

- Should default storage live in project directory or user config directory?

## Prompt for Build Agent

Inspect the repository first. Create a concise implementation plan with files, tests, commands, and expected results. Implement only the v1 CLI described here. Run automated and manual verification before claiming done.

## Superpowers Build Handoff

Use the `superpowers-gpt` workflow on this document. Start with `superpowers-using-superpowers`, then `superpowers-writing-plans`, then execute, review, and verify with the appropriate Superpowers skills.
