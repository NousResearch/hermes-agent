# Receipt Sorter Automation Agent Build Handoff

## Mission

Build an automation script that scans a local folder of receipt PDFs/images, extracts basic metadata, renames files consistently, and writes a CSV index.

## Product Vision

Turn a messy downloads folder into an organized, searchable receipt archive without requiring a full document-management system.

## User Experience Goals

- User points the script at an input folder.
- Script processes supported files safely.
- Original files are not destroyed.
- Output names and CSV are predictable.

## Non-Negotiable Requirements

- Dry-run mode before modifying/copying files.
- Never delete originals.
- Produce a CSV index.
- Handle extraction failures gracefully.
- Log what happened.

## Out of Scope

- Cloud sync.
- Accounting software integration.
- Perfect OCR accuracy.
- Multi-user workflow.

## Technical Architecture

- CLI/script with arguments for input folder, output folder, and dry-run.
- File discovery layer.
- Metadata extraction layer using available PDF/OCR tools in the repo/environment.
- Rename/copy layer.
- CSV writer and log output.

## Data Model

Receipt index row:
- `original_path`
- `new_path`
- `vendor`
- `date`
- `amount`
- `status`
- `notes`

## Integrations

- Local filesystem.
- Optional OCR/PDF library depending on environment.

## Implementation Phases

1. Inspect repo and available dependencies.
2. Implement file discovery and dry-run reporting.
3. Implement best-effort metadata extraction.
4. Implement safe copy/rename.
5. Implement CSV index and logging.
6. Add tests with sample fixture files.

## Build Tasks

- Add tests for filename generation.
- Add tests for CSV writing.
- Add tests for dry-run no-write behavior.
- Implement script arguments.
- Implement processing pipeline.
- Add README usage examples.

## Testing Requirements

- Unit tests for parsing and filename sanitization.
- Tests proving dry-run does not create/copy output files.
- Tests for duplicate filename handling.
- Fixture-based test for at least one sample input.

## Verification Commands / Checks

- Run test suite.
- Run dry-run on sample folder.
- Run real mode on copied sample folder.
- Confirm originals remain unchanged.
- Confirm CSV contains one row per attempted file.

## Acceptance Criteria

- User can run dry-run and preview actions.
- User can run real mode and get organized copies.
- CSV index is created.
- Failed extractions are recorded, not fatal.

## Done Means

- Tests pass.
- Dry-run behavior is proven.
- Manual sample run works.
- README documents usage and safety behavior.

## Known Risks

- OCR quality varies.
- Vendor/date/amount parsing should be best-effort for v1.

## Open Questions

- Should organized files be copied or moved by default? V1 should copy.
- What filename format should be default: date-vendor-amount or vendor-date-amount?

## Prompt for Build Agent

Inspect the repository and dependencies first. Plan a safe implementation with dry-run tests before write behavior. Do not implement destructive file operations. Verify with fixture data before claiming done.

## Superpowers Build Handoff

Use the `superpowers-gpt` workflow on this document. Start with `superpowers-using-superpowers`, then `superpowers-writing-plans`, then execute, review, and verify with the appropriate Superpowers skills.
