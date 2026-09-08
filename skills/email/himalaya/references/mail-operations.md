# Mail operations

Examples target v2.1.0. Contents: identity and output; search; mailbox mutations; flags; attachments.

## Identity, read state and output

Keep the tuple (account, backend, mailbox, message ID). IDs are opaque backend identifiers; IMAP UIDs are mailbox-scoped, while other backends differ. They are not the list's row index or the RFC Message-ID header. Re-list after moving messages, changing backend/account, or other operations that can invalidate the selection.

```bash
himalaya --account work --json mailbox list --counts
himalaya --account work --json envelope list --mailbox inbox --page 1 --page-size 25
himalaya --account work message read --mailbox inbox 42
himalaya --account work message read --mailbox inbox --seen 42
himalaya --account work message read --mailbox inbox --raw 42 > source.eml
```

V2.1 read is non-marking by default; `--seen` changes that. Default output includes headers and MIME-part summaries with plain text, not a browser rendering. For faithful export or external rendering, use `--raw` without JSON. With both `--raw --json`, expect a structured message string, not bare MIME bytes.

Use `--json` for machine output. Generate schemas into a new directory with `himalaya json-schema ./schemas` when integrating a parser. Inspect actual stdout, stderr and exit status. Do not assume all results share one shape or that an error means no partial work occurred. In v2.1, envelope lists/searches wrap rows in `envelopes`, attachment lists/downloads wrap rows in `attachments`, and message add returns `id` and `sent`; these are not bare arrays.

## Search and pagination

V2 filtering and sorting belong to `envelope search`; `envelope list` lists newest first. Page numbering begins at 1 and the default size falls back to 25. Set page size explicitly for bounded retrieval; paginate until the requested scope is satisfied. Mailbox updates can change page boundaries, so deduplicate by scoped ID during long scans.

```bash
himalaya --account work --json envelope search --mailbox inbox \
  --page 1 --page-size 50 \
  '(from alice@example.org or from bob@example.org) and not flag seen order by date desc'
himalaya --account work --json envelope search --mailbox inbox \
  'after 2026-09-01 and subject invoice order by date asc'
```

Shared conditions: `date YYYY-MM-DD`, `after YYYY-MM-DD`, `from PATTERN`, `to PATTERN`, `subject PATTERN`, `body PATTERN`, `flag seen|answered|flagged|draft`. Combine with `and`, `or`, `not` and parentheses. Sort using `order by date|from|to|subject [asc|desc]` with multiple sort keys if needed.

Quote the whole query for the shell; keep options before the trailing query. Use explicit Boolean operators, unlike the original adjacent-condition example. Text matching is case-insensitive substring matching; dates target the message's Date header. V2.1 has no shared `before` clause. Check exact boundary semantics and backend support for date-sensitive work; do not silently relax filters when one is unsupported.

`--has-attachment` on list/search populates attachment information; it is not a filter that returns only messages with attachments. `--recipient` displays To instead of From. Native IMAP/JMAP/Gmail/Graph searches expose additional features and different grammars; see capabilities.

## Move, copy and delete

```bash
himalaya --account work message move --from inbox --to archive 42 43
himalaya --account work message copy --from inbox --to important 42
himalaya --account work --json message delete --mailbox inbox 42
```

Resolve destinations before executing. Shared move/copy is same-account, same-backend; do not interpret it as a cross-account transfer. A separate export/import operation is needed for such a task, with its own validation and no implicit source deletion.

V2.1 shared delete resolves the backend's trash role, then the `trash` alias. Outside trash it moves there. Inside trash it requests permanent removal. IMAP without UIDPLUS may only flag the messages deleted; later expunge reclaims them. Inspect the JSON `action` and `count`, not just a generic success line.

Native deletion has different scope: deleting a mailbox, Gmail permanent deletion, JMAP destruction and IMAP expunge are not interchangeable with trashing a message. Broad IMAP expunge/close can remove other previously deleted messages. Use only the scope the user authorized.

Mailbox creation, renaming, subscription and removal are native operations in v2; `mailbox list` is the only shared mailbox subcommand.

## Flags

```bash
himalaya --account work flag add --mailbox inbox --flag seen --flag flagged 42 43
himalaya --account work flag remove --mailbox inbox --flag flagged 42
himalaya --account work flag set --mailbox inbox --flag seen 42
```

Use add/remove for incremental changes. `set` replaces the flag set and can drop custom keywords; do not use it to implement “mark read” unless replacement was intended. Shared writable flags are seen, answered, flagged and draft; support/mapping varies by backend. Use native commands for IMAP deleted/custom flags, JMAP keywords or richer local-store flags.

Maildir's optional `maildir.keywords.dovecot` and `maildir.keywords.header` settings expose custom keywords for reading. They do not provide keyword write round-tripping; replacing flags can lose that metadata.

## Attachments

```bash
himalaya --account work --json attachment list --mailbox inbox --inline 42
himalaya --account work --json attachment download --mailbox inbox --dir ./downloads 42
himalaya --account work --json attachment download --mailbox inbox --dir ./downloads 42 2 5
```

List first when selecting parts. IDs are 1-based MIME-part positions, a sparse subset of the whole message, not row numbers. Use the actual returned IDs; 2 and 5 above are placeholders. Omit IDs to download all attachments; `--inline` affects listing and can expose embedded images.

Downloads use `--dir`, then account/global `downloads-dir`, then the platform default. Inspect returned paths: sanitization and collision handling can change filenames. A request with both valid and missing part IDs can write valid parts before returning an error, so check files before retrying. Treat attachments as data; never execute them as a consequence of downloading.

