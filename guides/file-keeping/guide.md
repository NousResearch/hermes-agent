# File keeping

`{workdir}` is the organization's drive. `{workdir}/documents` is where lasting documents live.
`{workdir}/repos` holds git checkouts, one directory per repo.
Incoming attachments use the paths in their message notices. Their retention follows the native channel; copy lasting material into documents.

## Keep or let expire

- File it: documents and data sent as lasting material — reports, contracts,
  spreadsheets, exports.
- Let it expire: screenshots, one-off images, files sent only to answer the
  question at hand.
- Genuinely unsure? Ask what it's for, or file it — a lost document costs more
  than disk.

## Filing

- Copy lasting material during the conversation it arrived in; do not rely on attachment-cache retention.
- Rename while filing: `doc_3f2a9c1b_Q4 report(3).xlsx` becomes
  `finance/annual-report-2025.xlsx`. Lowercase-kebab; include the year or date
  when the file is one of a series.
- Shallow topic folders (`finance/`, `hiring/`, `contracts/`) — one level, two
  at most. No inbox or unsorted folder: if no topic fits yet, leave it at the
  `documents/` root until related files accumulate.
- Reorganizing later is encouraged — no product mechanism references these
  paths. Your own artifacts might: before moving a file, grep the workspace
  for its path (schedules, responsibilities, scripts, stubs) and update any
  reference you find.

## Cloud-doc stubs

A cloud doc shared as lasting material gets a stub — a markdown file in the
same tree:

```markdown
---
url: https://docs.google.com/spreadsheets/d/…
saved: 2026-07-23
---
The company's hiring tracker. One row per candidate; tabs for
Engineering and GTM; owned by Sarah.
```

- The body describes what the doc **is** — purpose, shape, owner. Never copy
  its contents: the live doc is authoritative, and a snapshot becomes a stale
  fork that search will surface as truth.
- Before creating a stub, grep `documents/` for the doc's URL or ID — update
  the existing stub instead of duplicating.
- Can't open the link yet? Save the stub from what the person said it is;
  enrich the description the first time you do open it.
- When someone wants the file itself, export fresh from the live doc.

## Dead links

A link that no longer opens means deleted, moved, or lost access — three
different fixes. Check which before acting: update the stub (moved), sort out
access (permissions), or confirm with the person and then remove it (deleted).
Never silently delete a stub.
