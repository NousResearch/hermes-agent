---
name: excel-business-ontology
description: Analyze complex Excel workbooks into evidence-traceable business ontology candidates, queryable data, standardized outputs, and read-table deliverables.
version: 1.23.26
author: Zhang Lu (zhanglunet)
homepage: https://anp.asia/workbuddy/
metadata:
  hermes:
    tags: [excel, business-intelligence, data-governance, ontology, local-first]
    requires_toolsets: [terminal]
---

# Excel Business Ontology Skill

Use this skill when a user needs to understand, reconcile, standardize, query, or
deliver a complex Excel workbook. It uses `excel-business-ontology` to extract
deterministic workbook evidence and produce business-object, metric, dimension,
calculation, and relationship candidates. It does not decide business meaning,
publish authoritative definitions, or upload the user's files.

## When to Use

- The user asks to map sheets, formulas, source cells, or workbook relationships.
- The user needs evidence-backed KPI checks, cross-sheet reconciliation, or a read-table deliverable.
- The user wants to turn a workbook into a reviewable business ontology workspace.

Do not use it to silently change a source workbook, infer an authoritative metric
definition, or process files outside the paths the user names.

## Prerequisites

The skill installs and runs the pinned `excel-business-ontology@1.23.26` package.
Check the local environment before installing:

```text
node --version
npm --version
python --version
```

Required: Node.js >=18, npm, Python 3.10+, and pip. The core analysis does not
require Microsoft Excel Desktop, an API key, a token, or a business-system credential.

## How to Run

1. Ask the user for the exact Excel input path, workbook/Sheet scope, and a new output workspace path.
2. State what will be read and written, and wait for confirmation before accessing files.
3. Install the pinned package only after confirmation:

```text
npm view excel-business-ontology@1.23.26 version
npm install -g excel-business-ontology@1.23.26
ebo --version
ebo compatibility
```

4. Install the package's locked Python dependencies:

```text
python -m pip install -r "$(ebo path)/requirements.lock"
```

On Windows PowerShell, use the equivalent backslash path from `ebo path`.

5. Run both preflights in the new workspace:

```text
ebo build preflight
ebo query preflight
```

Continue only when both return `"status": "passed"`.

## Quick Reference

```text
ebo --version
ebo compatibility
ebo build preflight
ebo query preflight
ebo build --help
ebo query --help
```

For a new analysis, never guess the input scope. Ask the user to confirm the
workbook, Sheet names, output directory, and delivery goal before running build.

## Procedure

### New workspace

1. Confirm the input and output paths with the user.
2. Run the version and compatibility checks.
3. Run both preflights.
4. Read only the confirmed Excel scope.
5. Extract workbook, Sheet, formula, source-cell, and reference evidence.
6. Present candidate objects, metrics, dimensions, classifications, and calculations with evidence.
7. Pause for human confirmation of semantic choices.
8. Generate standardized or read-table outputs only after that confirmation.

### Existing workspace

Read the SQLite schema in read-only mode before any new-version build command:

```text
python3 -c "import sqlite3,sys,pathlib; print(sqlite3.connect(pathlib.Path(sys.argv[1]).resolve().as_uri()+'?mode=ro', uri=True).execute('PRAGMA user_version').fetchone()[0])" "<workspace>/data/business-ontology.sqlite"
```

- Schema 15: install 1.23.26 and continue after backup checks.
- Exactly Schema 14: explain the irreversible 14 → 15 migration, ask for confirmation, and back up first.
- Below 14: stop and direct the user to the version-by-version upgrade chain; do not run 1.23.26 build commands.

## Pitfalls

- Do not install `latest`; this skill is pinned to `1.23.26`.
- Do not run `ebo build`, including `status`, against an unreviewed old workspace.
- Do not overwrite source Excel files or delete existing workspaces.
- Do not read unconfirmed directories or upload Excel, databases, credentials, or results.
- A machine-generated candidate is not an approved business definition. Surface conflicts and wait for the user.
- Preserve `Default Permissions` or the narrowest available local access.

## Privacy and Safety

This skill is local-first. It requires no secret and must not transmit user files,
workspaces, analysis results, API keys, tokens, cookies, or passwords. Network
access is limited to retrieving the pinned npm package and its locked dependencies,
after the user has confirmed the installation.
