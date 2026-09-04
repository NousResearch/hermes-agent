---
name: nocodb
description: Manage NocoDB bases, tables, fields, and records.
version: 1.0.0
author: Anbarasu (DarkPhoenix2704)
license: MIT
platforms: [macos, linux, windows]
required_environment_variables:
  - name: NOCODB_TOKEN
    prompt: NocoDB API token
    help: "NocoDB → account menu → Account Settings → Tokens → Add New Token. Sent as the xc-token header."
    required_for: every command
  - name: NOCODB_URL
    prompt: NocoDB instance URL (default https://app.nocodb.com)
    help: "Self-hosted instances only — the origin, with no /api suffix and no trailing slash."
    required_for: self-hosted instances
prerequisites:
  env_vars: [NOCODB_TOKEN]
  commands: [curl, jq]
metadata:
  hermes:
    tags: [NocoDB, Database, Spreadsheet, No-Code, API, Records]
    related_skills: [airtable]
    requires_toolsets: [terminal]
    homepage: https://nocodb.com/docs/product-docs
---

# NocoDB Skill

Drive the NocoDB v3 REST API from the command line: browse workspaces and
bases, read and write records, and change schema (tables, fields, views,
filters, sorts). This is the full admin surface — the NocoDB MCP server, by
contrast, is record-CRUD only.

It does not run NocoDB itself, sync external data sources, or manage billing.
Workspace, view, script, team, and API-token endpoints are Enterprise-plan
features (self-hosted or cloud) and error out on the free plan.

## When to Use

- The user names a NocoDB base, table, or record, or pastes an `app.nocodb.com`
  (or self-hosted NocoDB) link.
- Bulk record work: import, backfill, deduplicate, or export rows.
- Schema work the MCP server cannot do: create tables, add fields, build views,
  attach filters and sorts.
- The user wants a lightweight database for tracking something and has NocoDB
  available.

Prefer the `nocodb` MCP server when the task is plain record CRUD on a single
base and the user already authorized it — it needs no token handling. Use this
skill for everything structural.

## Prerequisites

- `NOCODB_TOKEN` — API token. NocoDB → account menu → **Account Settings** →
  **Tokens** → **Add New Token**. Sent as the `xc-token` header.
- `NOCODB_URL` — only for self-hosted instances. Defaults to
  `https://app.nocodb.com`. Use the bare origin: `https://nocodb.example.com`,
  no `/api` suffix, no trailing slash.
- `NOCODB_VERBOSE=1` — optional; prints each name→ID resolution to stderr.
- `curl` and `jq` on `PATH` for the Bash script. The PowerShell script uses
  `Invoke-RestMethod` and needs neither.

The token grants full access to every base it can reach. Never echo it into
transcripts or commit it.

## How to Run

Invoke through the `terminal` tool from the skill directory. Pick the script
that matches the platform — both expose an identical 72-command surface:

- macOS / Linux: `scripts/nocodb.sh <command> [args...]`
- Windows: `pwsh -File scripts/nocodb.ps1 <command> [args...]`

Every example below uses the Bash form. There is no `nc` binary — do not
shorten the invocation, since `nc` is netcat.

Arguments are always hierarchical, widest scope first:

```
WORKSPACE → BASE → TABLE → VIEW/FIELD → RECORD
```

Names or IDs both work. Names cost an extra lookup per level; IDs are exact.
ID prefixes: `w`=workspace, `p`=base, `m`=table, `c`=field, `vw`=view.

## Quick Reference

| Command | Args | Notes |
|---|---|---|
| `workspace:list` | — | Enterprise |
| `base:list` | `WORKSPACE` | → `p…` base IDs |
| `base:create` | `WORKSPACE` `'{"title":"…"}'` | |
| `table:list` | `BASE` | → `m…` table IDs |
| `table:create` | `BASE` `'{"title":"…"}'` | |
| `field:list` | `BASE` `TABLE` | title, type, `c…` ID |
| `field:create` | `BASE` `TABLE` `'{"title":"…","type":"…"}'` | |
| `view:list` | `BASE` `TABLE` | Enterprise |
| `record:list` | `BASE` `TABLE` `[PAGE] [SIZE] [WHERE] [SORT] [FIELDS]` | 25/page default |
| `record:get` | `BASE` `TABLE` `RECORD_ID [FIELDS]` | |
| `record:create` | `BASE` `TABLE` `'{"fields":{…}}'` | |
| `record:update` | `BASE` `TABLE` `RECORD_ID` `'{…}'` | |
| `record:update-many` | `BASE` `TABLE` `'[{"id":1,"fields":{…}}]'` | |
| `record:delete` | `BASE` `TABLE` `ID` or `'[{"id":31}]'` | |
| `record:count` | `BASE` `TABLE` `[WHERE]` | |
| `link:list` \| `link:add` \| `link:remove` | `BASE` `TABLE` `FIELD` `RECORD_ID` `[…]` | link payload `[{"id":42}]` |
| `filter:create` \| `sort:create` | `BASE` `TABLE` `VIEW` `'{…}'` | view-level, Enterprise |
| `attachment:upload` | `BASE` `TABLE` `RECORD_ID` `FIELD` `PATH` | multipart |
| `where:help` | — | full filter syntax |

Field types: `SingleLineText`, `LongText`, `Number`, `Decimal`, `Currency`,
`Percent`, `Email`, `URL`, `PhoneNumber`, `Date`, `DateTime`, `Time`,
`SingleSelect`, `MultiSelect`, `Checkbox`, `Rating`, `Attachment`, `Links`,
`User`, `JSON`.

Run `scripts/nocodb.sh` with no arguments for the complete command list.

### Where-filter syntax

```
(field,operator,value)              (name,eq,John)
(field,operator)                    (notes,blank)
(field,operator,sub_op[,value])     (created_at,isWithin,pastWeek)
```

Operators: `eq`, `neq`, `like`, `nlike`, `in`, `gt`, `lt`, `gte`, `lte`,
`blank`, `notblank`, `checked`, `notchecked`.

Combine with a **tilde** prefix — `~and`, `~or`, `~not`. Plain `and`/`or` is
the single most common mistake and the script rejects it:

```bash
scripts/nocodb.sh record:list MyBase Tasks 1 25 \
  "(due_date,lt,today)~and(priority,eq,high)~and(completed,notchecked)"
```

## Procedure

1. **Confirm credentials are loaded.** Run the Verification command below. A
   `NOCODB_TOKEN required` error means the env var never reached the shell.

2. **Resolve the target once, then reuse IDs.** Passing a name instead of an ID
   costs a lookup on every call, and resolving a *base* name scans every
   workspace the token can see:

   ```bash
   scripts/nocodb.sh base:list wabc1234xyz        # → pdef5678uvw
   scripts/nocodb.sh table:list pdef5678uvw       # → mghi9012rst
   scripts/nocodb.sh field:list pdef5678uvw mghi9012rst
   ```

   On a free plan there is no `workspace:list` to seed step one. Read the base
   ID out of the NocoDB URL instead — `app.nocodb.com/#/<workspace>/<baseId>/…`
   — and start at `table:list`.

3. **Inspect the schema before writing.** `field:list` gives exact titles,
   types, and IDs. Guessed field names are silently dropped by the API rather
   than rejected.

4. **Read before you write.** Preview the affected rows with the same filter
   you are about to mutate:

   ```bash
   scripts/nocodb.sh record:count pdef5678uvw mghi9012rst "(status,eq,stale)"
   scripts/nocodb.sh record:list  pdef5678uvw mghi9012rst 1 5 "(status,eq,stale)"
   ```

5. **Batch mutations.** Use `record:update-many` and array-form
   `record:delete` instead of a call per row.

6. **Page explicitly.** `record:list` returns 25 rows per page. Drive
   `record:count` first, then loop pages — never assume one page is the whole
   table.

7. **Write results with `write_file`** when the user wants an export. Pipe the
   JSON through `jq` inside the command rather than reformatting it yourself.

## Pitfalls

- **Plain `and`/`or` in filters.** Must be `~and` / `~or` / `~not`.
- **`record:create` wraps fields.** Payload is `{"fields":{…}}`; `record:update`
  takes the bare object `{…}`. Mixing them up returns a 400.
- **Unknown field names do not error.** They are dropped. Verify with
  `field:list` and re-read the record after writing.
- **Enterprise-gated endpoints.** Workspace, view, script, team, and API-token
  commands need an Enterprise plan (self-hosted or cloud). On the free plan
  `workspace:list` fails — pass the base ID directly instead.
- **`base:list` takes a workspace argument** and is therefore Enterprise-gated
  in practice. Free-plan users should skip it and read the base ID from the
  NocoDB URL.
- **Passing a base *name*** makes the script enumerate every workspace looking
  for it — another Enterprise-only path. Pass the `p…` ID.
- **Self-hosted URL shape.** `NOCODB_URL` is the origin only. A trailing slash
  or an `/api` suffix produces 404s on every call.
- **Record IDs are per-table integers**, not the `p…`/`m…` nanoid strings.
- **Deletes are immediate.** There is no undo through the API.

## Verification

```bash
scripts/nocodb.sh table:list <BASE_ID>
```

Prints one `title<TAB>id` row per table. Any output means the token, the URL,
and the dependencies are all correct. `NOCODB_TOKEN required` means the env var
never reached the shell; an HTML or 401 body means the token or `NOCODB_URL` is
wrong.

On Enterprise plans `scripts/nocodb.sh workspace:list` verifies the same thing
without needing a base ID up front.

## Attribution

Ported from [nocodb/agent-skills](https://github.com/nocodb/agent-skills)
(MIT). The `scripts/` files are vendored from that repository.
