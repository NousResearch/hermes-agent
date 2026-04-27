---
name: airtable
description: Read/write Airtable bases via REST API using curl. List bases, tables, and records; create, update, and delete records. No dependencies beyond curl.
version: 1.0.0
author: community
license: MIT
prerequisites:
  env_vars: [AIRTABLE_API_KEY]
  commands: [curl]
metadata:
  hermes:
    tags: [Airtable, Productivity, Database, API]
    homepage: https://airtable.com/developers/web/api/introduction
---

# Airtable REST API

Use Airtable's REST API via `curl` to list bases, inspect schemas, and run CRUD against records. No extra packages — `curl` plus Python stdlib for URL encoding is enough.

## Setup

1. Create a personal access token (PAT) at https://airtable.com/create/tokens
2. Grant these scopes (minimum):
   - `data.records:read` — read rows
   - `data.records:write` — create / update / delete rows
   - `schema.bases:read` — list bases and tables (step 2–3 of the procedure below)
3. Add to `~/.hermes/.env` (or set via `hermes setup`):
   ```
   AIRTABLE_API_KEY=pat_your_token_here
   ```
4. In the PAT UI, also add each base you want to access to the token's "Access" list. Tokens are scoped per-base.

> Note: legacy `key...` API keys were deprecated in Feb 2024. PATs (starting with `pat`) are the only supported format.

## API Basics

- **Base URL:** `https://api.airtable.com/v0`
- **Auth header:** `Authorization: Bearer $AIRTABLE_API_KEY`
- **Object IDs:** bases `app...`, tables `tbl...`, records `rec...`. Prefer IDs over names when table names have spaces or may change.
- **Rate limit:** 5 requests/sec/base. On `429`, back off and avoid parallel mutations into the same base.

## Quick Reference

```bash
AUTH="Authorization: Bearer $AIRTABLE_API_KEY"
BASE_ID=appXXXXXXXXXXXXXX
TABLE=Tasks   # or tblXXXXXXXXXXXXXX
```

List records (first 10):
```bash
curl -s "https://api.airtable.com/v0/$BASE_ID/$TABLE?maxRecords=10" -H "$AUTH"
```

Create a record:
```bash
curl -s -X POST "https://api.airtable.com/v0/$BASE_ID/$TABLE" \
  -H "$AUTH" -H "Content-Type: application/json" \
  -d '{"fields":{"Name":"New task","Status":"Todo"}}'
```

Update a record (partial — PATCH preserves other fields):
```bash
curl -s -X PATCH "https://api.airtable.com/v0/$BASE_ID/$TABLE/$RECORD_ID" \
  -H "$AUTH" -H "Content-Type: application/json" \
  -d '{"fields":{"Status":"Done"}}'
```

Delete a record:
```bash
curl -s -X DELETE "https://api.airtable.com/v0/$BASE_ID/$TABLE/$RECORD_ID" -H "$AUTH"
```

## Procedure

1. **Authenticate.** Confirm `AIRTABLE_API_KEY` is set. If empty, stop and ask the user to add it to `~/.hermes/.env`.
2. **Find the base.** List all bases the token can see:
   ```bash
   curl -s "https://api.airtable.com/v0/meta/bases" -H "$AUTH"
   ```
   Requires `schema.bases:read`. If the token lacks that scope, ask the user for the base ID directly.
3. **Inspect the schema.** List tables and fields for the chosen base:
   ```bash
   curl -s "https://api.airtable.com/v0/meta/bases/$BASE_ID/tables" -H "$AUTH"
   ```
   Use this to confirm table names, IDs, and field names before mutating data.
4. **CRUD against the target table.**
   - Read: `GET /v0/$BASE_ID/$TABLE`
   - Create: `POST /v0/$BASE_ID/$TABLE` with `{"fields": {...}}`
   - Update: `PATCH /v0/$BASE_ID/$TABLE/$RECORD_ID` with only the fields to change (use `PUT` for full replacement)
   - Delete: `DELETE /v0/$BASE_ID/$TABLE/$RECORD_ID`
5. **Paginate long lists.** The list endpoint caps at 100 records per page. If the response includes `"offset": "..."`, pass it back as `?offset=<value>` on the next call and repeat until the field is absent.

## Pitfalls

- **`filterByFormula` must be URL-encoded.** Use Python stdlib — no extra packages:
  ```bash
  ENC=$(python3 -c "import urllib.parse, sys; print(urllib.parse.quote(sys.argv[1], safe=''))" "{Status}='Todo'")
  curl -s "https://api.airtable.com/v0/$BASE_ID/$TABLE?filterByFormula=$ENC" -H "$AUTH"
  ```
- **Empty fields are omitted from responses.** If a record looks like it's missing fields, inspect the table schema (step 3) before concluding the field doesn't exist.
- **Tokens are per-base.** The PAT UI requires adding each base to the token's Access list. A 403 on a specific base usually means the base wasn't granted, not that the token is wrong.
- **PATCH vs PUT.** `PATCH` merges the supplied fields into the existing record; `PUT` replaces the record entirely, wiping any fields you didn't include. Default to `PATCH` unless you genuinely want to clear other fields.

## Verification

```bash
curl -s -o /dev/null -w "%{http_code}\n" "https://api.airtable.com/v0/meta/bases" \
  -H "Authorization: Bearer $AIRTABLE_API_KEY"
```

Expect `200` with a `bases` array. `401` means the key is wrong; `403` means the token is valid but lacks `schema.bases:read` (use step 2 workaround).
