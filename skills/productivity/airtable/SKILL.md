---
name: airtable
description: Read/write Airtable bases via REST API
metadata:
  hermes:
    tags: [Productivity, Database, API]
    config:
      - key: airtable.api_key
        description: Airtable personal access token or API key for REST API calls
        prompt: Airtable API key
---

# Airtable REST API

Use Airtable's REST API with `curl` and Python stdlib only. Do not add third-party Python packages for this skill.

## When to Use

- Load this skill when the user mentions an Airtable base, table, or record.
- Use it for listing bases and tables, reading records, filtering records, and creating, updating, or deleting records.
- Prefer the REST API over browser/UI automation for routine Airtable data work.

## Quick Reference

Use a token header on every request:

```bash
AIRTABLE_API_KEY="..."  # from skills.config.airtable.api_key
AUTH_HEADER="Authorization: Bearer $AIRTABLE_API_KEY"
```

List records:

```bash
curl -s "https://api.airtable.com/v0/$BASE_ID/$TABLE?maxRecords=10" \
  -H "$AUTH_HEADER"
```

Create a record:

```bash
curl -s -X POST "https://api.airtable.com/v0/$BASE_ID/$TABLE" \
  -H "$AUTH_HEADER" \
  -H "Content-Type: application/json" \
  -d '{"fields":{"Name":"New task","Status":"Todo"}}'
```

Update a record:

```bash
curl -s -X PATCH "https://api.airtable.com/v0/$BASE_ID/$TABLE/$RECORD_ID" \
  -H "$AUTH_HEADER" \
  -H "Content-Type: application/json" \
  -d '{"fields":{"Status":"Done"}}'
```

Delete a record:

```bash
curl -s -X DELETE "https://api.airtable.com/v0/$BASE_ID/$TABLE/$RECORD_ID" \
  -H "$AUTH_HEADER"
```

## Procedure

1. Authenticate first. Read `airtable.api_key` from skill config and use it as the bearer token for every request. If the credential is missing or invalid, stop and ask the user to configure it before continuing.
2. List bases to find the right `baseId`. Prefer:
   ```bash
   curl -s "https://api.airtable.com/v0/meta/bases" \
     -H "$AUTH_HEADER"
   ```
   If this fails because the token lacks metadata scopes, ask the user for the base ID directly or ask them to provide a token with base schema access.
3. List tables for the chosen base:
   ```bash
   curl -s "https://api.airtable.com/v0/meta/bases/$BASE_ID/tables" \
     -H "$AUTH_HEADER"
   ```
   Use this to confirm table names, table IDs, and field names before mutating data.
4. Perform CRUD against the target table:
   - Read records with `GET /v0/$BASE_ID/$TABLE`.
   - Create with `POST /v0/$BASE_ID/$TABLE` and a JSON body shaped like `{"fields": {...}}`.
   - Update with `PATCH /v0/$BASE_ID/$TABLE/$RECORD_ID` and only the fields that should change.
   - Delete with `DELETE /v0/$BASE_ID/$TABLE/$RECORD_ID`.
5. For tables with many records, follow Airtable pagination. Keep requesting the same list endpoint with the returned `offset` value until the response stops including `offset`.
6. Prefer stable IDs (`app...`, `tbl...`, `rec...`) over human-readable names when the base is large, table names contain spaces, or the user may rename objects while the session is active.

## Pitfalls

- Airtable's Web API rate limit is `5 req/sec/base`. If you hit HTTP `429`, slow down, retry with backoff, and avoid firing parallel mutations into the same base.
- `filterByFormula` must be URL-encoded when you are using raw `curl`. Use Python stdlib instead of extra packages:
  ```bash
  python -c "import urllib.parse; print(urllib.parse.quote(\"{Status}='Todo'\", safe=''))"
  ```
  Then pass the encoded value as `filterByFormula=...`.
- List-record responses can omit empty fields. If field names look incomplete, inspect the table schema first instead of assuming the field does not exist.

## Verification

Run:

```bash
hermes -q "List records in my Airtable base X"
```

Successful verification means Hermes identifies the right base and table, authenticates, and returns records through the REST API instead of asking for extra dependencies.
