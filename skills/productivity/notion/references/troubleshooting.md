# Troubleshooting and Common Confusions

Use this reference when the Notion API behaves in a way that looks wrong but usually is not.

## 404 even though the page exists

Most common cause: the integration is not shared on that page or data source.

Fix:
- Open the page or data source in Notion
- Click `...` or `Share`
- Connect the integration

This is the most common false-negative in Notion API work.

## I read the page, but I did not get the content

`GET /v1/pages/{page_id}` returns metadata and properties, not the page body.

Use instead:
- `GET /v1/blocks/{page_id}/children`, or
- `GET /v1/pages/{page_id}/markdown`

## I need the records in a table, not the text on a page

You probably want a data source query, not a page read.

Use:
- `GET /v1/data_sources/{data_source_id}` to inspect schema
- `POST /v1/data_sources/{data_source_id}/query` to retrieve records

## I found a row in search, but I do not know what it is

A database row is still a page.

Clues:
- `object == "page"` means page
- `parent.data_source_id` means the page is a record inside a data source
- `object == "data_source"` means query the collection itself

## database_id vs data_source_id

This is a major source of confusion in the 2025-09-03 API version.

Rule of thumb:
- use `data_source_id` for query and retrieval workflows on structured collections
- use `database_id` in parent payloads when creating a page inside a structured collection

When the API docs and examples disagree, check the live endpoint docs via `ntn api <path> --docs` or `--spec`.

## Search returned a page, but the real target is the collection

Common pattern:
- search finds a record page like an event, person, or meeting
- the task actually needs all matching records or the schema

Fix:
- inspect `parent`
- if the page has `parent.data_source_id`, switch to querying that data source

## Reading a hub page misses the structure

Many workspace landing pages are mostly `child_page` blocks.

Symptoms:
- `/pages/{id}` returns almost nothing useful
- `/blocks/{id}/children` reveals many subpages

Fix:
- treat it like a directory page, not a plain note

## Rate limited

Expected average rate is about 3 requests per second.

Fix:
- slow down
- batch less aggressively
- honor `Retry-After` on `429`

## Large properties look truncated

Some property item payloads paginate, especially relations, people, and other multi-value properties.

Use:
- `GET /v1/pages/{page_id}/properties/{property_id}`

## Good operating order

When a task is unclear, this order is usually right:
1. search
2. identify whether the result is a page or data source
3. if page, decide properties vs body
4. if data source, inspect schema then query
5. only then mutate content or properties
