# Data Sources, Databases, Pages, Blocks, Properties, and Views

Sources:

- `https://developers.notion.com/reference/database.md`
- `https://developers.notion.com/reference/data-source.md`
- `https://developers.notion.com/guides/get-started/upgrade-guide-2025-09-03.md`
- `https://developers.notion.com/guides/data-apis/working-with-databases.md`
- `https://developers.notion.com/guides/data-apis/working-with-page-content.md`
- `https://developers.notion.com/guides/data-apis/working-with-views.md`
- public OpenAPI spec

## Resource graph

Current model:

```text
database container
  └─ data_source table/schema/row parent
       ├─ page rows with properties
       └─ view definitions / queries
page body
  └─ block children tree
```

Database:

- `object: "database"`.
- Container that can have one or more data sources.
- Carries container fields: title, description, parent, icon, cover, URL/public URL, `is_inline`, `in_trash`, timestamps/users, `data_sources:[{id,name}]`.
- Permissions are managed on the database/container, not individual data sources.

Data source:

- `object: "data_source"`.
- Represents a table/schema/row collection.
- Carries `properties` schema.
- Parent is usually a `database_id`.
- Pages under a data source are rows.

## Migration rules from old database API

Old work often says “database” when it now means “data source.” Correct it by endpoint shape:

- Old retrieve schema: `GET /v1/databases/{database_id}`.
  - New: retrieve database to discover `data_sources`; call `GET /v1/data_sources/{data_source_id}` for schema.
- Old query rows: `POST /v1/databases/{database_id}/query`.
  - New: `POST /v1/data_sources/{data_source_id}/query`.
- Old create row parent: `parent: { database_id: ... }`.
  - New: `parent: { type: "data_source_id", data_source_id: ... }`.
- Old relations targeting database ID in writes.
  - New: relation schema/write targets use `data_source_id`.
- Rich-text database mentions remain database IDs, not data-source IDs.
- Old search result/filter value `database`.
  - New: `data_source`.

Database IDs and data-source IDs are not interchangeable.

## Page model

Page object fields include:

- `object: "page"`
- `id`
- created/edited timestamps/users
- `parent`
- `properties`
- `icon`, `cover`
- `url`, `public_url`
- `in_trash`
- optional lock fields in current OpenAPI responses

Parent behavior:

- Parent page/workspace: only title property is valid.
- Parent data source: properties must match the data-source schema.

Body content:

- Retrieve page returns metadata/properties, not body blocks.
- Use page ID as `block_id` with block-children endpoint for structured traversal.
- Use markdown endpoint for agent-readable body text.

Property completeness:

- Retrieve-page can truncate `people`, `relation`, `rich_text`, and `title` values around 25 refs.
- Use `GET /v1/pages/{page_id}/properties/{property_id}` for complete paginated values.

## Data-source query

Endpoint:

```text
POST /v1/data_sources/{data_source_id}/query
```

Common body fields:

- `filter`
- `sorts`
- `start_cursor`
- `page_size`
- `filter_properties`
- `in_trash`
- `result_type`, useful for wiki-like cases

Operational notes:

- Use filters and narrow `page_size` for large sources.
- Query can stop/cap around 10,000 results. For large syncs, use filters and webhooks instead of full polling.
- Always inspect query `request_status`; a response can be a valid page of results but still marked incomplete because the query-result limit was reached.
- `filter_properties[]` is a URL/query parameter on a `POST` endpoint, not a JSON body field in raw HTTP; SDKs may expose it as a named option.
- Query 503 guidance recommends backoff with jitter, smaller page size, and narrower filters/sorts.

## Property schemas and values

Data-source property object = schema/column definition:

- `id`, `name`, `description`, `type`, plus a type-specific config object.

Common schema property types:

```text
checkbox
created_by
created_time
date
email
files
formula
last_edited_by
last_edited_time
multi_select
number
people
phone_number
place
relation
rich_text
rollup
select
status
title
unique_id
url
```

Page property value object = row/page value:

- `id`, `type`, and a type-specific value.
- Some values are generated/read-only.
- Page property item endpoint returns either one value or a paginated list of property items, depending on property type.

Relation schema:

- Use `data_source_id` for write targets after `2025-09-03`.
- Related data source/database must usually be shared with the connection.

## Blocks and traversal

Block object fields:

- `object: "block"`
- `id`
- `parent`
- `type`
- created/edited timestamps/users
- `has_children`
- `in_trash`
- type-specific object keyed by `type`

Traversal:

1. Use page ID as block ID.
2. Call `GET /v1/blocks/{block_id}/children`.
3. For each block with `has_children: true`, recurse.
4. Preserve sibling order from the API response, but do not assume unrelated list endpoint ordering unless documented.

Appending:

- `PATCH /v1/blocks/{block_id}/children` appends/creates child blocks.
- Use `position` with `end`, `start`, or `after_block`.
- Existing blocks cannot be moved with append endpoint.
- Max 100 child blocks per request.
- Max two nesting levels in one request.

Updating/deleting:

- `PATCH /v1/blocks/{block_id}` replaces included block fields; omitted fields unchanged.
- `DELETE /v1/blocks/{block_id}` sets `in_trash: true`.
- Children are not updated by updating a parent block.

## Views

Views API exists in the 2026 docs/OpenAPI despite stale older FAQ text.

View object fields include:

- `object: "view"`
- `id`
- parent database
- `data_source_id` (null for dashboard)
- `name`
- `type`
- `filter`
- `sorts`
- `quick_filters`
- `configuration`
- timestamps/users
- URL

View types:

```text
table
board
list
calendar
timeline
gallery
form
chart
map
dashboard
```

View endpoints:

- `GET /v1/views?database_id=...` — list views for database.
- `GET /v1/views?data_source_id=...` — list views over data source.
- `GET /v1/views/{view_id}` — full config.
- `POST /v1/views` — create view.
- `PATCH /v1/views/{view_id}` — update view.
- `DELETE /v1/views/{view_id}` — delete view; cannot delete last remaining database view.
- `POST /v1/views/{view_id}/queries` — create cached query from view filters/sorts.
- `GET /v1/views/{view_id}/queries/{query_id}` — paginate cached query.
- `DELETE /v1/views/{view_id}/queries/{query_id}` — free cached query.

View query notes:

- Cached query expires after about 15 minutes.
- You cannot add extra filters/sorts to a view query.
- Use data-source query if you need ad-hoc filters/sorts.

Create/update/dashboard caveats:

- Views require API version `2025-09-03` or newer. Source: `https://developers.notion.com/guides/data-apis/working-with-views.md`.
- `POST /v1/views` requires `data_source_id` and exactly one placement parent: `database_id`, dashboard `view_id`, or `create_database`. Database IDs and data-source IDs are different. Source: `https://developers.notion.com/reference/create-view.md`.
- Dashboard view `configuration.rows` is read-only; manage dashboard layout by creating/deleting widget views. Widgets can only be placed with `view_id`, and the docs cap widgets at four per row. Source: `https://developers.notion.com/guides/data-apis/working-with-views.md`.
- `PATCH /v1/views/{view_id}` updates configuration by shallow merge and still needs required config fields. Source: `https://developers.notion.com/reference/update-view.md`.
- View-query access still depends on data-source/database access; `404 object_not_found` can mean the connection cannot access the underlying database. Source: `https://developers.notion.com/reference/get-view-query-results.md`.

## Search

Endpoint:

```text
POST /v1/search
```

Use for title discovery across shared pages/data sources. Do not use as authoritative inventory.

Filters:

- `filter.property: "object"`
- `filter.value: "page"` or `"data_source"`

Limitations:

- Not exhaustive.
- Indexing after sharing/OAuth is not immediate.
- Results can change while paginating.
- Use data-source query to filter within one data source.

## Comments

Pages are blocks for comment-listing purposes.

Endpoints:

- `GET /v1/comments?block_id=...` — list open comments.
- `POST /v1/comments` — add page comment or reply to discussion.
- `GET /v1/comments/{comment_id}` — retrieve.
- `PATCH /v1/comments/{comment_id}` — update.
- `DELETE /v1/comments/{comment_id}` — delete.

Limits:

- REST API cannot start a new inline discussion on an arbitrary text range.
- It can reply to existing discussions via `discussion_id`.
- It cannot retrieve resolved comments.
- Connections can only delete/update comments they created.

## Common mistakes

- Creating data-source rows with `database_id` parent in new code.
- Querying `/databases/{id}/query` for new work.
- Treating retrieve page as page-body retrieval.
- Not recursing `has_children` blocks.
- Forgetting page-property endpoint for complete relation/people/title/rich-text values.
- Assuming views are unsupported because older 2025 text says so.
