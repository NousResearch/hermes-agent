# Workspace Structure, Practical Mental Model

This reference captures the shape of a real Notion workspace sampled from the Alpic environment. It is not the Notion API spec. It is the operator mental model that helps decide which endpoint to use.

## What showed up in the sampled workspace

Top-level workspace pages included:
- `Marketing`
- `Sales`
- `Welcome to Alpic`
- `Build - (Customer Projects)`

Structured data sources included:
- `People`
- `Event calendar`
- `Weekly meetings marketing`
- `Weekly Sales meeting`

Sample pages inside data sources included:
- `Brendan Jowett`
- `MCP Release Party`
- `Platform pricing overhaul`
- `Marketing meeting #1`
- `Weekly Sales July 20th`

Sample nested doc pages included:
- `Skybridge weekly`
- `Weekly Tech Priorities`

## What this means

This workspace is a hybrid of four patterns:

1. **Hub pages**
   - Top-level pages act as navigation and context hubs.
   - `Welcome to Alpic` is a good example.

2. **Child pages inside hub pages**
   - A page body can contain many `child_page` blocks.
   - In the sample, `Welcome to Alpic` had many `child_page` entries, which means it behaves more like a directory or wiki home than a plain note.

3. **Structured data sources**
   - Collections like `People` and `Event calendar` are typed record stores.
   - Query them with `data_source_id`, not by crawling prose pages.

4. **Record pages inside data sources**
   - Every row is still a page.
   - Those pages have typed properties and may also have body content.

## Real schema examples

### `People`
Observed properties:
- `Name`, title
- `About`, rich_text
- `Person`, people
- `Membership Type`, select

Interpretation:
- This is a lightweight people directory.
- Start with property queries, then inspect page body only if the task requires richer notes.

### `Event calendar`
Observed properties included:
- `Marketing point person`, people
- `Date`, date
- `Link`, url
- `Format`, select
- `GOING`, select
- `Notes`, rich_text
- `Type`, select
- `Ticket cost`, rich_text
- `CFP link`, url
- `Sponsoring cost`, number
- `People Going`, people
- `CFP Deadline`, date

Interpretation:
- This is a structured operational tracker.
- Query and sort the data source first; use page body only for long-form notes.

## Body content examples

### `Welcome to Alpic`
Observed child blocks were mostly:
- `child_page`
- plus at least one `paragraph`

Interpretation:
- Reading only `GET /pages/{id}` will miss the actual structure.
- Read `GET /blocks/{id}/children` when the page acts like a hub.

### `Skybridge weekly`
Observed body blocks included:
- `paragraph`
- `heading_1`
- `toggle`
- `bulleted_list_item`

Interpretation:
- This is a document-style page.
- Use block reads or Markdown reads, not only metadata.

## Endpoint decision guide

### If the thing looks like a note, wiki page, or project memo
Use:
- `GET /v1/pages/{page_id}` for metadata
- `GET /v1/blocks/{page_id}/children` or `/markdown` for actual content

### If the thing looks like a table of people, meetings, or events
Use:
- `GET /v1/data_sources/{data_source_id}` for schema
- `POST /v1/data_sources/{data_source_id}/query` for records

### If you have a search result and need to decide what it is
Check:
- `object == "page"`, treat it as a page
- `object == "data_source"`, treat it as a structured collection
- `page.parent.data_source_id`, it is a record page inside a data source
- `page.parent.page_id`, it is nested under another page
- `page.parent.workspace`, it is top-level

## Common mistakes in a workspace like this

1. Reading a hub page as if it were only text
   - Many hub pages are mostly `child_page` blocks.

2. Treating data source rows as a separate object type
   - They are still pages.

3. Querying by page when a data source query is the real operation
   - Records belong to a typed collection; query the collection first.

4. Assuming `GET /pages/{id}` returns the document body
   - It does not. Use blocks or Markdown.

5. Mixing `database_id` and `data_source_id`
   - Query with `data_source_id`.
   - Create records with the parent form expected by the API shape in use.

## Fast mental shortcut

- **Hub page**: blocks matter.
- **Doc page**: Markdown or blocks matter.
- **Data source**: schema and query matter.
- **Row in data source**: properties first, body second.
