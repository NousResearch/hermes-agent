---
name: notion
description: >
  Read and write Notion pages and databases using the Notion API.
  Create pages, append content, update properties, query databases,
  and search across your entire Notion workspace.
version: 1.0.0
metadata:
  hermes:
    tags: [notion, productivity, notes, database, tasks]
    category: productivity
---

# Notion Integration Skill

## When to Use

Use this skill when the user wants to:

- **Read** a Notion page or database
- **Create** a new page or entry
- **Update** page properties (status, tags, checkbox, dates, etc.)
- **Append** notes or content to an existing page
- **Search** their Notion workspace for a page or database
- **Query** a database with filters (e.g. "show me all incomplete tasks")
- **Log** information persistently to Notion (meeting notes, research, daily logs)

## Setup (one-time)

1. Go to <https://www.notion.so/my-integrations>
2. Click **"New integration"**, give it a name (e.g. "Hermes Agent"), and copy the **Internal Integration Secret**
3. Add it to `~/.hermes/.env`:
   ```
   NOTION_API_KEY=secret_xxxxxxxxxxxxxxxxxxxx
   ```
4. In Notion, open each page/database you want Hermes to access → click **"..."** → **"Add connections"** → select your integration

## Procedure

### Searching / Listing

```
notion_search(query="meeting notes")          # find pages by name
notion_search(query="", filter_type="database")  # list all databases
```

### Reading a Page

```
notion_get_page(page_id="<id from search>")
```

Returns the full page title, metadata, and all block content as readable text.

### Creating a Page

In a database:
```
notion_create_page(
    parent_id="<database-id>",
    title="New Task",
    content="Details about the task",
    parent_type="database"
)
```

As a sub-page of another page:
```
notion_create_page(
    parent_id="<page-id>",
    title="Meeting Notes 2026-02-26",
    parent_type="page"
)
```

### Appending Content to a Page

```
notion_append_blocks(page_id="<id>", text="New paragraph here")

# Multiple blocks separated by blank lines:
notion_append_blocks(page_id="<id>", text="Item 1\n\nItem 2\n\nItem 3",
                     block_type="bulleted_list_item")

# Todo list:
notion_append_blocks(page_id="<id>", text="Buy groceries\n\nCall doctor",
                     block_type="to_do")
```

### Updating Page Properties

```
# Mark a task done:
notion_update_page(page_id="<id>", properties='{"Done": {"checkbox": true}}')

# Change status:
notion_update_page(page_id="<id>",
                   properties='{"Status": {"select": {"name": "In Progress"}}}')

# Set a date:
notion_update_page(page_id="<id>",
                   properties='{"Due Date": {"date": {"start": "2026-03-01"}}}')
```

### Querying a Database

```
# List all entries:
notion_query_database(database_id="<id>")

# Filter by checkbox:
notion_query_database(
    database_id="<id>",
    filter_json='{"property": "Done", "checkbox": {"equals": false}}'
)

# Sort by date descending:
notion_query_database(
    database_id="<id>",
    sorts_json='[{"property": "Created", "direction": "descending"}]'
)
```

## Pitfalls

- **"NOTION_API_KEY not set"** — Add the key to `~/.hermes/.env`, then restart Hermes.
- **"object_not_found"** — The integration hasn't been connected to that page/database. Open the page in Notion → "..." → "Add connections" → select your integration.
- **Property names are case-sensitive** — If `update_page` fails, double-check exact property names in Notion. Use `notion_get_page` to inspect the raw property keys.
- **Title property varies by database** — Some databases name it "Name", others "Title". `notion_create_page` tries "Name" first for databases. If creation fails, check the database's title property name.
- **Rate limits** — The Notion API is limited to ~3 requests/second. For bulk operations, add small delays between calls.
- **Pagination** — `notion_query_database` returns up to 100 results per call. For larger databases, use filters or increase `page_size`.

## Verification

After creating a page:
```
notion_search(query="<title you just created>")
```
The new page should appear in results.

After updating a property:
```
notion_get_page(page_id="<id>")
```
Confirm the property value changed in the output.
