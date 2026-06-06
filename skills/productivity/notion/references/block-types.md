# Notion Block Types

Reference for reading and writing common Notion blocks through the REST API.

Sources:

- `https://developers.notion.com/reference/block.md`
- `https://developers.notion.com/reference/get-block-children.md`
- `https://developers.notion.com/reference/patch-block-children.md`
- `https://developers.notion.com/reference/update-a-block.md`
- `https://developers.notion.com/guides/data-apis/working-with-page-content.md`

## Block shape

Every block has:

```json
{
  "object": "block",
  "id": "...",
  "parent": {"type": "page_id", "page_id": "..."},
  "type": "paragraph",
  "has_children": false,
  "in_trash": false,
  "paragraph": {"rich_text": []}
}
```

The type-specific object is keyed by `type`.

## Traversal

- A page ID can be used as a `block_id`.
- `GET /v1/blocks/{block_id}/children` returns only first-level children.
- Recurse when `has_children` is true.
- Do not assume child blocks are returned inline with their parent.

## Appending blocks

Endpoint:

```text
PATCH /v1/blocks/{block_id}/children
```

Use `position`, not legacy `after`:

```json
{
  "position": {"type": "end"},
  "children": [
    {"object":"block","type":"paragraph","paragraph":{"rich_text":[{"text":{"content":"Hello"}}]}}
  ]
}
```

Position variants:

```json
{"type":"end"}
{"type":"start"}
{"type":"after_block","after_block_id":"..."}
```

Limits:

- Max 100 child blocks per request.
- Max two nesting levels in one request.
- Existing blocks cannot be moved with append; use page move for pages where applicable.

## Common block creation snippets

Paragraph:

```json
{"object":"block","type":"paragraph","paragraph":{"rich_text":[{"text":{"content":"Hello world"}}]}}
```

Headings:

```json
{"object":"block","type":"heading_1","heading_1":{"rich_text":[{"text":{"content":"Title"}}]}}
{"object":"block","type":"heading_2","heading_2":{"rich_text":[{"text":{"content":"Section"}}]}}
{"object":"block","type":"heading_3","heading_3":{"rich_text":[{"text":{"content":"Subsection"}}]}}
{"object":"block","type":"heading_4","heading_4":{"rich_text":[{"text":{"content":"Detail"}}]}}
```

Bulleted and numbered list items:

```json
{"object":"block","type":"bulleted_list_item","bulleted_list_item":{"rich_text":[{"text":{"content":"Item"}}]}}
{"object":"block","type":"numbered_list_item","numbered_list_item":{"rich_text":[{"text":{"content":"Step"}}]}}
```

To-do:

```json
{"object":"block","type":"to_do","to_do":{"rich_text":[{"text":{"content":"Task"}}],"checked":false}}
```

Toggle:

```json
{"object":"block","type":"toggle","toggle":{"rich_text":[{"text":{"content":"Open"}}],"children":[]}}
```

Quote:

```json
{"object":"block","type":"quote","quote":{"rich_text":[{"text":{"content":"Something wise"}}]}}
```

Callout:

```json
{"object":"block","type":"callout","callout":{"rich_text":[{"text":{"content":"Important"}}],"icon":{"type":"emoji","emoji":"💡"},"color":"blue_background"}}
```

Code:

```json
{"object":"block","type":"code","code":{"rich_text":[{"text":{"content":"print('hello')"}}],"language":"python"}}
```

Divider:

```json
{"object":"block","type":"divider","divider":{}}
```

Bookmark:

```json
{"object":"block","type":"bookmark","bookmark":{"url":"https://example.com","caption":[]}}
```

Image external URL:

```json
{"object":"block","type":"image","image":{"type":"external","external":{"url":"https://example.com/photo.png"},"caption":[]}}
```

Image uploaded with File Upload API:

```json
{"object":"block","type":"image","image":{"type":"file_upload","file_upload":{"id":"FILE_UPLOAD_ID"},"caption":[]}}
```

Read-only/generated media caveats:

- `link_preview` blocks can be returned by the API but cannot be created or appended through the public block API; use bookmarks or rich-text links for API-created content. Source: `https://developers.notion.com/reference/block.md`.
- Rich-text `template_mention` values such as `today`, `now`, and `me` are populated template placeholders, not generic programmatic mention primitives. Source: `https://developers.notion.com/reference/rich-text.md`.

Child page:

```json
{"object":"block","type":"child_page","child_page":{"title":"Child title"}}
```

Table of contents:

```json
{"object":"block","type":"table_of_contents","table_of_contents":{"color":"default"}}
```

## Child-capable block types

Common child-capable types include:

```text
paragraph
quote
callout
toggle
bulleted_list_item
numbered_list_item
to_do
synced_block
column
column_list
table
table_row
child_page
child_database
meeting_notes
```

Always confirm current child support from the block reference before relying on less common types.

Meeting notes:

- `meeting_notes` replaced the older `transcription` block name in `2026-03-11` responses, but meeting-notes blocks are read-only; the API cannot create or update them. Source: `https://developers.notion.com/reference/block.md`.
- Query meeting notes with `POST /v1/blocks/meeting_notes/query` or retrieve existing meeting-notes blocks, then fetch `summary_block_id`, `notes_block_id`, and `transcript_block_id` children as needed. Sources: `https://developers.notion.com/reference/query-meeting-notes.md` and official OpenAPI operation `query-meeting-notes`.
- For page markdown reads, use `include_transcript=true` when transcript text is needed.

## Reading text

For rich-text blocks, concatenate `.plain_text` from the relevant rich-text array:

```text
paragraph.rich_text
heading_1.rich_text
heading_2.rich_text
heading_3.rich_text
heading_4.rich_text
bulleted_list_item.rich_text
numbered_list_item.rich_text
to_do.rich_text
toggle.rich_text
quote.rich_text
callout.rich_text
code.rich_text
```

Media blocks usually carry caption rich text plus one of:

```text
file.url + expiry_time
external.url
file_upload.id
```

## Updating and deleting

Update one block:

```text
PATCH /v1/blocks/{block_id}
```

Rules:

- Included fields replace the entire field value.
- Omitted fields remain unchanged.
- Children are not updated by parent update.
- Update page/database-specific fields with page/database/data-source endpoints, not block update.

Delete/trash one block:

```text
DELETE /v1/blocks/{block_id}
```

or patch `in_trash` where supported. In `2026-03-11`, use `in_trash`, not `archived`.

## Unsupported and unknown blocks

Notion can return unsupported block types as `unsupported`, and markdown export can return `<unknown .../>` placeholders. For unknown markdown blocks:

- try structured block retrieval by ID;
- confirm the connection has access;
- expect some UI block types to be partially or not supported by public API.
