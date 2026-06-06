# Notion Markdown Workflows

Sources:

- `https://developers.notion.com/guides/data-apis/working-with-markdown-content.md`
- `https://developers.notion.com/guides/data-apis/enhanced-markdown.md`
- `https://developers.notion.com/reference/retrieve-page-markdown.md`
- `https://developers.notion.com/reference/update-page-markdown.md`
- `https://developers.notion.com/reference/post-page.md`

## Surfaces

Create page from markdown:

```text
POST /v1/pages
```

Read page markdown:

```text
GET /v1/pages/{page_id}/markdown
```

Update page markdown:

```text
PATCH /v1/pages/{page_id}/markdown
```

Markdown is often better than block JSON for agents. Use block APIs when you need exact structure, unsupported block details, or precise media/object manipulation.

## Create page from markdown

Body shape:

```json
{
  "parent": {"type": "page_id", "page_id": "..."},
  "markdown": "# Title\n\nBody"
}
```

Rules:

- `markdown` is mutually exclusive with `content`/`children`.
- `template` cannot be combined with `children`.
- If title property is omitted, Notion can derive title from first H1.
- Under a data source, properties must match schema.

Shell quoting pitfall: send JSON with escaped newlines (`\n`). Do not let the shell convert them to literal newlines inside JSON.

## Read page markdown

Endpoint response includes a `page_markdown` object with:

- `object`
- `id`
- `markdown`
- `truncated`
- `unknown_block_ids`

Query parameter:

- `include_transcript=true|false` for meeting note transcripts; default false.

Unknown blocks appear when:

- page exceeds record/block limits around very large pages;
- content is not shared with connection;
- block type is unsupported by markdown conversion;
- Notion needs to protect structure such as child page/database content.

If `truncated` is true, fetch IDs in `unknown_block_ids` by calling the same endpoint on those IDs, or use the block API. A permission-denied unknown block can return `404 object_not_found`.

File/media URLs in markdown output are signed and expire. Re-fetch for fresh URLs.

## Update page markdown

Preferred commands:

### `update_content`

Exact search/replace operations.

```json
{
  "command": {
    "type": "update_content",
    "content_updates": [
      {"old_str": "Status: draft", "new_str": "Status: final"}
    ]
  }
}
```

Rules:

- `old_str` matching is exact and case-sensitive.
- Max `content_updates`: 100.
- If `old_str` matches multiple places, request fails unless `replace_all_matches: true` is set deliberately.

### `replace_content`

Replace whole page body.

```json
{
  "command": {
    "type": "replace_content",
    "new_str": "# New complete body"
  }
}
```

Deletion guard:

- Child pages/databases are protected by default.
- Set `allow_deleting_content: true` inside the command when intentionally deleting protected child content.

Legacy commands still exist but are not preferred:

- `insert_content`
- `replace_content_range`

They use selection/range strings and are more fragile; expect deprecation risk.

Update caveats:

- Update response returns full page markdown after update.
- Transcript text cannot be updated even if retrieved with `include_transcript=true`.
- Updates cannot target databases or non-page blocks.
- Synced page/unsupported targets can fail validation.

## Enhanced markdown syntax essentials

Enhanced markdown extends CommonMark with XML-like tags and attributes.

Indentation:

- Use tabs for child nesting.
- Child blocks are one tab deeper.

Escaping outside code blocks:

```text
\ * ~ ` $ [ ] < > { } | ^
```

Do not escape inside code blocks.

Headings:

```md
# H1
## H2
### H3
#### H4
```

H5/H6 collapse to H4. Toggle headings can use attributes such as `{toggle="true"}`.

Lists and todos:

```md
- bullet
1. numbered
- [ ] todo
- [x] done
```

Quote with line breaks:

```md
> first line<br>second line
```

Common block tags:

```md
<details>
<summary>Toggle title</summary>
	Nested content
</details>

<callout icon="💡" color="blue_bg">
	Important note
</callout>

<columns>
	<column>Left</column>
	<column>Right</column>
</columns>

<table_of_contents/>
<empty-block/>
```

Media tags include markdown images and XML-like audio/video/file/pdf tags. Re-fetch file URLs when needed because Notion-hosted file URLs expire.

Rich text:

- `**bold**`
- `*italic*`
- `~~strike~~`
- underline via `<span underline="true">text</span>`
- inline code
- links
- inline math
- colors via `<span color="blue">text</span>` and background colors with `_bg` suffix.

Mentions:

- `<mention-user .../>`
- `<mention-page ...>`
- `<mention-database ...>`
- `<mention-data-source ...>`
- `<mention-agent ...>`
- `<mention-date .../>`

Custom emoji:

```md
:emoji_name:
```

Citations:

```md
[^https://example.com]
```

## When not to use markdown

Use structured block/data-source APIs when:

- You need exact block IDs and structural edits.
- You must preserve unsupported blocks.
- You need a precise page-property update.
- You are manipulating files/media objects directly.
- You need to traverse child pages/databases with access checks.
