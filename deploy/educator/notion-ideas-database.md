# Ideas board — Notion database

The teammate's team-visible record of what it proposed and what was decided,
and why (design: "Ideas board"). A human creates it once, under the
teammate's home page, and shares it with the teammate's integration.

## Properties

| Property | Type | Values / notes | Who writes it |
|---|---|---|---|
| `Name` | title | the idea in one line | teammate on create |
| `Status` | select | `Proposed`, `Accepted`, `Rejected`, `Parked`, `In progress`, `Done` | teammate creates at `Proposed`; **humans** set `Accepted` / `Rejected` / `Parked`; teammate moves **its own** rows `Accepted → In progress → Done` |
| `Type` | select | `content gap`, `stale content`, `process`, `new asset/format`, `research finding`, `self-improvement`, `new job`, `autonomy` | teammate |
| `Rationale` | text | why it matters | teammate |
| `Evidence` | text | links (articles, tickets, PRs, data) | teammate |
| `Effort` | select | `S`, `M`, `L` | teammate |
| `Owner` | person | who acts on it once accepted | humans |
| `Decision note` | text | the human's reason for accept / reject / park — the teammate reads it and learns | **humans only** |

`Created by`, `Created time` and `Last edited time` are Notion's own; the
write guard uses the page's `created_by` to tell the teammate's rows from
anyone else's (it may only move rows its integration created).

`autonomy` is the Type the teammate uses to propose its own graduation for
an action (design: "Autonomy levels → Evidence for graduation").

## How the guard reads it

These names are data in `config.yaml` (`notion.write_guard.ideas`):

```yaml
ideas:
  data_sources: ["<this database's data source id>"]
  status_property: Status
  create_status: Proposed
  own_transitions:
    Accepted: [In progress]
    In progress: [Done]
  human_statuses: [Accepted, Rejected, Parked]
  human_properties: [Decision note]
```

Rename a property or a status here and the config must change with it.

## Create it

**By hand:** under the teammate's home page, `/database` → full page, name it
`Ideas`, add the properties above (exact names and option spellings), then
`•••` → Connections → add the teammate's integration.

**By API** (an admin's own integration, not the teammate's — the teammate's
write guard refuses schema changes). Notion API version `2025-09-03`:

```bash
curl -sS https://api.notion.com/v1/databases \
  -H "Authorization: Bearer $ADMIN_NOTION_TOKEN" \
  -H "Notion-Version: 2025-09-03" -H "Content-Type: application/json" \
  -d @- <<'JSON'
{
  "parent": {"type": "page_id", "page_id": "<teammate home page id>"},
  "title": [{"type": "text", "text": {"content": "Ideas"}}],
  "initial_data_source": {
    "properties": {
      "Name": {"title": {}},
      "Status": {"select": {"options": [
        {"name": "Proposed", "color": "gray"}, {"name": "Accepted", "color": "green"},
        {"name": "Rejected", "color": "red"}, {"name": "Parked", "color": "brown"},
        {"name": "In progress", "color": "blue"}, {"name": "Done", "color": "purple"}]}},
      "Type": {"select": {"options": [
        {"name": "content gap"}, {"name": "stale content"}, {"name": "process"},
        {"name": "new asset/format"}, {"name": "research finding"},
        {"name": "self-improvement"}, {"name": "new job"}, {"name": "autonomy"}]}},
      "Rationale": {"rich_text": {}},
      "Evidence": {"rich_text": {}},
      "Effort": {"select": {"options": [{"name": "S"}, {"name": "M"}, {"name": "L"}]}},
      "Owner": {"people": {}},
      "Decision note": {"rich_text": {}}
    }
  }
}
JSON
```

The response's `data_sources[0].id` is `NOTION_IDEAS_DATA_SOURCE_ID` in
`values.yaml` (a bare UUID — not `collection://…`).
