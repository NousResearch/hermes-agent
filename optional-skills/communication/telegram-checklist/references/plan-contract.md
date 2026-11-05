# Checklist plan contract

Use `write_file` to prepare JSON, then invoke the helper's `plan --file plan.json`
through `terminal` before previewing `create --from-plan plan.json --dry-run`.
Both paths are offline; only explicit create without `--dry-run` sends.

```json
{
  "target": {"chat": -1001234567890, "thread": 33},
  "title": "Weekly tasks",
  "shared": false,
  "collected_from_chat": true,
  "tasks": [
    {
      "text": "Review the proposal: https://t.me/c/1234567890/33/123",
      "sources": [
        {
          "link": "https://t.me/c/1234567890/33/123",
          "topic": "Proposals (33)",
          "message_id": 123,
          "media": "document",
          "says": "The sender requests review of the attached proposal."
        }
      ]
    }
  ]
}
```

- `target.chat` is required; `thread` is optional except under a topic grant.
  Use `"me"` for Saved Messages and omit `thread` there.
- `shared` and `collected_from_chat` must be JSON booleans, never strings.
  Both default false. Shared enables others to append and complete.
- `tasks` contains objects with non-empty string `text`. Source metadata is
  optional for dictated tasks; required for chat-derived tasks.
- With `collected_from_chat: true`, each task needs a source object whose direct
  `https://t.me/` URL appears as a complete standalone token inside the text.
  Prefixes, nested foreign URLs, and visually misleading Unicode suffixes
  do not satisfy this check. The agent still verifies semantic evidence.
- Limits: 30 tasks, title 255, each task 200 UTF-16 units. Plans reject
  normalized-identical task texts (compatibility Unicode, case, invisible
  format characters, whitespace). Merge semantic duplicates manually.
- Inspect every source's media before admitting a task. Allowed media labels:
  `text`, `photo`, `document`, `video`, `audio`, `voice`, `webpage`, `rich_media`.
  Unknown labels warn; they do not replace evidence inspection.
- `--from-plan` is exclusive of direct title/task/target/sharing options.
  The preview's `would_send` is the actual validated payload.

Never use the example IDs as authorization. Use only user-requested destinations
already present in the owning profile's allowlist.
