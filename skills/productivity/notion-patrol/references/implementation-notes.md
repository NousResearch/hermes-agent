# Notion Patrol implementation notes

Use these notes when maintaining or extending the Notion Patrol skill.

## Durable workflow lessons

- Treat the patrol as **read-only by design**: use Notion API reads for pages, block children, database queries, and metadata only. Do not add Notion create/update/delete/comment behavior unless the user explicitly changes the requirement.
- Do not hardcode Notion integration tokens in `patrol.js` or examples. Read `NOTION_API_KEY`, `NOTION_TOKEN`, or `NOTION_API_TOKEN`; support `.env` loading from the current directory, `~/.hermes/.env`, and the skill directory.
- Verify target IDs before or during implementation using read-only Notion calls. A 32-character ID may be a page or a database; handle both paths defensively.
- Keep the implementation dependency-free for the original requirement: Node.js 18+ standard `fetch`, `node:test`, and built-in modules only.
- Make the script importable and testable: export pure helpers, guard CLI execution with `if (require.main === module)`, and run `main()` only for direct invocation.
- Preserve Excel-friendly CSV output: UTF-8 BOM, Japanese headers, and proper CSV escaping.
- For link checks, prefer `HEAD`; fall back to `GET` only on `405`; classify `2xx/3xx` as `OK`, other HTTP statuses and network failures as `NG`.
- Ignore Notion-internal URLs such as `notion.so` and `app.notion.com` unless the user explicitly asks to check internal Notion links.

## Verification commands

```bash
node --test ~/.hermes/skills/productivity/notion-patrol/scripts/patrol.test.js
node --check ~/.hermes/skills/productivity/notion-patrol/scripts/patrol.js
node ~/.hermes/skills/productivity/notion-patrol/scripts/patrol.js --help
```

For live runs, ensure credentials are configured and keep output sanitized; never print tokens.
