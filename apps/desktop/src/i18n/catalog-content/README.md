# Localized catalog content

Display-only translations for the **dynamic metadata** the Capabilities page shows: skill, plugin,
MCP server, connector, tool and toolset names and descriptions, plus category labels. The UI
strings around them already live in the typed catalogs (`en.ts`, `ru.ts`, …); this layer covers
text that arrives from the backend or a public catalog feed and so cannot be a typed key.

## Rules

- Items are matched by their **stable technical id** (skill `name`, plugin key, MCP server name,
  tool name, toolset name). Ids, install identifiers, tool names and API calls are never changed.
- Language `en` (or any language without a dictionary) shows the original text. An item with no
  translation also shows the original. Nothing is machine-translated at display time.
- A translated entry keeps its original title and description: the detail view shows them under
  "Original (English)" together with the technical id, and both languages are searchable.
- Skill/plugin source files (`SKILL.md`, `plugin.yaml`) and MCP schemas are never rewritten.

## Files

| File | Role |
|---|---|
| `<lang>.json` | One dictionary per language, loaded lazily (own chunk, English never loads it) |
| `dictionaries.ts` | Registers the lazy loader for each language |
| `index.ts` | Lookup, fallback, search text, `useContentDictionary` support |
| `use-content-dictionary.ts` | React hook: starts the load and re-renders when it lands |

## Dictionary format

```jsonc
{
  "skill":   { "<id>": { "name": "…", "description": "…", "keywords": ["…"] } },
  "plugin":  { "<id>": { … } },
  "mcp":     { "<id>": { … } },
  "tool":    { "<id>": { … } },
  "toolset": { "<id>": { … } },
  "connector": { "<slug>": { … } },
  "categories": { "skills": { "<slug>": "…" }, "plugins": {}, "toolsets": {}, "labels": { "<english label>": "…" } },
  "pluginCategories": { "<key>": { "label": "…", "blurb": "…" } }
}
```

`keywords` are extra search words in the target language (synonyms people actually type).
Keep brand names, ids, commands, paths and URLs in Latin script; a connector keeps its brand title
and only gains a translated description.

## Adding a language

1. Add `<lang>.json` in this folder.
2. Register it in `dictionaries.ts`.
3. That is all: no component changes. Partial dictionaries are fine.

## User overlay without rebuilding

Any key above can also be supplied through the backend language pack
(`$HERMES_HOME/locales/<lang>.desktop.yaml`, flat dotted keys such as
`catalogContent.skill.my-skill.name`). Overlay entries win over the bundled dictionary and survive
app updates because they live outside the app.

## Tests

`catalog-content.test.ts` covers lookup, fallback for unknown ids and other languages, bilingual
search, stable ids, connector/MCP cards, toolsets and the overlay.
