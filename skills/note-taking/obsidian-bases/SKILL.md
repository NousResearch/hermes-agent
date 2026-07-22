---
name: obsidian-bases
description: Create and edit .base database views, filters, and formulas.
platforms: [linux, macos, windows]
version: 1.1.0
author: Harish Kukreja (counterposition), Hermes Agent (adapted from Steph Ango's official Obsidian skills)
license: MIT
metadata:
  hermes:
    tags: [obsidian, note-taking, yaml, bases, database-views]
    related_skills: [obsidian, obsidian-markdown, obsidian-cli, json-canvas]
---

# Obsidian Bases Skill

Create and edit Obsidian Bases: YAML `.base` files that render database-like
views (tables, cards, lists, maps) over the notes in a vault, with filters,
computed formula properties, and summaries. Adapted from Steph Ango's
MIT-licensed official Obsidian skills.

## When to Use

- Creating or editing `.base` files
- The user mentions Bases, table/card views over notes, filters, formulas,
  or summaries in Obsidian

Route elsewhere for broad vault work (`obsidian`), regular note authoring
(`obsidian-markdown`), `.canvas` files (`json-canvas`), or driving the
running app (`obsidian-cli`).

## Prerequisites

None — `.base` files are plain YAML and no running Obsidian app is
required (only Obsidian can render the views). Resolve the vault path
first (see the `obsidian` skill).

## How to Run

Read `.base` files with `read_file`, create them with `write_file`, and
make focused edits with `patch`. Find existing bases with `search_files`
using `pattern: "*.base"` and `target: "files"`.

## Quick Reference

### Schema

```yaml
# Global filters apply to ALL views in the base.
# A filter is either a single expression string, or an object with
# exactly ONE key (and / or / not) whose list items are themselves filters.
filters:
  and:
    - 'status == "active"'
    - not:
        - 'file.hasTag("archived")'

# Formula properties computed for every note, usable in all views
formulas:
  formula_name: 'expression'

# Display names and settings for properties
properties:
  property_name:
    displayName: "Display Name"
  formula.formula_name:
    displayName: "Formula Display Name"

# Custom summary formulas
summaries:
  custom_summary_name: 'values.mean().round(3)'

# One or more views
views:
  - type: table            # table | cards | list | map
    name: "View Name"
    limit: 10              # optional
    groupBy:               # optional
      property: property_name
      direction: ASC       # ASC | DESC
    filters:               # view-specific, same rules as global filters
      and:
        - 'status == "active"'
    order:                 # properties to display, in order
      - file.name
      - property_name
      - formula.formula_name
    summaries:             # map properties to summary formulas
      property_name: Average
```

### Filters

Expressions compare properties with `==`, `!=`, `>`, `<`, `>=`, `<=` and
combine with `&&`, `||`, `!`. Common predicates: `file.hasTag("x")`,
`file.hasLink("Note")`, `file.inFolder("Folder")`. Nest by making a list
item an `and`/`or`/`not` object.

### Properties

1. **Note properties** — frontmatter values: `note.author` or just `author`
2. **File properties** — metadata: `file.name`, `file.basename`,
   `file.path`, `file.folder`, `file.ext`, `file.size`, `file.ctime`,
   `file.mtime`, `file.tags`, `file.links`, `file.backlinks`, `file.embeds`
3. **Formula properties** — computed values: `formula.my_formula`

`this` refers to the base file itself in the main content area, to the
embedding file when embedded, and to the active file in the sidebar.

### Formulas

```yaml
formulas:
  total: "price * quantity"
  status_icon: 'if(done, "✅", "⏳")'
  created: 'file.ctime.format("YYYY-MM-DD")'
  days_old: '((now() - file.ctime) / 86400000).round(0)'
  days_until_due: 'if(due_date, ((date(due_date) - today()) / 86400000).round(0), "")'
```

Key functions: `date()`, `now()`, `today()`, `if(condition, then, else?)`,
`duration()`, `file()`, `link()`. The function reference for all types
(Date, String, Number, List, File, Link, Object, RegExp) is in
[FUNCTIONS_REFERENCE.md](references/FUNCTIONS_REFERENCE.md).

Subtracting two dates yields the difference in **milliseconds** — divide by
86400000 to get days (see `days_old` above). Add or subtract durations with
strings: `now() + "1 day"`, `today() + "7d"`; `duration()` is only needed
for arithmetic on parsed durations themselves.

### Views and summaries

View types: `table`, `cards`, `list`, and `map` (map needs lat/long
properties and the Maps community plugin). Built-in summary formulas
include `Average`, `Min`, `Max`, `Sum`, `Median`, `Stddev`, `Earliest`,
`Latest`, `Checked`, `Empty`, `Filled`, `Unique`. Worked examples (task
tracker, reading list, daily-notes index) are in
[EXAMPLES.md](references/EXAMPLES.md).

### Embedding

```markdown
![[MyBase.base]]              Embed base in a note
![[MyBase.base#View Name]]    Embed a specific view
```

## Procedure

1. Create the `.base` file in the vault with `write_file`.
2. Define scope with global `filters` (tag, folder, property, or date).
3. Add computed properties in `formulas` as needed.
4. Configure one or more `views`, each with `order` listing the properties
   to display; add `groupBy`, view filters, and `summaries` as needed.
5. Validate YAML syntax and references (see Verification).
6. If a running Obsidian instance is available, open the file to confirm
   the view renders (the `obsidian-cli` skill can do this).

## Pitfalls

- **Quoting**: wrap formulas containing double quotes in single quotes:
  `'if(done, "Yes", "No")'`. Quote any YAML string containing `:`, `#`,
  `[`, `{`, or other special characters.
- **Filter objects take exactly one key** — `and`, `or`, or `not`. Do not
  put several of them at the same level of the same object; nest instead.
- **Date subtraction returns milliseconds**, not days: divide by 86400000
  before rounding — `'((date(due) - today()) / 86400000).round(0)'`.
- **Missing null guards**: properties may be absent on some notes. Guard
  with `if()`: `'if(due, ((date(due) - today()) / 86400000).round(0), "")'`.
- **Undefined formula references**: every `formula.X` used in `order`,
  `properties`, or `summaries` must be defined under `formulas` — unknown
  references fail silently in Obsidian.

## Verification

Parse the file to confirm it is valid YAML (via `terminal`:
`python -c 'import yaml, sys; yaml.safe_load(open(sys.argv[1]))' file.base`).
Then re-read it with `read_file` and check: each filter object has exactly
one of `and`/`or`/`not`; every `formula.X` reference is defined; every view
has a `type` (and a distinct `name` when there are multiple views). Finally,
open it in Obsidian if available — a YAML error banner means a quoting
problem (see Pitfalls).

## References

- Worked examples: [references/EXAMPLES.md](references/EXAMPLES.md)
- Full function reference: [references/FUNCTIONS_REFERENCE.md](references/FUNCTIONS_REFERENCE.md)
- [Bases Syntax](https://help.obsidian.md/bases/syntax) ·
  [Functions](https://help.obsidian.md/bases/functions) ·
  [Views](https://help.obsidian.md/bases/views)
- Source: [kepano/obsidian-skills](https://github.com/kepano/obsidian-skills) by Steph Ango,
  MIT — full license in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)
