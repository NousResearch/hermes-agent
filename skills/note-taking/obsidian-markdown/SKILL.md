---
name: obsidian-markdown
description: Write Obsidian notes with wikilinks, embeds, and callouts.
platforms: [linux, macos, windows]
version: 1.1.0
author: Harish Kukreja (counterposition), Hermes Agent (adapted from Steph Ango's official Obsidian skills)
license: MIT
metadata:
  hermes:
    tags: [obsidian, markdown, note-taking, wikilinks, callouts, frontmatter]
    related_skills: [obsidian, obsidian-cli, obsidian-bases, json-canvas]
---

# Obsidian Flavored Markdown Skill

Author and edit Obsidian Flavored Markdown, which extends CommonMark and GFM
with wikilinks, embeds, callouts, properties, comments, and highlights. This
skill covers only the Obsidian-specific extensions — standard Markdown
(headings, lists, tables, code blocks) is assumed knowledge. Adapted from
Steph Ango's MIT-licensed official Obsidian skills.

## When to Use

- Notes that need Obsidian-specific syntax: wikilinks, embeds, callouts,
  frontmatter properties, tags, comments, or highlights
- Converting plain Markdown into Obsidian-idiomatic notes

Plain note creation, reading, listing, and searching stays with the generic
`obsidian` skill — load this one for the syntax layer. Route `.base`
database views → `obsidian-bases`; `.canvas` files → `json-canvas`; driving
the running app or plugin development → `obsidian-cli`.

## Prerequisites

None — notes are plain text files and no running Obsidian app is required.
Resolve the vault path first (see the `obsidian` skill): the
`OBSIDIAN_VAULT_PATH` environment variable, falling back to
`~/Documents/Obsidian Vault`.

## How to Run

Use the native file tools: `read_file` to inspect a note, `write_file` to
create one, `patch` for targeted edits with stable context, and
`search_files` to find notes by name or content. Pass concrete absolute
paths — file tools do not expand `$OBSIDIAN_VAULT_PATH`.

## Quick Reference

### Internal links (wikilinks)

```markdown
[[Note Name]]                Link to note
[[Note Name|Display Text]]   Custom display text
[[Note Name#Heading]]        Link to heading
[[Note Name#^block-id]]      Link to block
[[#Heading in same note]]    Same-note heading link
```

Define a block ID by appending `^block-id` to a paragraph or to a single
list item. To reference a whole list or quote, place the block ID on its
own line after the block.

### Embeds

Prefix any wikilink with `!` to embed its content inline:

```markdown
![[Note Name]]               Embed full note
![[Note Name#Heading]]       Embed section
![[image.png|300]]           Embed image with width
![[document.pdf#page=3]]     Embed PDF page
![[MyBase.base#View Name]]   Embed a Bases view
```

See [EMBEDS.md](references/EMBEDS.md) for audio, video, Bases, search
embeds, and external images.

### Callouts

```markdown
> [!note]
> Basic callout.

> [!warning] Custom Title
> Callout with a custom title.

> [!faq]- Collapsed by default
> Foldable callout (- collapsed, + expanded).
```

Common types: `note`, `tip`, `warning`, `info`, `example`, `quote`, `bug`,
`danger`, `success`, `failure`, `question`, `abstract`, `todo`. See
[CALLOUTS.md](references/CALLOUTS.md) for aliases, nesting, and custom CSS
callouts.

### Properties (frontmatter)

```yaml
---
title: My Note
date: 2026-01-15
tags:
  - project
aliases:
  - Alternative Name
cssclasses:
  - custom-class
---
```

See [PROPERTIES.md](references/PROPERTIES.md) for all property types and
tag syntax rules.

### Other Obsidian syntax

```markdown
#tag  #nested/tag            Inline tags
==Highlighted text==         Highlight
%%hidden comment%%           Hidden in reading view
$e^{i\pi} + 1 = 0$           Inline LaTeX ($$ ... $$ for blocks)
Text with a footnote[^1].    Footnotes ([^1]: content)
```

Mermaid diagrams go in fenced `mermaid` code blocks; add
`class NodeName internal-link;` to link diagram nodes to notes.

## Procedure

Creating an Obsidian note:

1. Add frontmatter properties (title, tags, aliases) at the top of the file.
2. Write content using standard Markdown structure plus the Obsidian syntax
   above.
3. Link related notes with `[[wikilinks]]`; use `[text](url)` Markdown links
   for external URLs.
4. Embed supporting content (notes, images, PDFs) with `![[...]]`.
5. Add callouts for highlighted information with `> [!type]`.
6. Write the note with `write_file`, or apply focused changes to an existing
   note with `patch`.

## Pitfalls

- Both `[[wikilinks]]` and standard Markdown links work for vault-internal
  targets; wikilinks are Obsidian's compact default and what this skill
  uses. Match the vault's existing convention when editing.
- Block IDs for lists and quotes must sit on a separate line after the
  block, not appended to the last line.
- Tags cannot start with a number; allowed characters are letters, numbers,
  underscores, hyphens, and forward slashes.
- Frontmatter must be the very first thing in the file, delimited by `---`
  lines, and must be valid YAML — quote values containing `:` or `#`.
- `%%` comments are invisible in reading view but still ship with the file.

## Verification

Re-read the result with `read_file` and check: frontmatter parses as YAML
between the opening `---` pair; every `[[wikilink]]` target exists in the
vault (check with `search_files` when correctness matters); `%%` markers
are balanced. If a running Obsidian instance is available, confirm
rendering in reading view (the `obsidian-cli` skill can open the note).

## References

- [Obsidian Flavored Markdown](https://help.obsidian.md/obsidian-flavored-markdown)
- [Internal links](https://help.obsidian.md/links)
- [Embed files](https://help.obsidian.md/embeds)
- [Callouts](https://help.obsidian.md/callouts)
- [Properties](https://help.obsidian.md/properties)
- Source: [kepano/obsidian-skills](https://github.com/kepano/obsidian-skills) by Steph Ango,
  MIT — full license in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)
