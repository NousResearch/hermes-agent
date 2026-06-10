---
name: obsidian
description: Read, search, create, and edit notes in the Obsidian vault. Hermes-level filesystem-first knowledge management with wikilinks, frontmatter, and structured notes.
platforms: [linux, macos, windows]
version: "1.1.0"
author: hermes
license: MIT
tags: [note-taking, knowledge-management, obsidian, vault, wikilinks, markdown]
---

# Obsidian Vault

Use this skill for filesystem-first Obsidian vault work: reading notes, listing notes, searching note files, creating notes, appending content, and adding wikilinks.

## Vault path

Resolve the vault path once at the start of vault operations. Priority order:

1. **`OBSIDIAN_VAULT_PATH`** environment variable (set in `~/.hermes/.env` or `/opt/data/.env`)
2. **Default**: `~/Documents/Obsidian Vault`
3. **Fallback**: `/opt/data/vault` (Linux server default)

Run this once to resolve:

```bash
echo "${OBSIDIAN_VAULT_PATH:-${HOME}/Documents/Obsidian Vault}"
```

File tools do not expand shell variables. Do not pass paths containing `$OBSIDIAN_VAULT_PATH` to `read_file`, `write_file`, `patch`, or `search_files`; resolve the vault path first and pass a concrete absolute path.

If the vault path is unknown, use `terminal` to resolve it. Once known, switch to file tools.

## Vault structure conventions

- `project-name/` — project folder with its own `README.md` index
- `project-name/dashboards/` — summary/dashboard notes
- `project-name/verticals/` or `project-name/sub-topic/` — categorized detail notes
- `project-name/templates/` — reusable templates
- Notes use YAML frontmatter for metadata: `tags`, `created`, `vertical`, `status`
- Inter-note linking via `[[Note Name]]` wikilinks

## Read a note

Use `read_file` with the resolved absolute path to the note. Prefer this over `cat` because it provides line numbers and pagination.

## List notes

Use `search_files` with `target: "files"` and the resolved vault path. Prefer this over `find` or `ls`.

- To list all markdown notes, use `pattern: "*.md"` under the vault path.
- To list a subfolder, search under that subfolder's absolute path.

## Search

Use `search_files` for both filename and content searches. Prefer this over `grep`, `find`, or `ls`.

- For filenames, use `search_files` with `target: "files"` and a filename `pattern`.
- For note contents, use `search_files` with `target: "content"`, the content regex as `pattern`, and `file_glob: "*.md"` when you want to restrict matches to markdown notes.

## Create a note

Use `write_file` with the resolved absolute path. Always include YAML frontmatter with at least `tags` and `created` fields. Always add wikilinks to related notes.

## Append to a note

Prefer a native file-tool workflow:

- Read the target note with `read_file`.
- Use `patch` for an anchored append when there is stable context (e.g., after a heading).
- Use `write_file` when rewriting the whole note is clearer.

## Targeted edits

Use `patch` for focused note changes. Prefer this over shell text rewriting.

## Wikilinks

Obsidian links notes with `[[Note Name]]` syntax. Notes in subfolders: `[[folder/Note Name]]`. Aliases: `[[Note Name|Display Text]]`. When creating notes, link related content.

## Project setup pattern

To bootstrap a new project in the vault:

1. Create `project-name/README.md` with YAML frontmatter and wikilinks
2. Create subdirectories as needed (dashboards, verticals, templates)
3. Populate with structured notes from research/source data
4. Link back to vault root README.md

## Diagnostics

- Check vault health: count notes, verify wikilinks resolve, check frontmatter validity
- Use `search_files` with `target: "files"`, `pattern: "*.md"` under vault root
- Verify `.obsidian/app.json` exists for vault config
