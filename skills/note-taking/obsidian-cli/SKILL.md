---
name: obsidian-cli
description: Drive the live Obsidian app from the command line.
platforms: [linux, macos, windows]
version: 1.1.0
author: Harish Kukreja (counterposition), Hermes Agent (adapted from Steph Ango's official Obsidian skills)
license: MIT
metadata:
  hermes:
    tags: [obsidian, cli, plugin-development, theme-development, note-taking]
    related_skills: [obsidian, obsidian-markdown, obsidian-bases, json-canvas]
prerequisites:
  commands: [obsidian]
---

# Obsidian CLI Skill

Operate a live Obsidian instance through its official `obsidian` CLI: note
operations, search, tasks, properties, and a plugin/theme development loop
with reload, error capture, screenshots, and DOM inspection. Adapted from
Steph Ango's MIT-licensed official Obsidian skills.

## When to Use

- The task needs the *running* Obsidian app: opening notes, live search,
  daily notes, tasks, properties, or vault commands
- Developing or debugging an Obsidian plugin or theme

Route elsewhere when the app is not required: generic vault filesystem work
→ `obsidian`; `.md` authoring semantics → `obsidian-markdown`; `.base`
files → `obsidian-bases`; `.canvas` files → `json-canvas`.

## Prerequisites

- The `obsidian` CLI installed and on `PATH`
  ([install docs](https://help.obsidian.md/cli))
- The Obsidian desktop app installed. It does not have to be running: if
  it is not, the first CLI command launches it (expect a startup delay
  and a GUI window — there is no headless mode)

## How to Run

Run `obsidian <command>` via the `terminal` tool. `obsidian help` lists
every command and is always up to date; full docs live at
https://help.obsidian.md/cli.

Syntax rules:

- **Parameters** take `key=value`; quote values with spaces:
  `obsidian create name="My Note" content="Hello world"`
- **Flags** are bare words: `obsidian create name="My Note" overwrite`
- Use `\n` and `\t` escapes for multiline content
- **File targeting**: `file=<name>` resolves like a wikilink (no path or
  extension needed); `path=<path>` is exact from the vault root. Without
  either, the active file is used.
- **Vault targeting** (in order): `vault=<name>` or `vault=<id>` as the
  first parameter pins a vault; otherwise, if the terminal's working
  directory is inside a vault, that vault is used; otherwise the currently
  active vault.

## Quick Reference

```bash
obsidian read file="My Note"
obsidian create name="New Note" content="# Hello" template="Template"
obsidian append file="My Note" content="New line"
obsidian search query="search term" limit=10
obsidian daily:read
obsidian daily:append content="- [ ] New task"
obsidian property:set name="status" value="done" file="My Note"
obsidian tasks daily todo
obsidian tags sort=count counts
obsidian backlinks file="My Note"
```

Use `--copy` on any command to copy output to the clipboard, the `open`
flag where supported to open the affected file in Obsidian (files are not
opened by default), and `total` on list commands to get a count.

## Procedure

Plugin/theme develop-test cycle — after each code change:

1. Reload the plugin: `obsidian plugin:reload id=my-plugin`
2. Check for errors, fix, and repeat from step 1 if any appear:
   `obsidian dev:errors`
3. Verify visually: `obsidian dev:screenshot path=screenshot.png` or
   `obsidian dev:dom selector=".workspace-leaf" text`
4. Check console output: `obsidian dev:console level=error`

Additional developer commands:

```bash
obsidian eval code="app.vault.getFiles().length"          # run JS in app
obsidian dev:css selector=".workspace-leaf" prop=background-color
obsidian dev:mobile on                                    # mobile emulation
```

`obsidian help` lists further developer commands, including CDP and
debugger controls.

## Pitfalls

- The first command launches Obsidian when it is not already running —
  on a headless machine (no display) the CLI cannot work at all.
- Multi-vault setups: without `vault=`, the vault is inferred from the
  working directory or falls back to the active vault — pin it explicitly
  in scripts.
- `file=` resolves wikilink-style and may match an unexpected note; use
  `path=` when exactness matters.
- Prefer native file tools (`read_file`, `write_file`, `patch`) for bulk
  file edits — the CLI is for app-side state (open notes, live search,
  plugin reloads), not mass file rewriting.

## Verification

Check the command's stdout and exit code in `terminal`. For state-changing
operations, confirm the result: re-read the note (`obsidian read` or
`read_file`), or take a screenshot with `obsidian dev:screenshot` for
UI-facing changes. For plugin work, `obsidian dev:errors` must come back
clean after a reload.

## References

- [Obsidian CLI docs](https://help.obsidian.md/cli)
- Source: [kepano/obsidian-skills](https://github.com/kepano/obsidian-skills) by Steph Ango,
  MIT — full license in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)
