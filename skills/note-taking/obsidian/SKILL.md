---
name: obsidian
description: Read, search, create, and edit notes in the Obsidian vault.
platforms: [linux, macos, windows]
---

# Obsidian Vault

Use this skill for filesystem-first Obsidian vault work: reading notes, listing notes, searching note files, creating notes, appending content, and adding wikilinks.

## Detecting Obsidian

Before modifying any file in the vault, check whether Obsidian is currently running. When Obsidian has a vault open, macOS FSEvents may not notify it of external file changes. This can make modified files appear to "disappear" from search, graph view, and other Obsidian features until the app is restarted.

Use `terminal` with a command such as `pgrep -x Obsidian` (macOS/Linux) or `tasklist /FI "IMAGENAME eq Obsidian.exe" 2>NUL | findstr /I Obsidian` (Windows) to detect an active Obsidian process. On macOS, `pgrep -l Obsidian` can also be used for a verbose check.

If Obsidian is running AND the file you intend to write is under a known vault path (the `OBSIDIAN_VAULT_PATH` environment variable, or the `~/Documents/Obsidian Vault` fallback), warn the user before making changes. Describe what you found and suggest they either quit Obsidian or accept that changes may not be reflected until the next restart.

Examples:

```
pgrep -x Obsidian  # exit code 0 → running, non-zero → not running
```

On Windows (when terminal supports it):

```
tasklist /FI "IMAGENAME eq Obsidian.exe" 2>NUL | findstr /I Obsidian
```

This check is recommended for any write-oriented operation (`write_file`, `patch`, `terminal` commands that edit files) operating inside the vault path. Read-only operations (`read_file`, `search_files`) do not need the check.

## Vault path

Use a known or resolved vault path before calling file tools.

The documented vault-path convention is the `OBSIDIAN_VAULT_PATH` environment variable, for example from `${HERMES_HOME:-~/.hermes}/.env`. If it is unset, use `~/Documents/Obsidian Vault`.

File tools do not expand shell variables. Do not pass paths containing `$OBSIDIAN_VAULT_PATH` to `read_file`, `write_file`, `patch`, or `search_files`; resolve the vault path first and pass a concrete absolute path. Vault paths may contain spaces, which is another reason to prefer file tools over shell commands.

If the vault path is unknown, `terminal` is acceptable for resolving `OBSIDIAN_VAULT_PATH` or checking whether the fallback path exists. Once the path is known, switch back to file tools.

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

Use `write_file` with the resolved absolute path and the full markdown content. Prefer this over shell heredocs or `echo` because it avoids shell quoting issues and returns structured results.

## Append to a note

Prefer a native file-tool workflow when it is not awkward:

- Read the target note with `read_file`.
- Use `patch` for an anchored append when there is stable context, such as adding a section after an existing heading or appending before a known trailing block.
- Use `write_file` when rewriting the whole note is clearer than constructing a fragile patch.

For an anchored append with `patch`, replace the anchor with the anchor plus the new content.

For a simple append with no stable context, `terminal` is acceptable if it is the clearest safe option.

## Targeted edits

Use `patch` for focused note changes when the current content gives you stable context. Prefer this over shell text rewriting.

## Wikilinks

Obsidian links notes with `[[Note Name]]` syntax. When creating notes, use these to link related content.
