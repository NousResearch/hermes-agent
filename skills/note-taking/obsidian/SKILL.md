---
name: obsidian
description: Read, search, and create notes in the Obsidian vault.
---

# Obsidian Vault

**Location:** Set via `OBSIDIAN_VAULT_PATH` environment variable (e.g. in `~/.hermes/.env`).

If unset, defaults to `~/Documents/Obsidian Vault`.

Note: Vault paths may contain spaces - always quote them.

## Preferred strategy

Do not start with broad shell searches across the whole machine.

Prefer this order:

1. If the user gives an absolute path, try to read that exact path first.
2. If the user gives only a filename, search inside the vault path only.
3. Use Hermes file tools first when available:
   - `read_file` for direct reads
   - `search_files` for filename/content search
4. Fall back to shell commands only when file tools are unavailable or insufficient.

## macOS + iCloud caveat

Many Obsidian vaults live in iCloud Drive under paths like:

```bash
~/Library/Mobile\ Documents/iCloud~md~obsidian/Documents/main_vault
```

If a file exists logically but reads fail from the terminal, the usual cause is
that iCloud has not downloaded the file locally yet. In that case:

- do not keep searching unrelated folders
- explain that the file may need to be opened in Finder/Obsidian first
- then retry the exact path

## Read a note

```bash
VAULT="${OBSIDIAN_VAULT_PATH:-$HOME/Documents/Obsidian Vault}"
cat "$VAULT/Note Name.md"
```

Preferred with Hermes file tools:

```text
read_file(path="/absolute/path/to/note.md")
```

## List notes

```bash
VAULT="${OBSIDIAN_VAULT_PATH:-$HOME/Documents/Obsidian Vault}"

# All notes
find "$VAULT" -name "*.md" -type f

# In a specific folder
ls "$VAULT/Subfolder/"
```

## Search

```bash
VAULT="${OBSIDIAN_VAULT_PATH:-$HOME/Documents/Obsidian Vault}"

# By filename
find "$VAULT" -name "*.md" -iname "*keyword*"

# By content
grep -rli "keyword" "$VAULT" --include="*.md"
```

Preferred with Hermes file tools:

```text
search_files(pattern="*keyword*", target="files", path=VAULT)
search_files(pattern="keyword", target="content", path=VAULT, file_glob="*.md")
```

If the user already gave a likely vault-relative folder such as `Notes/Side/...`,
search only within that subtree instead of the whole vault.

## Create a note

```bash
VAULT="${OBSIDIAN_VAULT_PATH:-$HOME/Documents/Obsidian Vault}"
cat > "$VAULT/New Note.md" << 'ENDNOTE'
# Title

Content here.
ENDNOTE
```

## Append to a note

```bash
VAULT="${OBSIDIAN_VAULT_PATH:-$HOME/Documents/Obsidian Vault}"
echo "
New content here." >> "$VAULT/Existing Note.md"
```

## Wikilinks

Obsidian links notes with `[[Note Name]]` syntax. When creating notes, use these to link related content.
