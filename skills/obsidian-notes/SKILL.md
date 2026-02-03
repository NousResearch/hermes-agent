---
name: obsidian
description: Read, search, and create notes in the Obsidian vault.
---

# Obsidian Vault

**Location:** `/home/teknium/Documents/Primary Vault`

Note: Path contains a space - always quote it.

## Read a note

```bash
cat "/home/teknium/Documents/Primary Vault/Note Name.md"
```

## List notes

```bash
# All notes
find "/home/teknium/Documents/Primary Vault" -name "*.md" -type f

# In a specific folder
ls "/home/teknium/Documents/Primary Vault/AI Research/"
```

## Search

```bash
# By filename
find "/home/teknium/Documents/Primary Vault" -name "*.md" -iname "*keyword*"

# By content
grep -rli "keyword" "/home/teknium/Documents/Primary Vault" --include="*.md"
```

## Create a note

```bash
cat > "/home/teknium/Documents/Primary Vault/New Note.md" << 'ENDNOTE'
# Title

Content here.
ENDNOTE
```

## Append to a note

```bash
echo "
New content here." >> "/home/teknium/Documents/Primary Vault/Existing Note.md"
```

## Wikilinks

Obsidian links notes with `[[Note Name]]` syntax. When creating notes, use these to link related content.
