---
name: macos-spotlight
description: Search macOS globally using Spotlight mdfind.
author: bytedance
platforms: [macos]
---

# macOS Spotlight Skill

This skill performs blazing-fast global file searches on macOS using the native Spotlight index (`mdfind`). It replaces slow, recursive directory traversal with instant, metadata-aware querying.

## When to Use

- When you need to find a file but do not know its exact directory on the Mac.
- When searching for specific text content inside PDFs, Word documents, or presentations globally.
- When filtering files by modification date across the entire filesystem.

## Prerequisites

- Operating System: **macOS** only.
- The `mdfind` command-line tool (built into macOS).

## Procedure

Use the `terminal` tool to execute the helper script `scripts/macos_mdfind.py`. Do NOT run raw `mdfind` commands directly in the shell to prevent context window token overflow. The python wrapper safely limits output lines.

### Options:
- `--name "text"`: Search by filename (case-insensitive).
- `--content "text"`: Search by file content (including rich documents).
- `--type "ext"`: Filter by file extension (e.g., `pdf`, `md`).
- `--days N`: Restrict to files modified within the last N days.
- `--onlyin "/path"`: Restrict search to a specific absolute directory.
- `--limit N`: Max results (default is 50).

### Examples

**Search for a PDF containing the word "budget" anywhere on the Mac:**
```bash
python3 scripts/macos_mdfind.py --content "budget" --type "pdf"
```

**Find markdown files modified in the last 7 days:**
```bash
python3 scripts/macos_mdfind.py --type "md" --days 7
```

**Find a project folder or file by name within the Documents folder:**
```bash
python3 scripts/macos_mdfind.py --name "Hermes" --onlyin "$HOME/Documents"
```

## Verification

If the script returns paths, use the standard `read_file` tool to read the contents of the target file, or the `terminal` tool to interact with the located files.
