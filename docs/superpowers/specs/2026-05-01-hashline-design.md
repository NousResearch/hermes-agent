# Hashline Line-Anchor Editing

**Date:** 2026-05-01
**Branch:** feat/hashline
**Status:** Implemented

## Summary

Port the Hashline editing technique (from oh-my-pi) to Hermes Agent. Each source line
is annotated with a 4-character content-derived hash anchor when reading files, enabling
the LLM to specify edit ranges using anchors instead of repeating context code
(old_string/new_string). This eliminates whitespace-sensitive string matching failures.

## Motivation

Traditional `str_replace` (fuzzy match) editing requires the model to reproduce surrounding
context verbatim. Even with 9-strategy fuzzy matching, whitespace and indentation
mismatches cause edit failures. Hashline removes this class of errors entirely by
decoupling edit targets from their textual content.

## Design Decisions

### Hash: 4-char base36 (SHA-256 truncated)

- **Why not oh-my-pi's BPE bigrams:** BPE bigrams are model-specific (cl100k/o200k/Claude);
  a generic approach works across all models.
- **Why 4 chars:** 36^4 = 1,679,616 distinct values. Birthday-paradox collision bound is
  ~1,296 lines. Files under 1000 lines have < 0.1% collision probability.
- **Why not 2 chars:** 36^2 = 1,296 values — collision at ~36 lines, unusable.

### Collision disambiguation

When two lines share the same hash (rare but possible), they get `#1`, `#2` suffixes.
This mirrors `format_hash_lines()` and `build_anchor_map()` using identical logic so
anchors from `read_file` resolve correctly in `patch_hashline`.

### Opt-in via parameter

- `read_file(hashline=True)` enables hashline output format.
- `patch(mode="hashline")` enables anchor-based editing.
- No global config flag — the parameter is the opt-in mechanism.

## Components

### `tools/line_hash.py` (new)

Pure computation module, no I/O.

| Function | Purpose |
|----------|---------|
| `compute_line_hash(line)` | 4-char base36 SHA-256 truncated hash |
| `format_hash_lines(lines, offset)` | Lines to `LINE:HASH\|content` format with collision disambiguation |
| `strip_hash_prefix(hashline)` | `LINE:HASH\|content` to raw content |
| `parse_anchor(tag)` | `"k7m2#1"` to `AnchorInfo(hash="k7m2", disambig=1)` |
| `parse_anchor_range(range)` | `"k7m2:a9f1"` to `("k7m2", "a9f1")` |
| `build_anchor_map(lines, offset)` | `{"k7m2": 1, "a9f1": 2, ...}` |
| `suggest_similar_anchors(target, map, lines)` | "Did you mean?" suggestions by hash prefix |

### `tools/file_operations.py` (modified)

- `FileOperations` ABC: added `patch_hashline(path, anchor_range, new_content)` abstract method.
- `ShellFileOperations`: implemented `patch_hashline` — reads file, builds anchor map,
  validates anchors (staleness guard), applies edit, writes back, generates diff, auto-lints.

### `tools/file_tools.py` (modified)

- `read_file_tool()`: added `hashline: bool = False` parameter; when true, re-formats
  content with `format_hash_lines` instead of plain line numbers.
- `patch_tool()`: added `mode="hashline"`, `anchor_range`, `new_content` parameters.
- `READ_FILE_SCHEMA`: added `hashline` boolean property.
- `PATCH_SCHEMA`: added `"hashline"` to mode enum, `anchor_range` and `new_content` properties.
- Dedup key: includes `hashline` flag to avoid format confusion.
- `_handle_read_file` / `_handle_patch`: pass new params through.

### `tests/tools/test_line_hash.py` (new)

Comprehensive unit tests for all functions in `line_hash.py`:
- `compute_line_hash`: determinism, length, charset, uniqueness, Unicode, whitespace sensitivity
- `format_hash_lines`: basic format, offset, collision disambiguation
- `strip_hash_prefix`: prefix removal, no-pipe fallback, content pipes
- `parse_anchor`: valid/invalid inputs, disambiguation
- `parse_anchor_range`: range, single, empty, malformed
- `build_anchor_map`: unique lines, duplicates, offset
- `suggest_similar_anchors`: prefix matching, no-match fallback

## Error Recovery

When an anchor is not found (stale content), `patch_hashline` returns:
1. The specific error (start/end anchor not found)
2. "Did you mean?" suggestions matching by hash prefix
3. Hint: "The file content may have changed since last read"

## Edit Operations

| Operation | anchor_range | new_content | Behavior |
|-----------|-------------|-------------|----------|
| Replace range | `"h1:h2"` | `"new code\n"` | Replace lines h1..h2 inclusive |
| Insert after | `"h1:h1"` | `"new line\n"` | Insert after line h1 |
| Delete range | `"h1:h2"` | `""` | Delete lines h1..h2 inclusive |
