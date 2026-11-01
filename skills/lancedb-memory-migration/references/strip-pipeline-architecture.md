# Strip Pipeline Architecture — 2026-05-17

## What "stage" Means Here

In `optimize_lance_memory_v2.py`, a **Stage** is a single processing step in the `StripPipeline`. Each Stage is:
- **Independent** — configured as a dict, toggled by `enabled: true/false`
- **Composable** — stages run sequentially, output of one feeds into next
- **Pluggable** — adding a new Stage type requires only adding a dict; no code changes needed

This is NOT the frontmatter `metadata.hermes.stage` field — that's a different concept (skill authoring lifecycle stages).

## Stage Configuration Shape

```python
{
    "id": "unique_stage_id",
    "name": "Human-readable name",
    "enabled": True,
    "type": "regex",          # "regex" or "quality"
    "action": "strip_before", # see Actions below
    "patterns": [...],        # list of regex strings or (pattern, replacement) tuples
    "flags": re.DOTALL | re.IGNORECASE,
    "stop_on_first": True,     # True = stop on first match; False = process all
}
```

## Actions in `_run_regex_stage`

| Action | Code | Use Case |
|--------|------|----------|
| `strip_before` | `text = text[m.end():]` | Remove match + everything before it (e.g., `[IMPORTANT:...]` frontmatter) |
| `strip_after_match` | `text = text[:m.start()] + text[m.end():]` | Remove match only, preserve prefix (e.g., keep `[assistant]\n`, strip following frontmatter) |
| `strip_after` | `text = text[:last.start()]` | Remove everything from last match to end of text |
| `replace` | `text = compiled.sub(repl, text)` | Substitute matched content with literal string |

### `strip_after_match` — The Critical One

Used when you need to remove something from the **middle** of the text while preserving the prefix. The pattern matches from the prefix to the end of the unwanted section. `text[:m.start()]` gets the prefix, `text[m.end():]` gets the remainder after the unwanted section.

**Example:** Removing skill frontmatter from `[assistant]\n\n## Overview\n## When to Use\nReal content`:
- Pattern: `\[assistant\]\n+(## [^#\n]{3,100}\n+){1,10}(?=\S)`
- `m.start() = 0`, `m.end() = 43` (position after `## Overview\n## When to Use\n`)
- Result: `text[:0] + text[43:]` = `Real content`
- But the actual implementation uses `text[:m.start()] + text[m.end():]` = `[assistant]\n` + `Real content` — preserving the `[assistant]` label so `last_pair_only` extraction works correctly.

### Pattern Design for `strip_after_match`

The pattern must match **from the prefix you want to keep** (e.g., `[assistant]`) **to the end of the unwanted section**. `m.start()` = prefix end, `m.end()` = remainder start.

Common pattern: `r'\[assistant\]\n+(## [^#\n]{3,100}\n+){1,10}(?=\S)'`
- Matches: `[assistant]\n` + ≥1 `##` section header lines + first real content line
- `m.start()` = 0 (start of `[assistant]`)
- `m.end()` = position after last `##` header, before first real paragraph

### Lookbehind Pitfall

Python's `re` module does NOT support variable-width lookbehind (`(?<=prefix.*)`). This caused the initial implementation to fail. Workaround:
- Use a plain pattern starting from the prefix: `r'\[assistant\]\n+(## title\n+){1,10}(?=\S)'`
- The action (`strip_after_match`) automatically handles `text[:m.start()]` to preserve the prefix

## Stage Processing Order

The order matters because later stages depend on the output of earlier ones:

1. `prefix` → strips `[IMPORTANT:...]---<yaml>---` frontmatter from the front of the text
2. `assistant_frontmatter` → strips `## Overview\n## When to Use\n...` section headers that remain in `[assistant]` blocks (needs `stop_on_first=True` to not match in the middle)
3. `trailing` → strips end-of-document markers (with lookahead `(?=[\n\s]*$)` to avoid matching body separators)
4. `embedded_meta` → replaces UUIDs and timestamps
5. `quality_gate` → filters out short/low-quality content

## `_estimate_clean_len()` — Correcting `strip_ratio`

After `split_into_twigs()`, each Twig's `len_clean = len(part)` — it does NOT reflect frontmatter that was stripped. `extract_meta()` calls `_estimate_clean_len()` to detect residual frontmatter patterns in the Twig content and correct `clean_len` / `strip_ratio`.

The estimator scans for three patterns:
1. `[IMPORTANT:...]` blocks
2. `---yaml---` frontmatter blocks  
3. `[assistant]` followed by ≥1 `##` section headers

If any are found, it subtracts their lengths from `len(content)` to get a more accurate estimate.