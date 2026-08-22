# mcp-unicode-sanitizer

Hermes MCP Gateway plugin that sanitizes every MCP tool description (and
`inputSchema` string surface) after the `tools/list` handshake, before it can
reach approval dialogs or model context. It implements the Unicode concealment
sanitization spec from **arXiv:2607.05744** — *Unicode TAG-Block Concealment
of Tool-Metadata Payloads in the Model Context Protocol*.

A malicious or compromised MCP server can hide prompt-injection instructions
in invisible Unicode (TAG-block, bidi overrides, zero-width characters). These
render as nothing in every terminal and chat UI — a human sees a clean tool
name — but the model's tokenizer decodes and forwards them verbatim. The plugin
closes that gap at the tool-metadata chokepoint, before metadata reaches
approval dialogs or model context.

## What it does

The plugin registers a single `sanitize_tool_metadata` hook. The MCP gateway
core (`tools/mcp_tool.py`) invokes that hook per tool at discovery time, with
a mutable copy of the tool definition. The handler:

1. Runs the vendored Unicode sanitization core over the whole tool
   (name + description + every `inputSchema` string surface).
2. Returns `{"tool": {...}}` containing the sanitized definition when safe.
3. Returns `{"quarantine": "reason"}` when the tool is not safe — the core
   then skips registering it (it is never delivered to approval dialogs or
   the model).

The handler is fail-closed and fail-safe: it never raises, never blocks the
gateway, and never lets a sanitized-dangerous tool through.

## Sanitization rules

| Rule | Transformation |
|------|----------------|
| 1 | NFC normalization (then NFKC for detection) |
| 2 | **Unicode TAG-block stripping (U+E0000–U+E007F)** — the primary fix |
| 3 | Bidi override removal |
| 4 | Invisible / zero-width character stripping (contextual for ZWJ/ZWNJ) |
| 5 | Confusable / homoglyph folding |
| 6 | Post-sanitization re-validation (fail-closed: flag on concealment *presence*) |
| 9 | Schema defaults are not consent — a sensitive-action keyword in a `default`/`enum` quarantines the tool |

Legitimate Unicode (emoji ZWJ sequences, Persian ZWNJ, non-Latin scripts,
diacritics) is preserved — ZWJ/ZWNJ are only stripped when they sit between two
ASCII letters (the keyword-splitting concealment pattern).

## Enabling

Plugins are opt-in. Add it to your allow-list:

```bash
hermes plugins enable mcp-unicode-sanitizer
# or edit ~/.hermes/config.yaml manually:
plugins:
  enabled:
    - mcp-unicode-sanitizer
```

## Configuration

Optional keys under `plugins.entries.mcp-unicode-sanitizer`:

| Key | Default | Effect |
|---|---|---|
| `quarantine_on_flag` | `true` | Drop tools whose residual text trips the keyword detector OR which carried concealment (TAG/bidi/invisible). When `false`, concealment-carrying tools are still sanitized (encoding stripped) but allowed through. |
| `log_level` | `"info"` | `"debug"` \| `"info"` \| `"warning"`. |

## Latency

Per-tool overhead is ~0.04ms, well within the 5ms budget enforced by
`tests/tools/test_mcp_unicode_sanitizer.py`.

## Tests

```bash
scripts/run_tests.sh tests/tools/test_mcp_unicode_sanitizer.py
```

## Attribution and licensing

The sanitization core in `sanitizer/` is the dependency-free (stdlib-only)
`mcp-unicode-sanitization` library (v1.0.0) implementing the spec derived from
arXiv:2607.05744. See `LICENSE` (MIT) and `NOTICE` in this directory.
