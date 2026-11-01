# hashline-guard

Strict-match and content-addressed patch anchors for the `patch` tool. Blocks
stale or ambiguous `old_string` anchors before the patch is applied, preventing
silent drift when a file has changed under the agent, and provides an
`anchored_patch` tool that pins edits to a SHA-256 hashline of the anchor plus
its surrounding context.

## What it does

Registers a `pre_tool_call` hook. When a `patch` call with `mode=replace`
targets a file, it verifies the `old_string` anchor is present **exactly once**
in the live file:

- **Exactly once** -> allow the patch.
- **Zero times** -> block: `old_string not found in live file — anchor drifted`.
- **Two or more** -> block: `old_string is ambiguous: found N times in live file`.
- **Empty old_string** -> block: `old_string must be non-empty`.
- **Missing file** -> block: `file not found: <path>`.

Blocked calls return `{'action': 'block', 'message': ...}` so the agent re-reads
the file and re-issues the patch against the current content.

The plugin also registers two tools:

- `anchored_patch` — apply a `mode=replace` patch pinned by `verify_anchor_by_hash`.
  The expected hashline selects the exact occurrence even when `old_string`
  appears multiple times. Read -> verify -> apply -> write is a single atomic
  handler operation.
- `hashline_compute` — return the current hashline for every occurrence of an
  `old_string` (with line numbers and context snippets) so the agent can
  discover the value to pin.

## Trigger conditions

- Tool: `patch`
- Mode: `replace`
- Requires both `path` and `old_string` in `function_args`

V4A multi-file patches (`mode=apply` / traversal patches) are skipped — those
have their own traversal checks. Non-patch tools are always skipped.

## Canonicalization policy

- Newline normalization: `CRLF`/`LFCR`/`CR` -> `LF` before matching/hashing.
- Trailing whitespace: kept byte-exact on each line.
- Versioned payload: `hashline:v1:<index>:<windowed_text>` so future format
  changes don't collide with prior hashline values.
- Window size is part of the computed hash shape; changing `window` changes the
  hash.

## Fail-open policy

Any unexpected error (IO error, permission, encoding) is logged at debug level
and the patch proceeds — the hook never blocks on infrastructure problems.
The only hard blocks are the verified anchor conditions above.

## Files

- `src/hashline_core.py` — `verify_anchor()`, `compute_hashline()`,
  `verify_anchor_by_hash()`, `find_all()`, `context_hash()`
- `__init__.py` — plugin entry: `register(ctx)` registers the `pre_tool_call`
  hook plus the `anchored_patch` and `hashline_compute` tools
- `tests/test_verify_anchor.py` — exact-match guard + hashline pinning + CRLF
  canonicalization
- `tests/test_anchored_patch.py` — anchored_patch handler behavior + atomicity
- `tests/test_e2e_anchored_patch.py` — end-to-end probe (pin -> stale-pin
  block -> re-pin -> wrong-hash block -> CRLF)

Note: the plugin directory name contains a hyphen, so `__init__.py` loads
`hashline_core` via `importlib.util.spec_from_file_location` rather than a
relative import.

## Example

```json
{
  "tool": "anchored_patch",
  "args": {
    "path": "src/engine/verbs/special.py",
    "old_string": "sp = sp + 1\n",
    "new_string": "sp += min(6, max_sp - sp)\n",
    "expected_hashline": "<computed hashline:v1 SHA-256 hex>",
    "window": 2
  }
}
```

Discover the pin with `hashline_compute`, then apply with `anchored_patch`.
If the file drifted, `anchored_patch` returns the found hashlines and line
numbers so the agent can re-pin in one turn.

## Rollback

```bash
hermes plugins disable hashline-guard
```
