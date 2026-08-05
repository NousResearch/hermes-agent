# hashline-guard

Strict-match pre-check for `patch` tool `old_string` anchors. Blocks stale or
ambiguous patch anchors before the patch is applied, preventing silent drift
when a file has changed under the agent.

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

## Trigger conditions

- Tool: `patch`
- Mode: `replace`
- Requires both `path` and `old_string` in `function_args`

V4A multi-file patches (`mode=apply` / traversal patches) are skipped — those
have their own traversal checks. Non-patch tools are always skipped.

## Fail-open policy

Any unexpected error (IO error, permission, encoding) is logged at debug level
and the patch proceeds — the hook never blocks on infrastructure problems.
The only hard blocks are the verified anchor conditions above.

## Files

- `hashline_core.py` — `verify_anchor(file_text, old_string)`, `context_hash(file_text, old_string, window=2)`
- `__init__.py` — plugin entry: `register(ctx)` -> `ctx.register_hook('pre_tool_call', on_pre_tool_call)`
- `test_verify_anchor.py` — regression tests (RED->GREEN verified)

Note: the plugin directory name contains a hyphen, so `__init__.py` loads
`hashline_core` via `importlib.util.spec_from_file_location` rather than a
relative import.

## Rollback

```bash
env -u HERMES_DELEGATED_CHILD_CONTEXT HERMES_KANBAN_BOARD=hashline-guard hermes plugins disable hashline-guard
```

## Future

`context_hash()` provides the foundation for a full-hashline primitive:
SHA-256 of the anchor plus up to `window` surrounding lines, usable for
content-addressed patch anchors and drift detection beyond exact-match counts.
