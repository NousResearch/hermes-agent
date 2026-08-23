# Claude Code One-Shot for Crate Rewrites — 2026-06-26

## Pattern: `claude -p` for self-contained code rewrites

When a task is a self-contained single-file or single-module rewrite (not a multi-file refactor), `claude -p` one-shots are faster than a full delegate_task subagent. The key advantage: Claude Code reads the files itself (no context copies needed from the controller).

### Command shape

```bash
claude -p "prompt" --dangerously-skip-permissions --max-turns 30 --model sonnet \
  --add-dir /path/to/crate 2>&1 | tail -30
```

### When to use this pattern

- **Single-file rewrite**: Replace a module with a new implementation (FB2 → FibCodeWireV1, 124-line batch format → wire-based)
- **Self-describing task**: The prompt includes all requirements, the codebase has enough docs
- **Isolated scope**: Changes are contained to one file (or one crate with `--add-dir`)
- **Verification is mechanical**: `cargo test --features fib` either passes or it doesn't

### When NOT to use

- Multi-file refactors (use delegate_task with the full plan instead)
- Tasks requiring deep context in other crates (use delegate_task with `context=` field)
- Research/analysis tasks (claude -p is code-gen, not research)

### Applied in 2026-06-26 session

Dispatched `claude -p` for Phase 2 (FB2 → FibCodeWireV1 replacement in poly-kv/src/codec.rs). The agent:
1. Read the plan from `.ares/plans/`
2. Read the existing codec.rs file
3. Implemented new FIB_WIRE_BATCH_MAGIC ("FBWB") format
4. Kept legacy FB2 fallback for backward compat
5. Updated encode_batch_compact and decode_batch_compact paths
6. 3 test failures on first run (fixed in controller — test assertion updates)

### Pitfall: slow startup

Claude Code takes ~30-60 seconds to initialize (loading context, building LSP). For a 1.5-hour task this is fine. For a 5-minute fix, it's faster to do it in the controller. The threshold: if the fix is <10 lines, do it in controller. If it's >50 lines with multiple test interactions, dispatch.

### Verification after return

Always run `cargo test` in the controller after claude -p returns. Don't trust the agent's self-report. In this session, the agent reported success but 3 tests failed — all fixable with assertion updates.
