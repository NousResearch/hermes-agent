# Mixed Claude Code model routing for embedded cluster implementation

Use when implementing a multi-phase plan with Claude Code where the user explicitly requests different Claude models by task difficulty.

## User preference captured

For Josh's ESP32-S3 / TinyStories cluster work:
- Use Claude Code for implementation when available.
- Use Fable for actually difficult sections if the local Claude Code account exposes that model alias.
- Use Sonnet or Opus for routine/mechanical sections.
- If Fable is unavailable, fall back to Opus with maximum effort.
- The controller still owns verification; Claude Code self-reports are not receipts.

## Difficulty routing

Routine / Sonnet:
- repo cleanup
- benchmark harness changes
- packet encode/decode tests
- role build flags
- docs and receipt scaffolding
- mechanical Python scripts

Opus:
- architecture review
- protocol correctness review
- model shard exporter design
- final claim-boundary review
- integration failure diagnosis

Fable if available, otherwise Opus max effort:
- custom Xtensa SIMD kernel changes
- ACCX/register/pipeline hazards
- int4 sharded matmul correctness
- tensor-parallel synchronization design
- transformer runtime and attention sharding
- TinyStories-33M exact memory/shard feasibility

## Controller discipline

Before dispatch:
1. Verify `claude auth status --text` is logged in.
2. Test model aliases with a tiny dry run if using nonstandard names like `fable`.
3. Put one task in one prompt; do not hand Claude Code an entire 7-phase plan.
4. Include exact files and verification commands in the task spec.

After every Claude Code task:
1. Run the build/tests locally in the controller session.
2. For hardware tasks, run the board and capture receipts directly.
3. Commit only after controller verification passes.
4. Record model used, command/log path, and verification output.

Do not claim Claude Code was used if auth failed or no command ran. In that case, prepare routing docs/specs only and report the auth blocker.
