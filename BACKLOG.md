# BACKLOG

## Stage 1 baseline - accepted with documented failures (2026-04-29)

Pinned upstream: 258449c468166f404ae28f97f6e4157ff72d0893
Test result: 36 failed, 17695 passed, 56 skipped
Failure clusters: WhatsApp identifier normalization, Slack DM progress
callbacks, TUI gateway protocol, ACP file reading, cache concurrency,
MCP structured content. None impact operator's intended use case
(coding agent invocation, skill execution, /escalate).

Walk-back to find green commit attempted; abandoned per operator
direction. Failures accepted as preexisting upstream baseline.
Future test runs check for NEW failures only.

Commit-prefix judgment call: the operator-provided shell snippet used
`docs(backlog)`, but the mandatory overlay prefix scheme does not include
that scope. Used `docs(claude)` instead to preserve the scheme.

## Chen v83 model-reference scan (2026-04-29)

`prompts/superprompts/chen-audit-protocol.md` is verbatim and must not be modified in this repo. The model-name grep found references; v83 needs an operator-ratified model-agnostic update in CDB before any future sync changes this file.
Extracted from: C:/Users/ptann/OneDrive/Work/motion granted/Main System/Motion-Granted-Production/.claude/skills/chen/chen-superprompt-v83.md
You do not care whether a defect came from the operator, from Claude, from a prior Chen session, from a spec, or from a "binding decision." If it is wrong, unsafe, unverified, incomplete, contradictory, or misleading, you flag it.
Claude and Chen are not truly independent minds. They share the same model family, similar priors, similar completion bias, similar tendency to smooth contradictions, and similar vulnerability to plausible but unverified narratives.
A prior Claude answer is not enough.
You are operating in Claude Code with repository access.
6. **Historical handoffs, summaries, prior Claude outputs, comments, assumptions**
Read code first. Determine what the code actually does before loading specs, prior handoffs, summaries, architecture prose, or prior Claude outputs.
Use when comparing implementation to architecture, handoffs, task lists, binding decisions, or prior Claude outputs.
