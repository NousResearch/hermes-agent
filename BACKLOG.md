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

## Stage 5 precedence default (2026-04-29)

`/tmp/precedence-test-result.txt` was missing or unresolved during unattended Stage 5. Defaulted to the user-global `~/.claude/agents/` overlay strategy per operator instruction.

## /escalate smoke test deferred (2026-04-29)

Deferred until full provider setup is complete. Command contract installed, but multi-family fan-out and cost tracking still need a live-provider smoke test.

## tdd-guard hook deferred (2026-04-29)

Stage 7 calls for a tdd-guard hook in dry-run for the first two sessions, but no concrete local tdd-guard command or package was present in the plan. Left as BACKLOG tuning item rather than installing an unknown hook.

## Overlay follow-up backlog

- Anthropic-coupled MCP/security-review picks: replace with provider-neutral tools if migrating off Anthropic.
- /escalate convergence detection v2: embedding similarity and LLM judge.
- scripts/sync-from-cdb.sh: formalize CDB to Hermes methodology sync.
- Chen v83 model-agnostic ratification: required if CDB updates v83.
- tdd-guard tuning after first two dry-run sessions.
- LiteLLM integration for unified billing/auth across providers.
- Quarterly retention review for `audits/` folder growth.
- Provider-cost dashboard for monthly role spend.

## Stage 8 verification drift (2026-04-29)

Final verification found `41 failed, 17690 passed, 56 skipped, 1 error`, versus the accepted Stage 1 baseline of `36 failed, 17695 passed, 56 skipped`. Treat this as an operator-attention item before relying on the overlay for CC hardening: either accept the new 41-failure baseline explicitly or investigate the five-failure drift.

Additional verification gaps to resolve:
- `pre-commit` is not installed in the WSL environment.
- Dashboard backend smoke returned connection refused on port 9119.
- Runtime skills smoke using `./hermes -c '/skills'` failed; verify the correct Hermes command-mode invocation.
- `/tmp/precedence-test-result.txt` was missing during final verification after Stage 5 had defaulted to user-global precedence.
- Model-agnostic grep fails only in vendored `docs/reference/claude-code-system-prompts/`; decide whether to exclude vendored reference docs from that gate.
