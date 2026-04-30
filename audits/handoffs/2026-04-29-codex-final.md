# Codex final handoff - Hermes methodology overlay

Date: 2026-04-29
Branch: `overlay/cdb-coding-agent-v0`
Accepted upstream pin: `258449c468166f404ae28f97f6e4157ff72d0893`
HEAD before this final handoff commit: `1a50b069f5da32c694e6f4770cc8c8befb4e436a`
Expected total overlay commits after this handoff commit: `30`

This handoff is committed as part of the final pushed SHA, so the exact final SHA is reported in the operator-facing Codex final response after commit creation.

## Stage completion

| Stage | Result | Notes |
|---|---|---|
| Stage 0 | Completed earlier | WSL repo and source staging established. Original operator gates were overridden by operator instruction to continue unattended. |
| Stage 1 | Accepted red baseline | Operator accepted pinned upstream despite test failures. `BACKLOG.md` records the accepted baseline as preexisting upstream failure clusters. |
| Stage 2 | Completed | Installed `AGENTS-OVERLAY.md` and cross-linked from upstream `AGENTS.md`. |
| Stage 3 | Completed with deferred manual review | Installed `.claude/`, personas, Chen v83, docs, workflows, top-level methodology, and `.codex/`. Chen v83 remained verbatim. Manual model-name review deferred to `audits/handoffs/2026-04-29-manual-review-deferred.md`. |
| Stage 4 | Completed | Installed 28 methodology skills under `.claude/skills/`; body model-name refs were role-refactored. |
| Stage 5 | Completed with default | Added Hermes-specific CLAUDE guidance, role registry, model application script, Claude settings/launch/rules/pre-commit/cost/doc-routing. Precedence result was missing; defaulted to user-global strategy and recorded in `BACKLOG.md`. |
| Stage 6 | Completed | Added legal-tech-coding patterns, reviewer persona, always-on awareness skills, Chen sidecar, and private-skills integration contract. |
| Stage 6.5 | Completed with smoke deferred | Installed `/escalate`; live multi-family smoke test deferred until provider setup is complete and recorded in `BACKLOG.md`. |
| Stage 7 | Completed with blocker surfaced | Installed MCP placeholders, Codex MCP reference, env placeholders, AI security workflow, repomix script, and vendored Claude Code system prompt reference. Trail of Bits skills clone hung; blocker recorded in `OPERATOR-INBOX/2026-04-29-blocker-trailofbits-skills.md`. |
| Stage 8 | Completed, verification red | Verification matrix ran. Several checks require interactive/operator validation; full tests are red. Details below. |
| Stage 9 | Partially completed by instruction | Added `INSTALL.md`, `docs/provider-migration.md`, and `docs/clay-onboarding.md`. Did not create PR or merge. |

## Judgment-call deviations

- Stage 1 red baseline was accepted per operator instruction instead of halting on the original gate.
- Stage 3 manual review was deferred; raw model-name scan output was documented instead of stopping for line-by-line operator review.
- Stage 5 precedence test defaulted to user-global because `/tmp/precedence-test-result.txt` was unavailable.
- Stage 6.5 live `/escalate` smoke test was skipped until full provider setup is ready.
- Stage 7 MCP server connectivity was not validated because committed config intentionally uses environment placeholders.
- Trail of Bits security skills were not installed because the clone operation hung; the blocker was surfaced and the run continued.
- `web/package-lock.json` was touched by `npm install` during verification and restored because it was validation noise, not an overlay change.
- No PR was created and no merge was performed per operator instruction.

## Verification matrix

| # | Check | Result | Notes |
|---|---|---|---|
| 1 | Hermes CLI launches | PASS | `./hermes --help` printed usage. |
| 2 | Runtime skills load | FAIL | `./hermes -c '/skills'` returned `No session found matching '/skills'`; likely command-mode invocation mismatch. |
| 3 | Methodology skills not in runtime | PASS | Grep returned empty. |
| 4 | Chen activates | NOT RUN | Requires interactive Claude Code. |
| 5 | Architect owns swarm | NOT RUN | Requires interactive Claude Code. |
| 6 | legal-tech-reviewer activates | NOT RUN | Requires interactive Claude Code persona activation. |
| 7 | legal-tech-context skill fires | NOT RUN | Requires Claude Code skill auto-invoke. |
| 8 | canonical-mutation-gate fires | NOT RUN | Requires Claude Code skill auto-invoke. |
| 9 | Cross-reference graph sanity | PASS | File-existence sanity check clean. |
| 10 | Chen v83 untouched | PASS | `wc -l` returned `778`; file was not modified. |
| 11 | Both AGENTS files exist | PASS | `AGENTS.md` and `AGENTS-OVERLAY.md` present. |
| 12 | Hermes AGENTS.md unchanged except footer | PASS | Diff only shows overlay footer. |
| 13 | Tests pass | FAIL | `41 failed, 17690 passed, 56 skipped, 1 error`; accepted Stage 1 baseline was `36 failed`. This drift needs operator decision. |
| 14 | Ruff passes | PASS | `ruff check .` passed with no Python files found under given paths. |
| 15 | Pre-commit installed | FAIL | `pre-commit` command unavailable. |
| 16 | Dead-code hook blocks | NOT RUN | Requires Claude Code hook execution. |
| 17 | Async ruff hook runs | NOT RUN | Requires Claude Code hook execution. |
| 18 | Dashboard backend | FAIL | Curl to port 9119 returned connection refused. |
| 19 | Dashboard frontend build | PASS | `web` build completed; large chunk warning only. |
| 20 | MCP servers in Claude Code | NOT RUN | Requires Claude Code `/mcp` with credentials. |
| 21 | MCP servers in Codex | PASS | `codex mcp list` showed all five placeholder servers enabled. |
| 22 | AI Security Review action | NOT RUN | Requires draft PR event. |
| 23 | Multi-model passthrough | NOT RUN | Kimi doctor connectivity confirmed earlier; full Hermes model invocation not run. |
| 24 | Codex CLI compat | PASS | `codex-cli 0.125.0`. |
| 25 | Clay-machine reproducibility | NOT RUN | Requires Clay's machine. |
| 26 | Model-agnostic mandate | FAIL | Hits are in vendored `docs/reference/claude-code-system-prompts/`; methodology files outside that reference tree were role-refactored. |
| 27 | Role-registry swap test | NOT RUN | Would write per-machine `~/.claude/agents/`. |
| 28 | Provider-migration dry run | NOT RUN | Requires operator-selected alternate provider. |
| 29 | Precedence test recorded | FAIL | `/tmp/precedence-test-result.txt` missing at final verification. |
| 30 | Output-routing hook blocks forbidden | NOT RUN | Requires Claude Code hook execution. |
| 31 | OPERATOR-INBOX scaffolding | PASS | Inbox and audit folders present. |
| 32 | Cost-snapshot script | PASS | Fails closed with missing local cost config, as designed. |
| 33 | No hardcoded home paths in scripts | PASS | Grep returned empty. |
| 34 | 8 rules files | PASS | Exactly eight `.claude/rules/*.md` files present. |
| 35 | Repomix pack | PASS | Generated non-empty `/tmp/hermes-overlay.xml`; repomix excluded suspicious files. |

Raw verification log: `/tmp/verification-matrix-results.md`
Full test log: `/tmp/final-run-tests.log`

## Operator attention before CC hardening

1. Decide whether to accept the final `41 failed` test state as the new overlay baseline or investigate the five-failure drift from the accepted `36 failed` baseline.
2. Install `pre-commit` in WSL before relying on the pre-push hooks.
3. Re-run the dashboard backend smoke with the correct service command and inspect why port 9119 did not come up.
4. Re-run/record the Claude Code precedence test; unattended Stage 5 defaulted to user-global.
5. Decide whether vendored reference docs under `docs/reference/claude-code-system-prompts/` are exempt from the model-agnostic grep gate.
6. Complete live interactive checks: Chen activation, architect dispatch, legal-tech reviewer/skills, output-routing hooks, MCP in Claude Code, and `/escalate` smoke.
7. Resolve or intentionally defer the Trail of Bits skills blocker in `OPERATOR-INBOX/2026-04-29-blocker-trailofbits-skills.md`.

## Current readiness

The methodology overlay is installed and pushed on the branch. It is ready for the CC hardening prompt as long as the operator treats the verification failures above as known red items, especially the 41-test final state.
