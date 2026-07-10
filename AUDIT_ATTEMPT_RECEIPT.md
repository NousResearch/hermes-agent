# External audit attempt receipt — 2026-07-10

## Scope and isolation

- Worktree: `/home/lfdm/worktrees/hermes-routing-audit-20260710`
- Audit branch: `audit/routing-isolation-20260710`
- Source baseline: `0da22bf07d9ffb8ebbd45504f6cff833935f4a76`
- Committed audit packet: `4b7ab479e9 docs: add isolated routing audit packet`
- Both auditors were launched in independent, detached tmux sessions. Neither had write, terminal, cron, Kanban, delegation, service-control, or configuration authority.

## Runs

| Auditor | Result file | Exit code | Terminal outcome |
|---|---|---:|---|
| Claude Max via `claude-delegate` | `audit-results/claude-max-audit.json` | 3 | `error_max_turns` after 12 turns; no final audit report returned |
| GLM 5.2 via `glm-code` | `audit-results/glm-5.2-audit.json` | 1 | `error_max_turns` after 12 turns; no final audit report returned |

Both processes exited normally after writing their status files. The failure was a bounded-agent exploration limit, not a transport interruption, permission denial, gateway restart, or worktree conflict.

## Evidence handling

- No source finding, root-cause theory, or patch proposal was produced by either auditor.
- The raw outputs contain only runner metadata and the `error_max_turns` result; they must not be interpreted as audit evidence.
- The same failed broad audit will not be retried a third time.

## Next method

Perform a direct, bounded local source trace against the immutable baseline, beginning with the actual delivery-owner and target-resolution symbols. Any implementation remains prohibited until a concrete regression test has been designed and demonstrated RED.
