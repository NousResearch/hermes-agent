# Hermes Operator Gate

Purpose: make Hermes maintenance start from live runtime truth, an isolated
worktree, and a small profile eval gate.

## Boundary

- Live Sawyer gateway root: `/Users/sawbeck/.hermes/hermes-agent`
- Sawyer profile state: `/Users/sawbeck/.hermes/profiles/sawyer`
- Do not assume `/Users/sawbeck/Projects/hermes-agent` is live.
- Do not patch `main` for non-trivial work. Use `.worktrees/codex-*`.

## Required Sequence

1. Run repo inventory from the live root:
   `python3 /Users/sawbeck/.codex/skills/repo-dev-setup/scripts/repo_inventory.py "$(git rev-parse --show-toplevel)" --report`
2. If changing agent, skill, workflow, MCP, or command surfaces, run:
   `python3 /Users/sawbeck/.codex/skills/agent-surface-audit/scripts/audit_agent_surfaces.py "$(git rev-parse --show-toplevel)"`
3. Create an isolated worktree:
   `git worktree add .worktrees/codex-<task> -b codex/<task>`
4. Capture live runtime status before editing:
   `python3 scripts/operator_status_receipt.py --profile sawyer --since "YYYY-MM-DD HH:MM:SS"`
5. Run the small operator eval gate before and after meaningful Slack, memory,
   tool, or wrapper changes:
   `python3 scripts/operator_eval_gate.py --profile-root /Users/sawbeck/.hermes/profiles/sawyer`
6. Verify with targeted tests through `scripts/run_tests.sh`.
7. Only restart the live gateway after the branch payload is verified and the
   intended runtime root is confirmed.

## Acceptance Gate

A Hermes operator change is not complete until it has:

- an isolated worktree branch
- targeted tests
- a status receipt showing the live gateway root and PID
- an eval receipt for the relevant Sawyer profile batch, when the change affects
  Slack behavior, memory, tools, wrappers, or answer shaping
- a final git status that separates intended code changes from runtime residue

## Why This Exists

Hermes has multiple local checkouts and runtime/profile surfaces. The failure
to prevent is simple: patching the wrong checkout or trusting a clean code diff
while the live gateway is running somewhere else.
