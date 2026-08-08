#!/usr/bin/env bash
# ruflo-swarm.sh — Generate a ready-to-use swarm invocation for the agent.
# Works in both modes:
#   Mode A: prints the ruflo__swarm_init MCP call for the agent to execute
#   Mode B: prints a delegate_task payload using npx coding agents
# Usage: ./ruflo-swarm.sh "<goal>" [swarm-type] [--mode a|b]
# Swarm types: coding (default), review, test, full, security

set -euo pipefail

GOAL="${1:-}"
SWARM_TYPE="${2:-coding}"
MODE="${3:-auto}"

if [[ -z "$GOAL" ]]; then
  echo "Usage: $0 <goal> [swarm-type] [--mode a|b]" >&2
  echo "Swarm types: coding (default), review, test, full, security" >&2
  exit 1
fi

case "$SWARM_TYPE" in
  coding)   AGENTS="coder:2,reviewer:1,tester:1,architect:1" ;;
  review)   AGENTS="reviewer:3,security:1,architect:1" ;;
  test)     AGENTS="tester:3,coder:1" ;;
  full)     AGENTS="coder:2,reviewer:2,tester:2,architect:1,security:1,docs:1" ;;
  security) AGENTS="security:3,coder:1,reviewer:1" ;;
  *)
    echo "Unknown swarm type: $SWARM_TYPE" >&2
    echo "Valid: coding, review, test, full, security" >&2
    exit 1
    ;;
esac

# Auto-detect mode: Ruflo CLI present → prefer Mode A
if [[ "$MODE" == "auto" ]]; then
  if command -v ruflo >/dev/null 2>&1; then
    MODE="a"
  else
    MODE="b"
  fi
fi

echo "# Swarm: $SWARM_TYPE ($AGENTS)"
echo "# Goal: $GOAL"
echo ""

if [[ "$MODE" == "a" ]]; then
  cat <<EOF
## Mode A — ruflo__swarm_init (MCP)

Call this MCP tool (the agent executes it, do NOT paste raw):

\`\`\`json
{
  "tool": "ruflo__swarm_init",
  "params": {
    "topology": "hierarchical",
    "maxAgents": 5,
    "strategy": "specialized",
    "config": {
      "goal": "$GOAL"
    }
  }
}
\`\`\`

Then track with ruflo__swarm_status / swarm_health, and ruflo__swarm_shutdown when done.
EOF
else
  cat <<EOF
## Mode B — Hermes-native (Ruflo not installed)

Use delegate_task with these parallel tasks:

\`\`\`
delegate_task(tasks=[
  {goal: "Implement: $GOAL", context: "Run in repo worktree A: git worktree add ../wt-a main"},
  {goal: "Review + test the result", context: "Run in worktree B"},
])
\`\`\`

Or single long-running agent:

\`\`\`
terminal(background=true, notify_on_complete=true,
  command="cd <repo> && npx claude-code -p '$GOAL'")
\`\`\`

Store results in shared memory afterwards:
\`\`\`
memory(action="add", target="swarm-outcome", content="<what was built>")
tdai_conversation_search(query="swarm-outcome")   # verify all profiles can see it
\`\`\`
EOF
fi
